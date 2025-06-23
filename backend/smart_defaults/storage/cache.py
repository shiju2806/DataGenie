"""
Smart Defaults Cache Layer
Redis-based caching with in-memory fallback and comprehensive error handling
"""

import asyncio
import logging
import json
import pickle
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Dict, List, Union, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass

# Handle Redis import gracefully
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    # Mock Redis classes for when redis is not installed
    class Redis:
        pass
    class ConnectionError(Exception):
        pass
    class TimeoutError(Exception):
        pass
    class RedisError(Exception):
        pass
    REDIS_AVAILABLE = False

# Import your models - handle both direct execution and module imports
try:
    from ..models.user_profile import UserProfile, UserPreferences
    from ..models.data_source import DataSource, SourceMetadata
    from ..models.recommendation import Recommendation
except ImportError:
    # For direct execution, create mock classes
    from typing import Any
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "test_role"
        preferences: Any = None

    @dataclass
    class UserPreferences:
        auto_connect_threshold: float = 0.85

    @dataclass
    class DataSource:
        id: str = "test_source_id"
        name: str = "Test Source"

    @dataclass
    class SourceMetadata:
        pass

    @dataclass
    class Recommendation:
        id: str = "test_rec_id"
        user_id: str = "test_user"

logger = logging.getLogger(__name__)

class CacheConfig:
    """Cache configuration with Redis and fallback settings"""

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 redis_db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 10,
                 socket_timeout: float = 5.0,
                 socket_connect_timeout: float = 5.0,
                 default_ttl: int = 3600,  # 1 hour default
                 use_memory_fallback: bool = True,
                 memory_max_size: int = 1000,  # Max items in memory cache
                 retry_attempts: int = 3,
                 retry_delay: float = 0.1):
        self.redis_url = redis_url
        self.redis_db = redis_db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.default_ttl = default_ttl
        self.use_memory_fallback = use_memory_fallback
        self.memory_max_size = memory_max_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

class MemoryCache:
    """In-memory cache fallback with LRU eviction"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            entry = self.cache[key]

            # Check expiration
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                self.delete(key)
                return None

            return entry['value']
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache with optional TTL"""
        try:
            # Evict old entries if at max size
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            expires_at = None
            if ttl:
                expires_at = time.time() + ttl

            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()
            return True

        except Exception as e:
            logger.error(f"❌ Memory cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        if key in self.cache:
            del self.cache[key]
            self.access_times.pop(key, None)
            return True
        return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        return True

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return

        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(lru_key)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = 0
        expired_count = 0
        current_time = time.time()

        for entry in self.cache.values():
            if entry.get('expires_at') and current_time > entry['expires_at']:
                expired_count += 1
            # Rough size estimation
            total_size += len(str(entry['value']))

        return {
            'total_keys': len(self.cache),
            'expired_keys': expired_count,
            'max_size': self.max_size,
            'estimated_size_bytes': total_size
        }

class CacheManager:
    """Async cache manager with Redis primary and memory fallback"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self.memory_cache = MemoryCache(config.memory_max_size) if config.use_memory_fallback else None
        self._initialized = False
        self._redis_available = False

    async def initialize(self):
        """Initialize Redis connection with fallback handling"""
        if self._initialized:
            return

        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis not installed - using memory-only cache")
            self._redis_available = False
            self._initialized = True
            return

        try:
            # Create Redis connection
            self.redis_client = redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=True,
                decode_responses=False  # We'll handle encoding ourselves
            )

            # Test connection
            await self.redis_client.ping()
            self._redis_available = True
            logger.info("✅ Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using memory fallback.")
            self._redis_available = False
            self.redis_client = None

        self._initialized = True

    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔐 Redis connections closed")

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage (handles dataclasses, dicts, etc.)"""
        try:
            if is_dataclass(value):
                # Convert dataclass to dict, then to JSON
                data = {'_type': 'dataclass', '_class': value.__class__.__name__, '_data': asdict(value)}
                return json.dumps(data, default=str).encode('utf-8')
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                # JSON-serializable types
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Fallback to pickle for complex objects
                data = {'_type': 'pickle', '_data': pickle.dumps(value).hex()}
                return json.dumps(data).encode('utf-8')
        except Exception as e:
            logger.error(f"❌ Serialization error: {e}")
            raise

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            json_data = json.loads(data.decode('utf-8'))

            if isinstance(json_data, dict) and '_type' in json_data:
                if json_data['_type'] == 'dataclass':
                    # For now, return the data dict - in production you'd reconstruct the dataclass
                    return json_data['_data']
                elif json_data['_type'] == 'pickle':
                    return pickle.loads(bytes.fromhex(json_data['_data']))

            return json_data

        except Exception as e:
            logger.error(f"❌ Deserialization error: {e}")
            return None

    async def _redis_operation_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute Redis operation with retry logic"""
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except (ConnectionError, TimeoutError, RedisError) as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.warning(f"⚠️ Redis operation failed after {self.config.retry_attempts} attempts: {e}")
                    self._redis_available = False

        raise last_error

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (Redis first, then memory fallback)"""
        if not self._initialized:
            await self.initialize()

        # Try Redis first
        if self._redis_available and self.redis_client:
            try:
                data = await self._redis_operation_with_retry(self.redis_client.get, key)
                if data:
                    value = self._deserialize_value(data)
                    # Also cache in memory for faster access
                    if self.memory_cache:
                        self.memory_cache.set(key, value, ttl=300)  # 5 min in memory
                    return value
            except Exception as e:
                logger.warning(f"⚠️ Redis get failed for key {key}: {e}")
                self._redis_available = False

        # Fallback to memory cache
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value

        return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (Redis primary, memory fallback)"""
        if not self._initialized:
            await self.initialize()

        ttl = ttl or self.config.default_ttl
        success = False

        # Try Redis first
        if self._redis_available and self.redis_client:
            try:
                serialized_value = self._serialize_value(value)
                await self._redis_operation_with_retry(self.redis_client.setex, key, ttl, serialized_value)
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Redis set failed for key {key}: {e}")
                self._redis_available = False

        # Always set in memory cache as backup
        if self.memory_cache:
            memory_success = self.memory_cache.set(key, value, ttl)
            success = success or memory_success

        return success

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._initialized:
            await self.initialize()

        success = False

        # Delete from Redis
        if self._redis_available and self.redis_client:
            try:
                result = await self._redis_operation_with_retry(self.redis_client.delete, key)
                success = bool(result)
            except Exception as e:
                logger.warning(f"⚠️ Redis delete failed for key {key}: {e}")
                self._redis_available = False

        # Delete from memory cache
        if self.memory_cache:
            memory_success = self.memory_cache.delete(key)
            success = success or memory_success

        return success

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._initialized:
            await self.initialize()

        # Check Redis first
        if self._redis_available and self.redis_client:
            try:
                result = await self._redis_operation_with_retry(self.redis_client.exists, key)
                return bool(result)
            except Exception as e:
                logger.warning(f"⚠️ Redis exists failed for key {key}: {e}")
                self._redis_available = False

        # Check memory cache
        if self.memory_cache:
            return self.memory_cache.get(key) is not None

        return False

    async def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries (optionally by pattern)"""
        if not self._initialized:
            await self.initialize()

        success = False

        # Clear Redis
        if self._redis_available and self.redis_client:
            try:
                if pattern:
                    # Get keys matching pattern and delete them
                    keys = await self._redis_operation_with_retry(self.redis_client.keys, pattern)
                    if keys:
                        await self._redis_operation_with_retry(self.redis_client.delete, *keys)
                else:
                    await self._redis_operation_with_retry(self.redis_client.flushdb)
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Redis clear failed: {e}")
                self._redis_available = False

        # Clear memory cache (pattern matching not implemented for simplicity)
        if self.memory_cache:
            if not pattern:  # Only full clear for memory cache
                memory_success = self.memory_cache.clear()
                success = success or memory_success

        return success

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment a counter in cache"""
        if not self._initialized:
            await self.initialize()

        # Try Redis first (atomic operation)
        if self._redis_available and self.redis_client:
            try:
                result = await self._redis_operation_with_retry(self.redis_client.incrby, key, amount)
                if ttl:
                    await self._redis_operation_with_retry(self.redis_client.expire, key, ttl)
                return result
            except Exception as e:
                logger.warning(f"⚠️ Redis increment failed for key {key}: {e}")
                self._redis_available = False

        # Fallback to memory cache (not atomic, but works)
        if self.memory_cache:
            current = self.memory_cache.get(key) or 0
            new_value = current + amount
            self.memory_cache.set(key, new_value, ttl)
            return new_value

        return None

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health and return metrics"""
        if not self._initialized:
            await self.initialize()

        health_data = {
            'redis_available': self._redis_available,
            'memory_cache_enabled': self.memory_cache is not None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Redis health
        if self._redis_available and self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                health_data['redis_info'] = {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory': redis_info.get('used_memory_human', '0B'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                }
            except Exception as e:
                health_data['redis_error'] = str(e)
                self._redis_available = False

        # Memory cache health
        if self.memory_cache:
            health_data['memory_cache_stats'] = self.memory_cache.stats()

        return health_data

    # High-level cache methods for smart defaults
    async def cache_user_profile(self, user_id: str, profile: UserProfile, ttl: int = 3600) -> bool:
        """Cache user profile with smart key"""
        key = f"user_profile:{user_id}"
        return await self.set(key, profile, ttl)

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get cached user profile"""
        key = f"user_profile:{user_id}"
        data = await self.get(key)

        if data and isinstance(data, dict):
            # Reconstruct UserProfile from cached data
            try:
                # Handle nested preferences object
                preferences_data = data.get('preferences')
                preferences = None
                if preferences_data:
                    if isinstance(preferences_data, dict):
                        preferences = UserPreferences(**preferences_data)
                    else:
                        # Already a UserPreferences object
                        preferences = preferences_data

                return UserProfile(
                    id=data.get('id'),
                    user_id=data.get('user_id'),
                    role=data.get('role'),
                    preferences=preferences
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to reconstruct UserProfile from cache: {e}")
                logger.warning(f"Data structure: {data}")

        return None

    async def cache_recommendations(self, user_id: str, recommendations: List[Recommendation], ttl: int = 1800) -> bool:
        """Cache user recommendations"""
        key = f"recommendations:{user_id}"
        return await self.set(key, recommendations, ttl)

    async def get_recommendations(self, user_id: str) -> List[Recommendation]:
        """Get cached recommendations"""
        key = f"recommendations:{user_id}"
        data = await self.get(key, [])
        return data if isinstance(data, list) else []

    async def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate all cache entries for a user"""
        pattern = f"*:{user_id}*" if self._redis_available else None
        success = True

        # Delete specific known keys
        keys_to_delete = [
            f"user_profile:{user_id}",
            f"recommendations:{user_id}",
            f"user_behavior:{user_id}",
            f"user_preferences:{user_id}"
        ]

        for key in keys_to_delete:
            result = await self.delete(key)
            success = success and result

        return success

    # PLACEHOLDER METHODS - These log what they would do in production
    async def warm_cache(self, user_ids: List[str]):
        """PLACEHOLDER: Warm cache with frequently accessed data"""
        logger.info(f"🔥 [PLACEHOLDER] Would warm cache for {len(user_ids)} users")
        # In production: Pre-load user profiles, recent recommendations, etc.

    async def cache_analytics(self) -> Dict[str, Any]:
        """PLACEHOLDER: Cache usage analytics"""
        logger.info("📊 [PLACEHOLDER] Would gather cache analytics: hit rates, popular keys, etc.")
        # In production: Track cache hit/miss rates, popular keys, performance metrics
        return {'status': 'placeholder'}

    async def optimize_cache(self):
        """PLACEHOLDER: Cache optimization and cleanup"""
        logger.info("⚡ [PLACEHOLDER] Would optimize cache: cleanup expired keys, rebalance memory")
        # In production: Clean expired keys, optimize memory usage, rebalance

# Factory function for easy initialization
async def create_cache_manager(
    redis_url: str = "redis://localhost:6379",
    use_memory_fallback: bool = True,
    **kwargs
) -> CacheManager:
    """Factory function to create and initialize cache manager"""
    config = CacheConfig(redis_url=redis_url, use_memory_fallback=use_memory_fallback, **kwargs)
    manager = CacheManager(config)
    await manager.initialize()
    return manager

# Example usage and testing
if __name__ == "__main__":
    async def test_cache():
        """Test cache operations"""

        try:
            # Initialize cache (will fall back to memory if Redis not available)
            cache = await create_cache_manager(
                redis_url="redis://localhost:6379",
                use_memory_fallback=True,
                default_ttl=300  # 5 minutes
            )

            try:
                # Test health check
                health = await cache.health_check()
                print(f"Cache Health: {health}")

                # Test basic operations
                await cache.set("test_key", {"message": "Hello, Cache!"}, ttl=60)

                value = await cache.get("test_key")
                print(f"Retrieved value: {value}")

                exists = await cache.exists("test_key")
                print(f"Key exists: {exists}")

                # Test user profile caching
                user_profile = UserProfile(
                    id="test_profile_id",
                    user_id="test_user_cache",
                    role="data_analyst",
                    preferences=UserPreferences(auto_connect_threshold=0.9)
                )

                await cache.cache_user_profile("test_user_cache", user_profile)
                print("✅ User profile cached")

                retrieved_profile = await cache.get_user_profile("test_user_cache")
                print(f"Retrieved profile: {retrieved_profile.role if retrieved_profile else 'None'}")

                # Test counter
                await cache.increment("page_views", 1)
                await cache.increment("page_views", 5)
                count = await cache.get("page_views")
                print(f"Page views count: {count}")

                # Test recommendations caching
                recommendations = [
                    Recommendation(id="rec1", user_id="test_user_cache"),
                    Recommendation(id="rec2", user_id="test_user_cache")
                ]

                await cache.cache_recommendations("test_user_cache", recommendations)
                cached_recs = await cache.get_recommendations("test_user_cache")
                print(f"Cached recommendations: {len(cached_recs)}")

                # Test cleanup
                await cache.delete("test_key")
                await cache.invalidate_user_cache("test_user_cache")

                print("✅ All cache tests passed!")

            finally:
                await cache.close()

        except Exception as e:
            print(f"❌ Cache test failed: {e}")
            import traceback
            traceback.print_exc()

    def test_memory_cache():
        """Test memory cache independently"""
        print("\n🧠 Testing Memory Cache:")

        memory_cache = MemoryCache(max_size=3)

        # Test basic operations
        memory_cache.set("key1", "value1", ttl=60)
        memory_cache.set("key2", {"nested": "value"})
        memory_cache.set("key3", [1, 2, 3])

        print(f"Get key1: {memory_cache.get('key1')}")
        print(f"Get key2: {memory_cache.get('key2')}")
        print(f"Get key3: {memory_cache.get('key3')}")

        # Test LRU eviction
        memory_cache.set("key4", "value4")  # Should evict least recently used
        print(f"After LRU eviction - key1: {memory_cache.get('key1')}")

        # Test stats
        stats = memory_cache.stats()
        print(f"Memory cache stats: {stats}")

        print("✅ Memory cache tests passed!")

    # Run tests
    print("🧪 Testing Cache Layer...")
    test_memory_cache()

    print("\n🔄 Testing Async Cache Operations...")
    asyncio.run(test_cache())