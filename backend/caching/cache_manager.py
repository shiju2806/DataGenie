# caching/cache_manager.py - Production Smart Cache Manager
"""
Intelligent caching system for DataGenie Multi-Source Analytics
Supports multiple backends with smart invalidation and performance optimization
"""

from typing import Any, Optional, Dict, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import hashlib
import pickle
import logging
from abc import ABC, abstractmethod

# Local imports with fallback
try:
    from core.models import CachePolicy, CacheKey, CacheEntry, CacheStats
    from core.exceptions import CacheException
    from config.settings import settings
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.models import CachePolicy, CacheKey, CacheEntry, CacheStats
    from core.exceptions import CacheException
    from config.settings import settings

logger = logging.getLogger(__name__)


# ===============================
# Cache Strategy Types
# ===============================

class CacheBackendType(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class InvalidationStrategy(str, Enum):
    TTL_BASED = "ttl_based"
    EVENT_BASED = "event_based"
    DEPENDENCY_BASED = "dependency_based"
    SMART_PREDICTION = "smart_prediction"


@dataclass
class CacheConfiguration:
    """Cache configuration for different data types"""
    policy: CachePolicy
    ttl_seconds: int
    max_size_mb: Optional[float] = None
    compression_enabled: bool = False
    backend_preference: List[CacheBackendType] = None

    def __post_init__(self):
        if self.backend_preference is None:
            self.backend_preference = [CacheBackendType.MEMORY, CacheBackendType.REDIS]


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_retrieval_time_ms: float = 0.0
    last_reset: datetime = None

    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.utcnow()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ===============================
# Cache Backend Interface
# ===============================

class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Store value with TTL"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        pass


# ===============================
# Memory Cache Backend
# ===============================

class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: List[str] = []
        self._metrics = CacheMetrics()

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        start_time = datetime.utcnow()

        try:
            if key in self._cache:
                value, expires_at = self._cache[key]

                # Check expiration
                if datetime.utcnow() > expires_at:
                    await self.delete(key)
                    self._metrics.misses += 1
                    return None

                # Update access order (move to end)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                self._metrics.hits += 1
                return value

            self._metrics.misses += 1
            return None

        finally:
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_avg_retrieval_time(elapsed_ms)

    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Store value with TTL"""
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = (value, expires_at)

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # Update metrics
            self._metrics.total_size_bytes += self._estimate_size(value)

            return True

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            if key in self._cache:
                value, _ = self._cache[key]
                self._metrics.total_size_bytes -= self._estimate_size(value)
                del self._cache[key]

                if key in self._access_order:
                    self._access_order.remove(key)

                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        if key not in self._cache:
            return False

        _, expires_at = self._cache[key]
        if datetime.utcnow() > expires_at:
            await self.delete(key)
            return False

        return True

    async def clear(self) -> bool:
        """Clear all cache entries"""
        self._cache.clear()
        self._access_order.clear()
        self._metrics = CacheMetrics()
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "memory",
            "entries_count": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "total_size_bytes": self._metrics.total_size_bytes,
            "hit_rate": self._metrics.hit_rate,
            "hits": self._metrics.hits,
            "misses": self._metrics.misses,
            "evictions": self._metrics.evictions,
            "avg_retrieval_time_ms": self._metrics.avg_retrieval_time_ms
        }

    async def _evict_lru(self):
        """Evict least recently used item"""
        if self._access_order:
            lru_key = self._access_order[0]
            await self.delete(lru_key)
            self._metrics.evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))

    def _update_avg_retrieval_time(self, elapsed_ms: float):
        """Update average retrieval time"""
        total_operations = self._metrics.hits + self._metrics.misses
        if total_operations > 0:
            self._metrics.avg_retrieval_time_ms = (
                    (self._metrics.avg_retrieval_time_ms * (total_operations - 1) + elapsed_ms) / total_operations
            )


# ===============================
# Redis Cache Backend
# ===============================

class RedisCacheBackend(CacheBackend):
    """Redis cache backend with cluster support"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis.url
        self.redis_client = None
        self._metrics = CacheMetrics()
        self._connection_pool = None

    async def _get_client(self):
        """Get Redis client with connection pooling"""
        if self.redis_client is None:
            try:
                import redis.asyncio as redis

                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # We handle our own serialization
                    max_connections=settings.redis.max_connections,
                    retry_on_timeout=settings.redis.retry_on_timeout,
                    socket_timeout=settings.redis.socket_timeout
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache backend connected successfully")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise CacheException("connection", "redis", str(e))

        return self.redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        start_time = datetime.utcnow()

        try:
            client = await self._get_client()

            serialized_data = await client.get(key)
            if serialized_data is None:
                self._metrics.misses += 1
                return None

            # Deserialize
            try:
                value = pickle.loads(serialized_data)
                self._metrics.hits += 1
                return value
            except Exception as e:
                logger.error(f"Failed to deserialize cached value for key {key}: {e}")
                await self.delete(key)  # Remove corrupted data
                self._metrics.misses += 1
                return None

        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            self._metrics.misses += 1
            return None

        finally:
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_avg_retrieval_time(elapsed_ms)

    async def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Store value with TTL"""
        try:
            client = await self._get_client()

            # Serialize value
            try:
                serialized_data = pickle.dumps(value)
            except Exception as e:
                logger.error(f"Failed to serialize value for key {key}: {e}")
                return False

            # Store with TTL
            success = await client.setex(key, ttl_seconds, serialized_data)

            if success:
                self._metrics.total_size_bytes += len(serialized_data)

            return bool(success)

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            client = await self._get_client()
            deleted_count = await client.delete(key)
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            client = await self._get_client()
            return bool(await client.exists(key))

        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries (use with caution)"""
        try:
            client = await self._get_client()
            await client.flushdb()
            self._metrics = CacheMetrics()
            return True

        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            client = await self._get_client()
            info = await client.info()

            return {
                "backend_type": "redis",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "hit_rate": self._metrics.hit_rate,
                "hits": self._metrics.hits,
                "misses": self._metrics.misses,
                "avg_retrieval_time_ms": self._metrics.avg_retrieval_time_ms,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"backend_type": "redis", "error": str(e)}

    def _update_avg_retrieval_time(self, elapsed_ms: float):
        """Update average retrieval time"""
        total_operations = self._metrics.hits + self._metrics.misses
        if total_operations > 0:
            self._metrics.avg_retrieval_time_ms = (
                    (self._metrics.avg_retrieval_time_ms * (total_operations - 1) + elapsed_ms) / total_operations
            )


# ===============================
# Smart Cache Manager
# ===============================

class SmartCacheManager:
    """
    Intelligent cache manager with multiple backends and smart policies
    """

    def __init__(self):
        self.backends: Dict[CacheBackendType, CacheBackend] = {}
        self.configurations: Dict[str, CacheConfiguration] = {}
        self.dependencies: Dict[str, List[str]] = {}  # key -> dependent keys
        self.access_patterns: Dict[str, List[datetime]] = {}

        # Initialize backends based on configuration
        self._initialize_backends()
        self._setup_default_configurations()

    def _initialize_backends(self):
        """Initialize cache backends based on settings"""

        # Memory backend (always available)
        if settings.cache.enable_memory_cache:
            self.backends[CacheBackendType.MEMORY] = MemoryCacheBackend(
                max_size=settings.cache.max_memory_cache_size
            )
            logger.info("Memory cache backend initialized")

        # Redis backend (if enabled and available)
        if settings.cache.enable_redis_cache:
            try:
                self.backends[CacheBackendType.REDIS] = RedisCacheBackend()
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Redis cache backend failed to initialize: {e}")

    def _setup_default_configurations(self):
        """Setup default cache configurations for different data types"""

        # Query results
        self.configurations["query_results"] = CacheConfiguration(
            policy=CachePolicy.SMART_REFRESH,
            ttl_seconds=settings.cache.default_ttl,
            max_size_mb=50.0,
            compression_enabled=True,
            backend_preference=[CacheBackendType.REDIS, CacheBackendType.MEMORY]
        )

        # Data source metadata
        self.configurations["data_source_metadata"] = CacheConfiguration(
            policy=CachePolicy.LONG_TERM,
            ttl_seconds=settings.cache.long_ttl,
            compression_enabled=False,
            backend_preference=[CacheBackendType.MEMORY, CacheBackendType.REDIS]
        )

        # Join results
        self.configurations["join_results"] = CacheConfiguration(
            policy=CachePolicy.MEDIUM_TERM,
            ttl_seconds=settings.cache.default_ttl,
            max_size_mb=100.0,
            compression_enabled=True,
            backend_preference=[CacheBackendType.REDIS]
        )

        # Conflict resolutions
        self.configurations["conflict_resolutions"] = CacheConfiguration(
            policy=CachePolicy.PRECOMPUTE,
            ttl_seconds=settings.cache.long_ttl,
            compression_enabled=False,
            backend_preference=[CacheBackendType.MEMORY]
        )

        # User access evaluations
        self.configurations["access_evaluations"] = CacheConfiguration(
            policy=CachePolicy.SHORT_TERM,
            ttl_seconds=settings.cache.short_ttl,
            compression_enabled=False,
            backend_preference=[CacheBackendType.MEMORY]
        )

    async def get(self, key: str, data_type: str = "default") -> Optional[Any]:
        """
        Intelligent cache retrieval with backend fallback
        """
        config = self.configurations.get(data_type, self.configurations["query_results"])

        # Record access pattern
        self._record_access(key)

        # Try backends in preference order
        for backend_type in config.backend_preference:
            if backend_type in self.backends:
                try:
                    value = await self.backends[backend_type].get(key)
                    if value is not None:
                        logger.debug(f"Cache hit: {key} from {backend_type.value}")
                        return value
                except Exception as e:
                    logger.warning(f"Cache get failed on {backend_type.value}: {e}")
                    continue

        logger.debug(f"Cache miss: {key}")
        return None

    async def set(self,
                  key: str,
                  value: Any,
                  data_type: str = "default",
                  custom_ttl: Optional[int] = None) -> bool:
        """
        Intelligent cache storage with backend selection
        """
        config = self.configurations.get(data_type, self.configurations["query_results"])
        ttl = custom_ttl or config.ttl_seconds

        # Select best backend based on value characteristics
        selected_backend = self._select_optimal_backend(value, config)

        if selected_backend and selected_backend in self.backends:
            try:
                success = await self.backends[selected_backend].set(key, value, ttl)
                if success:
                    logger.debug(f"Cache set: {key} to {selected_backend.value}")

                    # Also store in memory if it's a small, frequently accessed item
                    if (selected_backend != CacheBackendType.MEMORY and
                            CacheBackendType.MEMORY in self.backends and
                            self._is_frequently_accessed(key)):
                        await self.backends[CacheBackendType.MEMORY].set(key, value, min(ttl, 300))

                return success

            except Exception as e:
                logger.error(f"Cache set failed on {selected_backend.value}: {e}")

        return False

    async def delete(self, key: str) -> bool:
        """Delete from all backends"""
        success = False

        for backend in self.backends.values():
            try:
                if await backend.delete(key):
                    success = True
            except Exception as e:
                logger.warning(f"Cache delete failed: {e}")

        # Clean up dependencies
        if key in self.dependencies:
            dependent_keys = self.dependencies[key]
            await self.invalidate_keys(dependent_keys)
            del self.dependencies[key]

        # Clean up access patterns
        if key in self.access_patterns:
            del self.access_patterns[key]

        return success

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache keys matching a pattern
        Returns number of keys invalidated
        """
        invalidated_count = 0

        # This is a simplified implementation
        # In production, you'd want more sophisticated pattern matching
        for backend in self.backends.values():
            try:
                stats = await backend.get_stats()
                # For now, we can't efficiently pattern-match across all backends
                # This would require backend-specific implementations
                pass
            except Exception as e:
                logger.warning(f"Pattern invalidation failed: {e}")

        return invalidated_count

    async def invalidate_keys(self, keys: List[str]) -> int:
        """Invalidate specific keys"""
        invalidated_count = 0

        for key in keys:
            if await self.delete(key):
                invalidated_count += 1

        return invalidated_count

    def add_dependency(self, key: str, dependent_keys: List[str]):
        """Add cache dependency relationships"""
        if key not in self.dependencies:
            self.dependencies[key] = []

        self.dependencies[key].extend(dependent_keys)

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "backends": {},
            "configurations": len(self.configurations),
            "dependencies": len(self.dependencies),
            "access_patterns": len(self.access_patterns)
        }

        for backend_type, backend in self.backends.items():
            try:
                backend_stats = await backend.get_stats()
                stats["backends"][backend_type.value] = backend_stats
            except Exception as e:
                stats["backends"][backend_type.value] = {"error": str(e)}

        return stats

    def _select_optimal_backend(self, value: Any, config: CacheConfiguration) -> Optional[CacheBackendType]:
        """Select optimal backend based on value characteristics and configuration"""

        # Estimate value size
        estimated_size = self._estimate_value_size(value)

        # Large values go to Redis if available
        if estimated_size > 1024 * 1024:  # > 1MB
            if CacheBackendType.REDIS in config.backend_preference and CacheBackendType.REDIS in self.backends:
                return CacheBackendType.REDIS

        # Small, frequently accessed values go to memory
        if estimated_size < 1024:  # < 1KB
            if CacheBackendType.MEMORY in config.backend_preference and CacheBackendType.MEMORY in self.backends:
                return CacheBackendType.MEMORY

        # Use first available backend from preference
        for backend_type in config.backend_preference:
            if backend_type in self.backends:
                return backend_type

        return None

    def _estimate_value_size(self, value: Any) -> int:
        """Estimate value size in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))

    def _record_access(self, key: str):
        """Record cache access for pattern analysis"""
        now = datetime.utcnow()

        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(now)

        # Keep only recent accesses (last hour)
        cutoff = now - timedelta(hours=1)
        self.access_patterns[key] = [
            access_time for access_time in self.access_patterns[key]
            if access_time > cutoff
        ]

    def _is_frequently_accessed(self, key: str) -> bool:
        """Check if key is frequently accessed"""
        if key not in self.access_patterns:
            return False

        # Consider frequently accessed if accessed more than 5 times in last hour
        return len(self.access_patterns[key]) > 5


# Global cache manager instance
cache_manager = SmartCacheManager()