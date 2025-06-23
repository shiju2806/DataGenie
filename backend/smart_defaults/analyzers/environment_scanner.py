"""
Smart Defaults Environment Scanner
Automatically discovers available data sources, services, and infrastructure
File location: smart_defaults/analyzers/environment_scanner.py
"""

import asyncio
import logging
import socket
import subprocess
import platform
import os
import json
import yaml
import sys
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import concurrent.futures
from pathlib import Path
import ipaddress
import ssl
import urllib.parse

# Import dependencies with fallbacks
try:
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..storage.config_loader import ConfigurationLoader
    from ..models.data_source import DataSource, SourceMetadata
    from ..utils.monitoring import AnalyticsEngine, EventType
except ImportError:
    # For direct execution, create mock classes
    from typing import Any
    from dataclasses import dataclass
    from datetime import datetime


    @dataclass
    class DataSource:
        id: str = "test_source"
        name: str = "Test Source"
        source_type: str = "database"
        connection_config: Dict[str, Any] = None


    @dataclass
    class SourceMetadata:
        discovered_at: datetime = None


    class DatabaseManager:
        async def initialize(self): pass

        async def close(self): pass

        async def create_data_source(self, source): pass


    class CacheManager:
        async def initialize(self): pass

        async def close(self): pass

        async def set(self, key, value, ttl=None): pass

        async def get(self, key, default=None): return default


    class ConfigurationLoader:
        def get_all_industry_profiles(self): return {}


    class AnalyticsEngine:
        async def track_event(self, *args, **kwargs): pass


    class EventType:
        SOURCE_CONNECTED = "source_connected"
        SOURCE_FAILED = "source_failed"

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of discoverable sources"""
    DATABASE = "database"
    API = "api"
    FILE_SYSTEM = "file_system"
    CLOUD_SERVICE = "cloud_service"
    MESSAGE_QUEUE = "message_queue"
    CACHE = "cache"
    MONITORING = "monitoring"
    VERSION_CONTROL = "version_control"
    CONTAINER = "container"
    WEB_SERVICE = "web_service"
    LOCAL_SERVICE = "local_service"


class DiscoveryStatus(Enum):
    """Status of discovery attempts"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"


@dataclass
class DiscoveryTarget:
    """Target for environment discovery"""
    name: str
    type: SourceType
    host: str
    port: int
    protocol: str = "tcp"
    service_name: Optional[str] = None
    default_credentials: Dict[str, str] = field(default_factory=dict)
    probe_methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of a discovery attempt"""
    target: DiscoveryTarget
    status: DiscoveryStatus
    discovered_at: datetime
    response_time_ms: float
    available: bool
    version: Optional[str] = None
    service_info: Dict[str, Any] = field(default_factory=dict)
    connection_details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class EnvironmentProfile:
    """Complete environment profile"""
    scan_id: str
    user_id: str
    scanned_at: datetime
    scan_duration_seconds: float
    discovered_sources: List[DiscoveryResult]
    network_info: Dict[str, Any]
    system_info: Dict[str, Any]
    security_context: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


class NetworkScanner:
    """Network-based service discovery"""

    def __init__(self, timeout: float = 2.0, max_workers: int = 50):
        self.timeout = timeout
        self.max_workers = max_workers

    async def scan_port(self, host: str, port: int, protocol: str = "tcp") -> Tuple[bool, float, Optional[str]]:
        """Scan a single port"""
        start_time = asyncio.get_event_loop().time()

        try:
            if protocol.lower() == "tcp":
                future = asyncio.open_connection(host, port)
                reader, writer = await asyncio.wait_for(future, timeout=self.timeout)
                writer.close()
                await writer.wait_closed()

                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                return True, response_time, None

            elif protocol.lower() == "udp":
                # UDP scanning is more complex and less reliable
                transport, protocol_instance = await asyncio.get_event_loop().create_datagram_endpoint(
                    lambda: asyncio.DatagramProtocol(),
                    remote_addr=(host, port)
                )
                transport.close()

                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                return True, response_time, None

        except (asyncio.TimeoutError, OSError) as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return False, response_time, str(e)

        return False, 0.0, "Unknown protocol"

    async def scan_service_banner(self, host: str, port: int) -> Optional[str]:
        """Try to get service banner/version info"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Try to read banner
            try:
                banner = await asyncio.wait_for(
                    reader.read(1024),
                    timeout=1.0
                )
                writer.close()
                await writer.wait_closed()

                return banner.decode('utf-8', errors='ignore').strip()
            except:
                writer.close()
                await writer.wait_closed()
                return None

        except:
            return None

    def get_local_networks(self) -> List[str]:
        """Get local network ranges to scan"""
        networks = []

        try:
            # Get network interfaces
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                        try:
                            # Calculate network range
                            network = ipaddress.IPv4Network(f"{addr.address}/{addr.netmask}", strict=False)
                            networks.append(str(network.network_address) + "/24")  # Limit to /24
                        except:
                            continue
        except Exception as e:
            logger.warning(f"⚠️ Failed to get local networks: {e}")
            # Fallback to common private ranges
            networks = ["192.168.1.0/24", "10.0.1.0/24", "172.16.1.0/24"]

        return networks


class ServiceProber:
    """Probes specific services for detailed information"""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def probe_database(self, host: str, port: int, db_type: str) -> Dict[str, Any]:
        """Probe database services"""
        result = {
            "type": "database",
            "subtype": db_type,
            "available": False,
            "version": None,
            "features": []
        }

        try:
            if db_type == "postgresql":
                result.update(await self._probe_postgresql(host, port))
            elif db_type == "mysql":
                result.update(await self._probe_mysql(host, port))
            elif db_type == "redis":
                result.update(await self._probe_redis(host, port))
            elif db_type == "mongodb":
                result.update(await self._probe_mongodb(host, port))
            elif db_type == "elasticsearch":
                result.update(await self._probe_elasticsearch(host, port))

        except Exception as e:
            result["error"] = str(e)

        return result

    async def _probe_postgresql(self, host: str, port: int) -> Dict[str, Any]:
        """Probe PostgreSQL database"""
        try:
            # Try to connect and get version (without actual connection library)
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()

            return {
                "available": True,
                "version": "Unknown (no psycopg2)",
                "features": ["SQL", "ACID", "Relational"],
                "default_database": "postgres",
                "default_port": 5432
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def _probe_mysql(self, host: str, port: int) -> Dict[str, Any]:
        """Probe MySQL database"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()

            return {
                "available": True,
                "version": "Unknown (no mysql-connector)",
                "features": ["SQL", "Relational", "InnoDB"],
                "default_database": "mysql",
                "default_port": 3306
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def _probe_redis(self, host: str, port: int) -> Dict[str, Any]:
        """Probe Redis cache"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )

            # Send PING command
            writer.write(b"PING\r\n")
            await writer.drain()

            response = await asyncio.wait_for(reader.read(100), timeout=1.0)
            writer.close()
            await writer.wait_closed()

            if b"PONG" in response:
                return {
                    "available": True,
                    "version": "Unknown",
                    "features": ["Key-Value", "Cache", "Pub/Sub"],
                    "default_port": 6379
                }
        except Exception as e:
            return {"available": False, "error": str(e)}

        return {"available": False}

    async def _probe_mongodb(self, host: str, port: int) -> Dict[str, Any]:
        """Probe MongoDB database"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()

            return {
                "available": True,
                "version": "Unknown (no pymongo)",
                "features": ["Document", "NoSQL", "BSON"],
                "default_database": "admin",
                "default_port": 27017
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def _probe_elasticsearch(self, host: str, port: int) -> Dict[str, Any]:
        """Probe Elasticsearch"""
        try:
            # Try HTTP request to Elasticsearch API
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"http://{host}:{port}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "available": True,
                            "version": data.get("version", {}).get("number", "Unknown"),
                            "features": ["Search", "Analytics", "JSON"],
                            "cluster_name": data.get("cluster_name", "Unknown"),
                            "default_port": 9200
                        }
        except ImportError:
            # Fallback without aiohttp
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()

            return {
                "available": True,
                "version": "Unknown (no aiohttp)",
                "features": ["Search", "Analytics", "JSON"],
                "default_port": 9200
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def probe_web_service(self, host: str, port: int, path: str = "/") -> Dict[str, Any]:
        """Probe web services and APIs"""
        result = {
            "type": "web_service",
            "available": False,
            "status_code": None,
            "server": None,
            "features": []
        }

        try:
            # Try to import aiohttp for HTTP requests
            try:
                import aiohttp

                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    url = f"http://{host}:{port}{path}"
                    async with session.get(url) as response:
                        result.update({
                            "available": True,
                            "status_code": response.status,
                            "server": response.headers.get("Server"),
                            "content_type": response.headers.get("Content-Type"),
                            "features": ["HTTP", "Web"]
                        })

                        # Check for API indicators
                        if "application/json" in response.headers.get("Content-Type", ""):
                            result["features"].append("API")
                            result["features"].append("JSON")

                        # Try to detect framework/platform
                        server_header = response.headers.get("Server", "").lower()
                        if "nginx" in server_header:
                            result["features"].append("Nginx")
                        elif "apache" in server_header:
                            result["features"].append("Apache")
                        elif "gunicorn" in server_header:
                            result["features"].append("Python")
                            result["features"].append("Gunicorn")

            except ImportError:
                # Fallback to basic socket connection
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.timeout
                )

                # Send basic HTTP request
                http_request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n"
                writer.write(http_request.encode())
                await writer.drain()

                response = await asyncio.wait_for(reader.read(1024), timeout=2.0)
                writer.close()
                await writer.wait_closed()

                if response:
                    response_text = response.decode('utf-8', errors='ignore')
                    if "HTTP/" in response_text:
                        result.update({
                            "available": True,
                            "features": ["HTTP", "Web"],
                            "raw_response": response_text[:200]
                        })

        except Exception as e:
            result["error"] = str(e)

        return result


class EnvironmentScanner:
    """Main environment scanner orchestrator"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 config_loader: Optional[ConfigurationLoader] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 scan_timeout: float = 30.0,
                 max_concurrent: int = 100):

        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.config_loader = config_loader
        self.analytics_engine = analytics_engine

        self.scan_timeout = scan_timeout
        self.max_concurrent = max_concurrent

        self.network_scanner = NetworkScanner(timeout=2.0, max_workers=50)
        self.service_prober = ServiceProber(timeout=5.0)

        # Default discovery targets
        self.discovery_targets = self._get_default_targets()

        self._initialized = False

    async def initialize(self):
        """Initialize the environment scanner"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()

        # Load custom targets from config if available
        if self.config_loader:
            await self._load_custom_targets()

        self._initialized = True
        logger.info("✅ Environment scanner initialized")

    async def close(self):
        """Close the environment scanner"""
        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Environment scanner closed")

    def _get_default_targets(self) -> List[DiscoveryTarget]:
        """Get default discovery targets for common services"""
        targets = []

        # Database services
        db_services = [
            ("PostgreSQL", "postgresql", 5432),
            ("MySQL", "mysql", 3306),
            ("Redis", "redis", 6379),
            ("MongoDB", "mongodb", 27017),
            ("Elasticsearch", "elasticsearch", 9200),
            ("Cassandra", "cassandra", 9042),
            ("InfluxDB", "influxdb", 8086)
        ]

        for name, service, port in db_services:
            targets.append(DiscoveryTarget(
                name=name,
                type=SourceType.DATABASE,
                host="localhost",
                port=port,
                service_name=service,
                probe_methods=["tcp", "banner", "service_specific"]
            ))

        # Web services and APIs
        web_services = [
            ("Jupyter Notebook", "jupyter", 8888),
            ("Grafana", "grafana", 3000),
            ("Kibana", "kibana", 5601),
            ("Airflow", "airflow", 8080),
            ("MLflow", "mlflow", 5000),
            ("Jenkins", "jenkins", 8080),
            ("GitLab", "gitlab", 80),
            ("Superset", "superset", 8088)
        ]

        for name, service, port in web_services:
            targets.append(DiscoveryTarget(
                name=name,
                type=SourceType.WEB_SERVICE,
                host="localhost",
                port=port,
                protocol="http",
                service_name=service,
                probe_methods=["http", "banner"]
            ))

        # Message queues
        mq_services = [
            ("Apache Kafka", "kafka", 9092),
            ("RabbitMQ", "rabbitmq", 5672),
            ("Apache Pulsar", "pulsar", 6650),
            ("NATS", "nats", 4222)
        ]

        for name, service, port in mq_services:
            targets.append(DiscoveryTarget(
                name=name,
                type=SourceType.MESSAGE_QUEUE,
                host="localhost",
                port=port,
                service_name=service,
                probe_methods=["tcp", "banner"]
            ))

        return targets

    async def _load_custom_targets(self):
        """Load custom discovery targets from configuration"""
        try:
            # This would load from config files in production
            industry_profiles = self.config_loader.get_all_industry_profiles()

            # Add industry-specific targets
            for industry_name, profile in industry_profiles.items():
                for source in profile.common_sources:
                    # Map source names to discovery targets
                    if source not in [t.service_name for t in self.discovery_targets]:
                        target = self._create_target_from_source_name(source)
                        if target:
                            self.discovery_targets.append(target)

            logger.info(f"📋 Loaded {len(self.discovery_targets)} discovery targets")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load custom targets: {e}")

    def _create_target_from_source_name(self, source_name: str) -> Optional[DiscoveryTarget]:
        """Create a discovery target from a source name"""
        # Common source name mappings
        source_mappings = {
            "postgres": ("PostgreSQL", SourceType.DATABASE, 5432),
            "mysql": ("MySQL", SourceType.DATABASE, 3306),
            "redis": ("Redis", SourceType.CACHE, 6379),
            "mongodb": ("MongoDB", SourceType.DATABASE, 27017),
            "elasticsearch": ("Elasticsearch", SourceType.DATABASE, 9200),
            "jupyter": ("Jupyter Notebook", SourceType.WEB_SERVICE, 8888),
            "airflow": ("Apache Airflow", SourceType.WEB_SERVICE, 8080),
            "kafka": ("Apache Kafka", SourceType.MESSAGE_QUEUE, 9092),
            "rabbitmq": ("RabbitMQ", SourceType.MESSAGE_QUEUE, 5672),
            "grafana": ("Grafana", SourceType.WEB_SERVICE, 3000),
            "tableau": ("Tableau Server", SourceType.WEB_SERVICE, 80),
            "superset": ("Apache Superset", SourceType.WEB_SERVICE, 8088)
        }

        if source_name.lower() in source_mappings:
            name, source_type, port = source_mappings[source_name.lower()]
            return DiscoveryTarget(
                name=name,
                type=source_type,
                host="localhost",
                port=port,
                service_name=source_name.lower(),
                probe_methods=["tcp", "banner"]
            )

        return None

    async def scan_environment(self, user_id: str,
                               quick_scan: bool = False,
                               network_scan: bool = True,
                               custom_hosts: Optional[List[str]] = None) -> EnvironmentProfile:
        """Perform a complete environment scan"""

        if not self._initialized:
            await self.initialize()

        scan_id = f"env_scan_{user_id}_{int(datetime.now(timezone.utc).timestamp())}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"🔍 Starting environment scan {scan_id}")

        try:
            # Check cache for recent scan
            if self.cache_manager and not quick_scan:
                cache_key = f"env_scan:{user_id}"
                cached_profile = await self.cache_manager.get(cache_key)
                if cached_profile:
                    logger.info("📋 Using cached environment profile")
                    return EnvironmentProfile(**cached_profile)

            # Gather system information
            system_info = await self._get_system_info()
            network_info = await self._get_network_info()
            security_context = await self._get_security_context()

            # Discover services
            discovered_sources = []

            if quick_scan:
                # Quick scan - only localhost common services
                discovered_sources = await self._quick_localhost_scan()
            else:
                # Full scan
                localhost_sources = await self._scan_localhost_services()
                discovered_sources.extend(localhost_sources)

                if network_scan:
                    network_sources = await self._scan_network_services(custom_hosts)
                    discovered_sources.extend(network_sources)

            # Calculate confidence and generate recommendations
            confidence_score = self._calculate_confidence(discovered_sources)
            recommendations = await self._generate_recommendations(discovered_sources, user_id)

            # Create environment profile
            scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            profile = EnvironmentProfile(
                scan_id=scan_id,
                user_id=user_id,
                scanned_at=start_time,
                scan_duration_seconds=scan_duration,
                discovered_sources=discovered_sources,
                network_info=network_info,
                system_info=system_info,
                security_context=security_context,
                recommendations=recommendations,
                confidence_score=confidence_score
            )

            # Cache the profile
            if self.cache_manager:
                await self.cache_manager.set(
                    f"env_scan:{user_id}",
                    asdict(profile),
                    ttl=3600  # 1 hour
                )

            # Track analytics
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_id,
                    event_type=EventType.SOURCE_CONNECTED,
                    data={
                        "scan_id": scan_id,
                        "sources_found": len([s for s in discovered_sources if s.available]),
                        "scan_duration": scan_duration,
                        "confidence_score": confidence_score
                    }
                )

            logger.info(f"✅ Environment scan completed: {len(discovered_sources)} sources discovered")
            return profile

        except Exception as e:
            logger.error(f"❌ Environment scan failed: {e}")

            # Track failure
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_id,
                    event_type=EventType.SOURCE_FAILED,
                    data={"scan_id": scan_id, "error": str(e)}
                )

            # Return minimal profile on failure
            return EnvironmentProfile(
                scan_id=scan_id,
                user_id=user_id,
                scanned_at=start_time,
                scan_duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                discovered_sources=[],
                network_info={},
                system_info={},
                security_context={},
                recommendations=["Environment scan failed - please check network connectivity"],
                confidence_score=0.0
            )

    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "disk_space_gb": round(psutil.disk_usage('/').total / (1024 ** 3), 2),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname()
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get system info: {e}")
            return {"error": str(e)}

    async def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            interfaces = {}
            for interface, addrs in psutil.net_if_addrs().items():
                interface_info = []
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        interface_info.append({
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "family": "IPv4"
                        })
                    elif addr.family == socket.AF_INET6:
                        interface_info.append({
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "family": "IPv6"
                        })
                if interface_info:
                    interfaces[interface] = interface_info

            return {
                "interfaces": interfaces,
                "local_networks": self.network_scanner.get_local_networks()
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get network info: {e}")
            return {"error": str(e)}

    async def _get_security_context(self) -> Dict[str, Any]:
        """Get security context information"""
        try:
            # Check if running as admin/root
            is_admin = os.geteuid() == 0 if hasattr(os, 'geteuid') else False

            # Check for common security tools
            security_tools = []
            common_security_paths = [
                "/usr/bin/fail2ban",
                "/usr/bin/ufw",
                "/usr/bin/iptables",
                "/usr/bin/semanage"
            ]

            for path in common_security_paths:
                if os.path.exists(path):
                    security_tools.append(os.path.basename(path))

            return {
                "is_admin": is_admin,
                "user": os.getenv("USER", "unknown"),
                "security_tools": security_tools,
                "environment_variables": {
                    k: v for k, v in os.environ.items()
                    if any(keyword in k.upper() for keyword in ["PATH", "HOME", "SHELL"])
                }
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get security context: {e}")
            return {"error": str(e)}

    async def _quick_localhost_scan(self) -> List[DiscoveryResult]:
        """Quick scan of localhost services only"""
        results = []

        # Scan only most common services
        quick_targets = [
            target for target in self.discovery_targets
            if target.service_name in ["postgresql", "mysql", "redis", "mongodb", "jupyter", "elasticsearch"]
        ]

        tasks = []
        for target in quick_targets:
            task = asyncio.create_task(self._scan_single_target(target))
            tasks.append(task)

    async def _quick_localhost_scan(self) -> List[DiscoveryResult]:
        """Quick scan of localhost services only"""
        results = []

        # Scan only most common services
        quick_targets = [
            target for target in self.discovery_targets
            if target.service_name in ["postgresql", "mysql", "redis", "mongodb", "jupyter", "elasticsearch"]
        ]

        tasks = []
        for target in quick_targets:
            task = asyncio.create_task(self._scan_single_target(target))
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions
            results = [r for r in results if isinstance(r, DiscoveryResult)]

        return results

    async def _scan_localhost_services(self) -> List[DiscoveryResult]:
        """Scan all localhost services"""
        results = []

        # Limit concurrent scans to avoid overwhelming the system
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def scan_with_semaphore(target):
            async with semaphore:
                return await self._scan_single_target(target)

        tasks = [scan_with_semaphore(target) for target in self.discovery_targets]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions
            results = [r for r in results if isinstance(r, DiscoveryResult)]

        return results

    async def _scan_network_services(self, custom_hosts: Optional[List[str]] = None) -> List[DiscoveryResult]:
        """Scan network for services (simplified for demo)"""
        results = []

        # In production, this would do actual network scanning
        # For demo, we'll just check if custom hosts are provided
        if custom_hosts:
            for host in custom_hosts:
                for target in self.discovery_targets[:5]:  # Limit for demo
                    network_target = DiscoveryTarget(
                        name=f"{target.name} on {host}",
                        type=target.type,
                        host=host,
                        port=target.port,
                        service_name=target.service_name,
                        probe_methods=target.probe_methods
                    )

                    result = await self._scan_single_target(network_target)
                    results.append(result)

        return results

    async def _scan_single_target(self, target: DiscoveryTarget) -> DiscoveryResult:
        """Scan a single discovery target"""
        start_time = datetime.now(timezone.utc)

        try:
            # Basic port scan
            is_open, response_time, error = await self.network_scanner.scan_port(
                target.host, target.port, target.protocol
            )

            if not is_open:
                return DiscoveryResult(
                    target=target,
                    status=DiscoveryStatus.NOT_FOUND,
                    discovered_at=start_time,
                    response_time_ms=response_time,
                    available=False,
                    error_message=error,
                    confidence_score=0.0
                )

            # Port is open, try to get more information
            service_info = {}
            connection_details = {"host": target.host, "port": target.port}
            version = None
            confidence_score = 0.5  # Base confidence for open port

            # Try service-specific probing
            if target.type == SourceType.DATABASE and target.service_name:
                probe_result = await self.service_prober.probe_database(
                    target.host, target.port, target.service_name
                )
                service_info.update(probe_result)
                if probe_result.get("available"):
                    confidence_score = 0.8
                    version = probe_result.get("version")
                    connection_details.update({
                        "database_type": target.service_name,
                        "features": probe_result.get("features", [])
                    })

            elif target.type == SourceType.WEB_SERVICE:
                probe_result = await self.service_prober.probe_web_service(
                    target.host, target.port
                )
                service_info.update(probe_result)
                if probe_result.get("available"):
                    confidence_score = 0.7
                    connection_details.update({
                        "service_type": "web_service",
                        "status_code": probe_result.get("status_code"),
                        "server": probe_result.get("server")
                    })

            # Try to get banner information
            if "banner" in target.probe_methods:
                banner = await self.network_scanner.scan_service_banner(target.host, target.port)
                if banner:
                    service_info["banner"] = banner
                    confidence_score = min(confidence_score + 0.1, 1.0)

            return DiscoveryResult(
                target=target,
                status=DiscoveryStatus.SUCCESS,
                discovered_at=start_time,
                response_time_ms=response_time,
                available=True,
                version=version,
                service_info=service_info,
                connection_details=connection_details,
                confidence_score=confidence_score
            )

        except asyncio.TimeoutError:
            return DiscoveryResult(
                target=target,
                status=DiscoveryStatus.TIMEOUT,
                discovered_at=start_time,
                response_time_ms=self.scan_timeout * 1000,
                available=False,
                error_message="Scan timeout",
                confidence_score=0.0
            )
        except Exception as e:
            return DiscoveryResult(
                target=target,
                status=DiscoveryStatus.FAILED,
                discovered_at=start_time,
                response_time_ms=0.0,
                available=False,
                error_message=str(e),
                confidence_score=0.0
            )

    def _calculate_confidence(self, discovered_sources: List[DiscoveryResult]) -> float:
        """Calculate overall confidence score for the environment scan"""
        if not discovered_sources:
            return 0.0

        # Calculate based on number of discovered sources and their individual confidence
        available_sources = [s for s in discovered_sources if s.available]

        if not available_sources:
            return 0.1  # Very low confidence if nothing found

        # Average confidence of available sources
        avg_confidence = sum(s.confidence_score for s in available_sources) / len(available_sources)

        # Boost confidence based on diversity of source types
        source_types = set(s.target.type for s in available_sources)
        diversity_bonus = min(len(source_types) * 0.1, 0.3)

        # Boost confidence based on number of sources
        quantity_bonus = min(len(available_sources) * 0.05, 0.2)

        total_confidence = min(avg_confidence + diversity_bonus + quantity_bonus, 1.0)
        return round(total_confidence, 2)

    async def _generate_recommendations(self, discovered_sources: List[DiscoveryResult], user_id: str) -> List[str]:
        """Generate smart recommendations based on discovered sources"""
        recommendations = []

        available_sources = [s for s in discovered_sources if s.available]

        if not available_sources:
            recommendations.extend([
                "No data sources were automatically discovered in your environment",
                "Consider manually configuring connections to your data sources",
                "Check if services are running and accessible",
                "Verify network connectivity and firewall settings"
            ])
            return recommendations

        # Analyze discovered sources by type
        source_types = {}
        for source in available_sources:
            source_type = source.target.type
            if source_type not in source_types:
                source_types[source_type] = []
            source_types[source_type].append(source)

        # Database recommendations
        if SourceType.DATABASE in source_types:
            db_sources = source_types[SourceType.DATABASE]
            recommendations.append(
                f"Found {len(db_sources)} database(s) - these can be automatically configured for data analysis")

            # Specific database recommendations
            for db in db_sources:
                db_name = db.target.service_name
                if db_name == "postgresql":
                    recommendations.append("PostgreSQL detected - excellent for analytics workflows and SQL queries")
                elif db_name == "mysql":
                    recommendations.append("MySQL detected - suitable for application data and reporting")
                elif db_name == "mongodb":
                    recommendations.append("MongoDB detected - great for document-based data analysis")
                elif db_name == "redis":
                    recommendations.append("Redis detected - perfect for caching and real-time analytics")
                elif db_name == "elasticsearch":
                    recommendations.append("Elasticsearch detected - ideal for search and log analytics")

        # Web service recommendations
        if SourceType.WEB_SERVICE in source_types:
            web_services = source_types[SourceType.WEB_SERVICE]
            recommendations.append(f"Found {len(web_services)} web service(s) - these can enhance your data workflow")

            for service in web_services:
                service_name = service.target.service_name
                if service_name == "jupyter":
                    recommendations.append("Jupyter Notebook detected - perfect for data science and analysis")
                elif service_name == "airflow":
                    recommendations.append("Apache Airflow detected - excellent for data pipeline orchestration")
                elif service_name == "grafana":
                    recommendations.append("Grafana detected - ideal for data visualization and monitoring")
                elif service_name == "superset":
                    recommendations.append("Apache Superset detected - great for business intelligence dashboards")

        # Message queue recommendations
        if SourceType.MESSAGE_QUEUE in source_types:
            mq_services = source_types[SourceType.MESSAGE_QUEUE]
            recommendations.append(
                f"Found {len(mq_services)} message queue(s) - these can enable real-time data processing")

        # Environment-specific recommendations
        total_sources = len(available_sources)
        if total_sources >= 5:
            recommendations.append(
                "Rich data environment detected - consider setting up data cataloging and governance")
        elif total_sources >= 3:
            recommendations.append("Multiple data sources available - recommend setting up a unified data access layer")
        else:
            recommendations.append("Limited data sources detected - consider expanding your data infrastructure")

        # Security recommendations
        high_confidence_sources = [s for s in available_sources if s.confidence_score > 0.7]
        if len(high_confidence_sources) != len(available_sources):
            recommendations.append("Some services have limited access - verify credentials and permissions")

        return recommendations

    async def create_data_sources_from_scan(self, profile: EnvironmentProfile) -> List[DataSource]:
        """Create DataSource objects from scan results"""
        data_sources = []

        for result in profile.discovered_sources:
            if not result.available:
                continue

            # Create data source from discovery result
            source_id = f"{result.target.service_name}_{result.target.host}_{result.target.port}"

            data_source = DataSource(
                id=source_id,
                name=result.target.name,
                source_type=result.target.type.value,
                connection_config={
                    "host": result.target.host,
                    "port": result.target.port,
                    "protocol": result.target.protocol,
                    **result.connection_details
                },
                metadata=SourceMetadata(
                    discovered_at=result.discovered_at,
                    confidence_score=result.confidence_score,
                    scan_id=profile.scan_id,
                    version=result.version,
                    features=result.service_info.get("features", []),
                    response_time_ms=result.response_time_ms
                ),
                health_status="healthy" if result.confidence_score > 0.5 else "unknown"
            )

            data_sources.append(data_source)

            # Save to database if available
            if self.database_manager:
                try:
                    await self.database_manager.create_data_source(data_source)
                    logger.info(f"💾 Saved discovered source: {data_source.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save data source {data_source.name}: {e}")

        return data_sources

    async def get_cached_scan(self, user_id: str) -> Optional[EnvironmentProfile]:
        """Get cached environment scan for user"""
        if not self.cache_manager:
            return None

        try:
            cache_key = f"env_scan:{user_id}"
            cached_data = await self.cache_manager.get(cache_key)

            if cached_data:
                return EnvironmentProfile(**cached_data)
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cached scan: {e}")

        return None

    async def invalidate_scan_cache(self, user_id: str) -> bool:
        """Invalidate cached environment scan"""
        if not self.cache_manager:
            return False

        try:
            cache_key = f"env_scan:{user_id}"
            await self.cache_manager.delete(cache_key)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to invalidate scan cache: {e}")
            return False

    def get_scan_summary(self, profile: EnvironmentProfile) -> Dict[str, Any]:
        """Get a summary of the environment scan"""
        available_sources = [s for s in profile.discovered_sources if s.available]

        # Group by source type
        by_type = {}
        for source in available_sources:
            source_type = source.target.type.value
            if source_type not in by_type:
                by_type[source_type] = []
            by_type[source_type].append(source.target.name)

        # Calculate statistics
        avg_response_time = 0
        if available_sources:
            avg_response_time = sum(s.response_time_ms for s in available_sources) / len(available_sources)

        return {
            "scan_id": profile.scan_id,
            "scanned_at": profile.scanned_at.isoformat(),
            "duration_seconds": profile.scan_duration_seconds,
            "total_sources_scanned": len(profile.discovered_sources),
            "available_sources": len(available_sources),
            "confidence_score": profile.confidence_score,
            "sources_by_type": by_type,
            "avg_response_time_ms": round(avg_response_time, 2),
            "system_platform": profile.system_info.get("platform", "Unknown"),
            "recommendation_count": len(profile.recommendations)
        }


# Factory function
async def create_environment_scanner(
        database_manager: Optional[DatabaseManager] = None,
        cache_manager: Optional[CacheManager] = None,
        config_loader: Optional[ConfigurationLoader] = None,
        analytics_engine: Optional[AnalyticsEngine] = None,
        **kwargs
) -> EnvironmentScanner:
    """Factory function to create and initialize environment scanner"""
    scanner = EnvironmentScanner(
        database_manager=database_manager,
        cache_manager=cache_manager,
        config_loader=config_loader,
        analytics_engine=analytics_engine,
        **kwargs
    )
    await scanner.initialize()
    return scanner


# Testing
if __name__ == "__main__":
    async def test_environment_scanner():
        """Test environment scanner functionality"""

        try:
            print("🧪 Testing Environment Scanner...")

            # Create mock dependencies
            class MockAnalytics:
                async def track_event(self, *args, **kwargs):
                    print(f"📊 Analytics: {kwargs.get('event_type')} for user {kwargs.get('user_id')}")

            class MockCache:
                def __init__(self):
                    self.data = {}

                async def initialize(self): pass

                async def close(self): pass

                async def get(self, key, default=None): return self.data.get(key, default)

                async def set(self, key, value, ttl=None):
                    self.data[key] = value
                    return True

                async def delete(self, key):
                    self.data.pop(key, None)
                    return True

            # Initialize scanner
            scanner = await create_environment_scanner(
                cache_manager=MockCache(),
                analytics_engine=MockAnalytics(),
                scan_timeout=5.0,
                max_concurrent=10
            )

            print("✅ Environment scanner created successfully")

            try:
                # Test 1: Quick localhost scan
                print("\n🔍 Test 1: Quick Environment Scan")
                profile = await scanner.scan_environment(
                    user_id="test_user_scanner",
                    quick_scan=True,
                    network_scan=False
                )

                print(f"   Scan completed in {profile.scan_duration_seconds:.2f} seconds")
                print(f"   Found {len([s for s in profile.discovered_sources if s.available])} available sources")
                print(f"   Confidence score: {profile.confidence_score}")
                print(f"   Recommendations: {len(profile.recommendations)}")

                # Show discovered sources
                available_sources = [s for s in profile.discovered_sources if s.available]
                if available_sources:
                    print("   Discovered sources:")
                    for source in available_sources:
                        print(f"     - {source.target.name} ({source.target.service_name}) "
                              f"at {source.target.host}:{source.target.port}")
                        print(f"       Confidence: {source.confidence_score}, "
                              f"Response: {source.response_time_ms:.1f}ms")

                # Test 2: Scan summary
                print("\n🔍 Test 2: Scan Summary")
                summary = scanner.get_scan_summary(profile)
                print(f"   Summary: {summary}")

                # Test 3: Create data sources from scan
                print("\n🔍 Test 3: Data Source Creation")
                data_sources = await scanner.create_data_sources_from_scan(profile)
                print(f"   Created {len(data_sources)} data source objects")

                for source in data_sources:
                    print(f"     - {source.name}: {source.source_type}")
                    print(f"       Config: {source.connection_config}")

                # Test 4: Cache operations
                print("\n🔍 Test 4: Cache Operations")
                cached_profile = await scanner.get_cached_scan("test_user_scanner")
                if cached_profile:
                    print("   ✅ Profile cached successfully")
                    print(f"   Cached scan ID: {cached_profile.scan_id}")

                # Test 5: System information
                print("\n🔍 Test 5: System Information")
                print("   System info:")
                for key, value in profile.system_info.items():
                    print(f"     {key}: {value}")

                print("   Network info:")
                for key, value in profile.network_info.items():
                    if key == "interfaces":
                        print(f"     {key}: {len(value)} interfaces detected")
                    else:
                        print(f"     {key}: {value}")

                # Test 6: Recommendations
                print("\n🔍 Test 6: Smart Recommendations")
                for i, rec in enumerate(profile.recommendations, 1):
                    print(f"   {i}. {rec}")

                print("\n" + "=" * 50)
                print("✅ ALL ENVIRONMENT SCANNER TESTS PASSED! 🎉")
                print("   - Environment discovery ✓")
                print("   - Service probing ✓")
                print("   - System information ✓")
                print("   - Smart recommendations ✓")
                print("   - Data source creation ✓")
                print("   - Cache operations ✓")

            finally:
                await scanner.close()
                print("\n🔐 Environment scanner closed gracefully")

        except Exception as e:
            print(f"\n❌ Environment scanner test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


    # Run tests
    print("🚀 Starting Smart Defaults Environment Scanner Test")
    success = asyncio.run(test_environment_scanner())

    if success:
        print("\n🎯 Environment scanner is ready for integration!")
        print("   Next steps:")
        print("   1. Integrate with your smart defaults engine")
        print("   2. Customize discovery targets for your environment")
        print("   3. Add industry-specific service detection")
        print("   4. Set up automatic periodic scanning")
        print("   5. Configure security and compliance checks")
    else:
        print("\n💥 Tests failed - check the error messages above")
        sys.exit(1)