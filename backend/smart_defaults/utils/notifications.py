"""
Smart Defaults Notifications System
Intelligent notification delivery for recommendations, alerts, and system updates
File location: smart_defaults/utils/notifications.py
"""

import asyncio
import logging
import json
import smtplib
import ssl
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import aiosmtplib
from collections import defaultdict, deque
import hashlib
import uuid

# Import dependencies with fallbacks
try:
    from ..models.user_profile import UserProfile
    from ..models.recommendation import Recommendation
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..utils.monitoring import AnalyticsEngine, EventType
except ImportError:
    # For direct execution, create mock classes
    from dataclasses import dataclass
    from datetime import datetime


    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "data_analyst"


    @dataclass
    class Recommendation:
        id: str = "test_rec"
        user_id: str = "test_user"
        source_id: str = "test_source"
        confidence_score: float = 0.8


    class DatabaseManager:
        async def initialize(self): pass

        async def close(self): pass

        async def store_notification(self, notification): pass

        async def get_user_notifications(self, user_id): return []

        async def update_notification_status(self, notification_id, status): pass

        async def get_notification_history(self, user_id, hours=24): return []


    class CacheManager:
        async def initialize(self): pass

        async def close(self): pass

        async def get(self, key, default=None): return default

        async def set(self, key, value, ttl=None): pass

        async def increment(self, key, amount=1): return amount


    class AnalyticsEngine:
        async def track_event(self, *args, **kwargs): pass


    class EventType:
        NOTIFICATION_SENT = "notification_sent"
        NOTIFICATION_FAILED = "notification_failed"
        NOTIFICATION_OPENED = "notification_opened"
        NOTIFICATION_CLICKED = "notification_clicked"

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications"""
    RECOMMENDATION = "recommendation"
    ALERT = "alert"
    SYSTEM_UPDATE = "system_update"
    FEEDBACK_REQUEST = "feedback_request"
    POLICY_VIOLATION = "policy_violation"
    PERFORMANCE_REPORT = "performance_report"
    ONBOARDING = "onboarding"
    DIGEST = "digest"


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SMS = "sms"
    TEAMS = "teams"
    DISCORD = "discord"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class NotificationTemplate:
    """Template for notifications"""
    id: str
    name: str
    notification_type: NotificationType
    channel: NotificationChannel

    # Template content
    subject_template: str
    body_template: str
    html_template: Optional[str] = None

    # Template metadata
    variables: List[str] = field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.NORMAL
    retry_count: int = 3
    retry_delay_seconds: int = 300

    # Template settings
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NotificationPreferences:
    """User notification preferences"""
    user_id: str

    # Channel preferences
    preferred_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.EMAIL])
    disabled_channels: List[NotificationChannel] = field(default_factory=list)

    # Type preferences
    enabled_types: List[NotificationType] = field(default_factory=lambda: list(NotificationType))
    disabled_types: List[NotificationType] = field(default_factory=list)

    # Timing preferences
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 8  # 8 AM
    timezone: str = "UTC"

    # Frequency preferences
    digest_frequency: str = "daily"  # daily, weekly, monthly
    max_notifications_per_hour: int = 5
    max_notifications_per_day: int = 20

    # Contact information
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    webhook_url: Optional[str] = None

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NotificationMessage:
    """Individual notification message"""
    id: str
    user_id: str
    notification_type: NotificationType
    channel: NotificationChannel
    priority: NotificationPriority

    # Message content
    subject: str
    body: str
    html_body: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    # Message metadata
    template_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None

    # Response tracking
    opened: bool = False
    clicked: bool = False
    replied: bool = False


@dataclass
class NotificationConfig:
    """Configuration for notification system"""

    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    from_email: str = "noreply@smartdefaults.com"
    from_name: str = "Smart Defaults"

    # Slack settings
    slack_bot_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None

    # Webhook settings
    webhook_timeout: int = 30
    webhook_retries: int = 3

    # Rate limiting
    global_rate_limit_per_minute: int = 100
    per_user_rate_limit_per_hour: int = 10

    # Retry settings
    default_retry_delay: int = 300  # 5 minutes
    max_retry_delay: int = 3600  # 1 hour

    # Template settings
    template_cache_ttl: int = 3600

    # Performance settings
    batch_size: int = 50
    worker_threads: int = 5


class RateLimiter:
    """Rate limiting for notifications"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.user_counters = defaultdict(lambda: defaultdict(int))
        self.global_counter = 0
        self.window_start = datetime.now(timezone.utc)

    async def can_send(self, user_id: str, preferences: NotificationPreferences) -> bool:
        """Check if user can receive notification based on rate limits"""
        now = datetime.now(timezone.utc)

        # Check global rate limit (per minute)
        global_key = f"rate_limit:global:{now.strftime('%Y-%m-%d-%H-%M')}"
        global_count = await self.cache.get(global_key, 0)

        if global_count >= 100:  # Default global limit
            logger.warning(f"⚠️ Global rate limit exceeded: {global_count}/100")
            return False

        # Check user hourly rate limit
        hourly_key = f"rate_limit:user:{user_id}:{now.strftime('%Y-%m-%d-%H')}"
        hourly_count = await self.cache.get(hourly_key, 0)

        if hourly_count >= preferences.max_notifications_per_hour:
            logger.warning(
                f"⚠️ User {user_id} hourly rate limit exceeded: {hourly_count}/{preferences.max_notifications_per_hour}")
            return False

        # Check user daily rate limit
        daily_key = f"rate_limit:user:{user_id}:{now.strftime('%Y-%m-%d')}"
        daily_count = await self.cache.get(daily_key, 0)

        if daily_count >= preferences.max_notifications_per_day:
            logger.warning(
                f"⚠️ User {user_id} daily rate limit exceeded: {daily_count}/{preferences.max_notifications_per_day}")
            return False

        return True

    async def record_send(self, user_id: str):
        """Record that a notification was sent"""
        now = datetime.now(timezone.utc)

        # Increment counters
        global_key = f"rate_limit:global:{now.strftime('%Y-%m-%d-%H-%M')}"
        hourly_key = f"rate_limit:user:{user_id}:{now.strftime('%Y-%m-%d-%H')}"
        daily_key = f"rate_limit:user:{user_id}:{now.strftime('%Y-%m-%d')}"

        await self.cache.increment(global_key)
        await self.cache.increment(hourly_key)
        await self.cache.increment(daily_key)

        # Set TTLs
        await self.cache.set(global_key, await self.cache.get(global_key), ttl=60)
        await self.cache.set(hourly_key, await self.cache.get(hourly_key), ttl=3600)
        await self.cache.set(daily_key, await self.cache.get(daily_key), ttl=86400)


class NotificationDeliveryProvider:
    """Base class for notification delivery providers"""

    def __init__(self, channel: NotificationChannel, config: Dict[str, Any]):
        self.channel = channel
        self.config = config

    async def send(self, message: NotificationMessage) -> bool:
        """Send a notification message"""
        raise NotImplementedError

    async def validate_config(self) -> bool:
        """Validate provider configuration"""
        return True


class EmailProvider(NotificationDeliveryProvider):
    """Email notification provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(NotificationChannel.EMAIL, config)

    async def send(self, message: NotificationMessage) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message.subject
            msg['From'] = f"{self.config.get('from_name', 'Smart Defaults')} <{self.config.get('from_email')}>"
            msg['To'] = self.config.get('to_email', 'user@example.com')

            # Add text part
            text_part = MIMEText(message.body, 'plain')
            msg.attach(text_part)

            # Add HTML part if available
            if message.html_body:
                html_part = MIMEText(message.html_body, 'html')
                msg.attach(html_part)

            # Add attachments
            for attachment in message.attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.get('data', b''))
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.get("filename", "attachment")}'
                )
                msg.attach(part)

            # Send email
            await aiosmtplib.send(
                msg,
                hostname=self.config.get('smtp_host', 'localhost'),
                port=self.config.get('smtp_port', 587),
                start_tls=self.config.get('smtp_use_tls', True),
                username=self.config.get('smtp_username'),
                password=self.config.get('smtp_password')
            )

            logger.info(f"📧 Email sent successfully to {msg['To']}")
            return True

        except Exception as e:
            logger.error(f"❌ Email sending failed: {e}")
            return False


class SlackProvider(NotificationDeliveryProvider):
    """Slack notification provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(NotificationChannel.SLACK, config)
        self.bot_token = config.get('slack_bot_token')

    async def send(self, message: NotificationMessage) -> bool:
        """Send Slack notification"""
        try:
            if not self.bot_token:
                logger.warning("⚠️ Slack bot token not configured")
                return False

            # Format message for Slack
            slack_message = {
                "channel": self.config.get('channel', '#general'),
                "text": message.subject,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": message.subject
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message.body
                        }
                    }
                ]
            }

            # Add priority indicators
            if message.priority in [NotificationPriority.HIGH, NotificationPriority.URGENT,
                                    NotificationPriority.CRITICAL]:
                slack_message["blocks"].insert(0, {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *{message.priority.value.upper()} PRIORITY*"
                    }
                })

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.bot_token}',
                    'Content-Type': 'application/json'
                }

                async with session.post(
                        'https://slack.com/api/chat.postMessage',
                        headers=headers,
                        json=slack_message
                ) as response:
                    result = await response.json()

                    if result.get('ok'):
                        logger.info(f"💬 Slack message sent successfully")
                        return True
                    else:
                        logger.error(f"❌ Slack sending failed: {result.get('error')}")
                        return False

        except Exception as e:
            logger.error(f"❌ Slack sending failed: {e}")
            return False


class WebhookProvider(NotificationDeliveryProvider):
    """Webhook notification provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(NotificationChannel.WEBHOOK, config)

    async def send(self, message: NotificationMessage) -> bool:
        """Send webhook notification"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.warning("⚠️ Webhook URL not configured")
                return False

            # Prepare webhook payload
            payload = {
                "message_id": message.id,
                "user_id": message.user_id,
                "type": message.notification_type.value,
                "priority": message.priority.value,
                "subject": message.subject,
                "body": message.body,
                "created_at": message.created_at.isoformat(),
                "variables": message.variables
            }

            # Send webhook
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"🔗 Webhook sent successfully to {webhook_url}")
                        return True
                    else:
                        logger.error(f"❌ Webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"❌ Webhook sending failed: {e}")
            return False


class InAppProvider(NotificationDeliveryProvider):
    """In-app notification provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(NotificationChannel.IN_APP, config)
        self.notifications_store = config.get('notifications_store', {})

    async def send(self, message: NotificationMessage) -> bool:
        """Send in-app notification"""
        try:
            # Store notification in user's in-app queue
            user_notifications = self.notifications_store.get(message.user_id, [])

            notification_data = {
                "id": message.id,
                "type": message.notification_type.value,
                "priority": message.priority.value,
                "subject": message.subject,
                "body": message.body,
                "created_at": message.created_at.isoformat(),
                "read": False,
                "clicked": False
            }

            user_notifications.append(notification_data)
            self.notifications_store[message.user_id] = user_notifications

            # Keep only last 100 notifications per user
            if len(user_notifications) > 100:
                self.notifications_store[message.user_id] = user_notifications[-100:]

            logger.info(f"📱 In-app notification stored for user {message.user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ In-app notification failed: {e}")
            return False

    async def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get user's in-app notifications"""
        notifications = self.notifications_store.get(user_id, [])

        if unread_only:
            notifications = [n for n in notifications if not n.get('read', False)]

        return sorted(notifications, key=lambda x: x['created_at'], reverse=True)

    async def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Mark notification as read"""
        user_notifications = self.notifications_store.get(user_id, [])

        for notification in user_notifications:
            if notification['id'] == notification_id:
                notification['read'] = True
                return True

        return False


class NotificationDigestEngine:
    """Handles digest notifications (batching multiple notifications)"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.pending_digests = defaultdict(list)

    async def should_send_digest(self, user_id: str, preferences: NotificationPreferences) -> bool:
        """Check if it's time to send a digest"""
        now = datetime.now(timezone.utc)

        if preferences.digest_frequency == "daily":
            # Send at 9 AM in user's timezone
            return now.hour == 9
        elif preferences.digest_frequency == "weekly":
            # Send on Monday at 9 AM
            return now.weekday() == 0 and now.hour == 9
        elif preferences.digest_frequency == "monthly":
            # Send on 1st of month at 9 AM
            return now.day == 1 and now.hour == 9

        return False

    async def create_digest_message(self, user_id: str,
                                    notifications: List[NotificationMessage]) -> NotificationMessage:
        """Create a digest message from multiple notifications"""
        if not notifications:
            return None

        # Group notifications by type
        grouped = defaultdict(list)
        for notif in notifications:
            grouped[notif.notification_type].append(notif)

        # Build digest content
        subject = f"Smart Defaults Digest - {len(notifications)} Updates"
        body_parts = [
            f"Here's your digest of {len(notifications)} notifications:\n"
        ]

        for notif_type, notifs in grouped.items():
            body_parts.append(f"\n📋 {notif_type.value.title()} ({len(notifs)} items):")
            for notif in notifs[:5]:  # Limit to 5 per type
                body_parts.append(f"  • {notif.subject}")

            if len(notifs) > 5:
                body_parts.append(f"  ... and {len(notifs) - 5} more")

        body_parts.append("\nLogin to Smart Defaults to see full details.")

        # Create digest message
        digest_message = NotificationMessage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            notification_type=NotificationType.DIGEST,
            channel=NotificationChannel.EMAIL,  # Default to email for digests
            priority=NotificationPriority.NORMAL,
            subject=subject,
            body="\n".join(body_parts)
        )

        return digest_message


class NotificationEngine:
    """Main notification engine orchestrating all components"""

    def __init__(self,
                 config: NotificationConfig,
                 db_manager: DatabaseManager,
                 cache_manager: CacheManager,
                 analytics_engine: AnalyticsEngine):

        self.config = config
        self.db = db_manager
        self.cache = cache_manager
        self.analytics = analytics_engine

        # Initialize components
        self.rate_limiter = RateLimiter(cache_manager)
        self.digest_engine = NotificationDigestEngine(db_manager)

        # Initialize providers
        self.providers = {}
        self._initialize_providers()

        # Template cache
        self.template_cache = {}

        # Retry queue
        self.retry_queue = deque()

        # Worker tasks
        self.worker_tasks = []
        self.running = False

    def _initialize_providers(self):
        """Initialize notification delivery providers"""
        # Email provider
        email_config = {
            'smtp_host': self.config.smtp_host,
            'smtp_port': self.config.smtp_port,
            'smtp_username': self.config.smtp_username,
            'smtp_password': self.config.smtp_password,
            'smtp_use_tls': self.config.smtp_use_tls,
            'from_email': self.config.from_email,
            'from_name': self.config.from_name
        }
        self.providers[NotificationChannel.EMAIL] = EmailProvider(email_config)

        # Slack provider
        if self.config.slack_bot_token:
            slack_config = {
                'slack_bot_token': self.config.slack_bot_token,
                'slack_signing_secret': self.config.slack_signing_secret
            }
            self.providers[NotificationChannel.SLACK] = SlackProvider(slack_config)

        # Webhook provider
        webhook_config = {
            'timeout': self.config.webhook_timeout,
            'retries': self.config.webhook_retries
        }
        self.providers[NotificationChannel.WEBHOOK] = WebhookProvider(webhook_config)

        # In-app provider
        in_app_config = {}
        self.providers[NotificationChannel.IN_APP] = InAppProvider(in_app_config)

    async def start(self):
        """Start the notification engine"""
        self.running = True

        # Start worker tasks
        for i in range(self.config.worker_threads):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(task)

        # Start retry processor
        retry_task = asyncio.create_task(self._retry_processor())
        self.worker_tasks.append(retry_task)

        # Start digest processor
        digest_task = asyncio.create_task(self._digest_processor())
        self.worker_tasks.append(digest_task)

        logger.info(f"🚀 Notification engine started with {len(self.worker_tasks)} workers")

    async def stop(self):
        """Stop the notification engine"""
        self.running = False

        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        logger.info("🛑 Notification engine stopped")

    async def send_notification(self,
                                user_id: str,
                                notification_type: NotificationType,
                                subject: str,
                                body: str,
                                priority: NotificationPriority = NotificationPriority.NORMAL,
                                variables: Dict[str, Any] = None,
                                template_id: str = None,
                                scheduled_at: datetime = None) -> str:
        """Send a notification to a user"""

        # Get user preferences
        preferences = await self.get_user_preferences(user_id)

        # Check if notification type is enabled
        if notification_type in preferences.disabled_types:
            logger.info(f"📵 Notification type {notification_type.value} disabled for user {user_id}")
            return None

        # Check rate limits
        if not await self.rate_limiter.can_send(user_id, preferences):
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id}")
            return None

        # Check quiet hours
        if await self._is_quiet_hours(preferences):
            # Schedule for later if not urgent
            if priority not in [NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
                scheduled_at = await self._next_active_time(preferences)
                logger.info(f"🔇 Scheduling notification for user {user_id} after quiet hours")

        # Determine channels to use
        channels = await self._get_delivery_channels(preferences, priority)

        notification_ids = []

        # Create notification for each channel
        for channel in channels:
            # Create notification message
            message = NotificationMessage(
                id=str(uuid.uuid4()),
                user_id=user_id,
                notification_type=notification_type,
                channel=channel,
                priority=priority,
                subject=subject,
                body=body,
                template_id=template_id,
                variables=variables or {},
                scheduled_at=scheduled_at
            )

            # Store in database
            await self.db.store_notification(message)

            # Queue for sending
            await self._queue_notification(message)

            notification_ids.append(message.id)

        logger.info(f"📤 Queued {len(notification_ids)} notifications for user {user_id}")
        return notification_ids[0] if notification_ids else None

    async def send_recommendation_notification(self, recommendation: Recommendation, user_profile: UserProfile):
        """Send notification for a new recommendation"""

        # Determine notification content based on confidence
        if recommendation.confidence_score >= 0.85:
            subject = f"🎯 Auto-Connected: {recommendation.source_id}"
            body = f"We automatically connected you to {recommendation.source_id} based on your {user_profile.role} role. Confidence: {recommendation.confidence_score:.0%}"
            priority = NotificationPriority.NORMAL
        elif recommendation.confidence_score >= 0.60:
            subject = f"💡 Recommended: {recommendation.source_id}"
            body = f"We recommend connecting to {recommendation.source_id} for your {user_profile.role} workflow. Confidence: {recommendation.confidence_score:.0%}"
            priority = NotificationPriority.NORMAL
        else:
            subject = f"📋 Available: {recommendation.source_id}"
            body = f"{recommendation.source_id} is available to connect. Review if it fits your {user_profile.role} needs."
            priority = NotificationPriority.LOW

        await self.send_notification(
            user_id=user_profile.user_id,
            notification_type=NotificationType.RECOMMENDATION,
            subject=subject,
            body=body,
            priority=priority,
            variables={
                'recommendation_id': recommendation.id,
                'source_id': recommendation.source_id,
                'confidence_score': recommendation.confidence_score,
                'user_role': user_profile.role
            }
        )

    async def get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences with caching"""
        cache_key = f"notification_preferences:{user_id}"

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return NotificationPreferences(**json.loads(cached))

        # TODO: Load from database
        # For now, return defaults
        preferences = NotificationPreferences(user_id=user_id)

        # Cache for 1 hour
        await self.cache.set(cache_key, json.dumps(asdict(preferences)), ttl=3600)

        return preferences

    async def update_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Update user notification preferences"""
        # TODO: Save to database
        logger.info(f"💾 [PLACEHOLDER] Saving preferences for user {user_id}")

        # Update cache
        cache_key = f"notification_preferences:{user_id}"
        await self.cache.set(cache_key, json.dumps(asdict(preferences)), ttl=3600)

    async def _queue_notification(self, message: NotificationMessage):
        """Queue notification for processing"""
        queue_key = f"notification_queue:{message.priority.value}"

        # Add to priority queue
        notification_data = json.dumps(asdict(message), default=str)
        await self.cache.set(f"notification:{message.id}", notification_data, ttl=86400)

        # Add to processing queue
        await self.cache.set(f"{queue_key}:{message.id}", message.id, ttl=86400)

        logger.debug(f"📥 Queued notification {message.id} with priority {message.priority.value}")

    async def _worker_loop(self, worker_name: str):
        """Main worker loop for processing notifications"""
        logger.info(f"🔄 Starting worker {worker_name}")

        while self.running:
            try:
                # Process notifications by priority
                for priority in [NotificationPriority.CRITICAL, NotificationPriority.URGENT,
                                 NotificationPriority.HIGH, NotificationPriority.NORMAL, NotificationPriority.LOW]:

                    queue_key = f"notification_queue:{priority.value}"

                    # Get batch of notifications
                    notification_ids = await self._get_queued_notifications(queue_key, self.config.batch_size)

                    if notification_ids:
                        await self._process_notification_batch(notification_ids, worker_name)
                        break  # Process higher priority first

                # Small delay to prevent busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"❌ Worker {worker_name} error: {e}")
                await asyncio.sleep(5)

        logger.info(f"🛑 Worker {worker_name} stopped")

    async def _get_queued_notifications(self, queue_key: str, limit: int) -> List[str]:
        """Get notifications from queue"""
        # TODO: Implement proper queue with Redis/Database
        # For now, return placeholder
        logger.debug(f"📥 [PLACEHOLDER] Getting {limit} notifications from {queue_key}")
        return []

    async def _process_notification_batch(self, notification_ids: List[str], worker_name: str):
        """Process a batch of notifications"""
        for notification_id in notification_ids:
            try:
                await self._process_single_notification(notification_id, worker_name)
            except Exception as e:
                logger.error(f"❌ {worker_name} failed to process {notification_id}: {e}")

    async def _process_single_notification(self, notification_id: str, worker_name: str):
        """Process a single notification"""
        # Load notification from cache/database
        notification_data = await self.cache.get(f"notification:{notification_id}")
        if not notification_data:
            logger.warning(f"⚠️ Notification {notification_id} not found")
            return

        message = NotificationMessage(**json.loads(notification_data))

        # Check if scheduled for future
        if message.scheduled_at and message.scheduled_at > datetime.now(timezone.utc):
            logger.debug(f"⏰ Notification {notification_id} scheduled for {message.scheduled_at}")
            return

        # Get user preferences
        preferences = await self.get_user_preferences(message.user_id)

        # Final rate limit check
        if not await self.rate_limiter.can_send(message.user_id, preferences):
            logger.warning(f"⚠️ Rate limit hit during processing for user {message.user_id}")
            return

        # Get provider
        provider = self.providers.get(message.channel)
        if not provider:
            logger.error(f"❌ No provider for channel {message.channel.value}")
            await self._mark_notification_failed(message, "No provider available")
            return

        # Update provider config with user-specific info
        await self._configure_provider(provider, message, preferences)

        # Send notification
        success = await provider.send(message)

        if success:
            # Mark as sent
            message.status = NotificationStatus.SENT
            message.sent_at = datetime.now(timezone.utc)

            # Record rate limit
            await self.rate_limiter.record_send(message.user_id)

            # Track analytics
            await self.analytics.track_event(
                EventType.NOTIFICATION_SENT,
                user_id=message.user_id,
                notification_type=message.notification_type.value,
                channel=message.channel.value,
                priority=message.priority.value
            )

            logger.info(f"✅ {worker_name} sent notification {notification_id} via {message.channel.value}")

        else:
            # Handle failure
            await self._handle_notification_failure(message)

        # Update database
        await self.db.update_notification_status(notification_id, message.status.value)

    async def _configure_provider(self, provider: NotificationDeliveryProvider, message: NotificationMessage,
                                  preferences: NotificationPreferences):
        """Configure provider with user-specific settings"""
        if message.channel == NotificationChannel.EMAIL and preferences.email:
            provider.config['to_email'] = preferences.email
        elif message.channel == NotificationChannel.SLACK and preferences.slack_user_id:
            provider.config['channel'] = f"@{preferences.slack_user_id}"
        elif message.channel == NotificationChannel.WEBHOOK and preferences.webhook_url:
            provider.config['webhook_url'] = preferences.webhook_url

    async def _handle_notification_failure(self, message: NotificationMessage):
        """Handle failed notification delivery"""
        message.retry_count += 1

        if message.retry_count <= message.max_retries:
            # Schedule retry
            message.status = NotificationStatus.RETRYING
            retry_delay = min(
                self.config.default_retry_delay * (2 ** (message.retry_count - 1)),
                self.config.max_retry_delay
            )

            retry_time = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
            message.scheduled_at = retry_time

            # Add to retry queue
            self.retry_queue.append((retry_time, message.id))

            logger.warning(
                f"🔄 Scheduling retry {message.retry_count}/{message.max_retries} for {message.id} in {retry_delay}s")

        else:
            # Max retries exceeded
            await self._mark_notification_failed(message, "Max retries exceeded")

    async def _mark_notification_failed(self, message: NotificationMessage, error: str):
        """Mark notification as permanently failed"""
        message.status = NotificationStatus.FAILED
        message.last_error = error

        # Track analytics
        await self.analytics.track_event(
            EventType.NOTIFICATION_FAILED,
            user_id=message.user_id,
            notification_type=message.notification_type.value,
            channel=message.channel.value,
            error=error
        )

        logger.error(f"❌ Notification {message.id} failed permanently: {error}")

    async def _retry_processor(self):
        """Process retry queue"""
        logger.info("🔄 Starting retry processor")

        while self.running:
            try:
                now = datetime.now(timezone.utc)

                # Process due retries
                while self.retry_queue and self.retry_queue[0][0] <= now:
                    retry_time, notification_id = self.retry_queue.popleft()

                    # Re-queue for processing
                    notification_data = await self.cache.get(f"notification:{notification_id}")
                    if notification_data:
                        message = NotificationMessage(**json.loads(notification_data))
                        await self._queue_notification(message)
                        logger.info(f"🔄 Retry queued for notification {notification_id}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Retry processor error: {e}")
                await asyncio.sleep(60)

    async def _digest_processor(self):
        """Process digest notifications"""
        logger.info("📊 Starting digest processor")

        while self.running:
            try:
                # Check every hour for digest opportunities
                await asyncio.sleep(3600)

                # TODO: Implement digest logic
                logger.debug("📊 [PLACEHOLDER] Checking for digest opportunities")

            except Exception as e:
                logger.error(f"❌ Digest processor error: {e}")
                await asyncio.sleep(3600)

    async def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is in user's quiet hours"""
        now = datetime.now(timezone.utc)

        # TODO: Convert to user's timezone
        current_hour = now.hour

        start_hour = preferences.quiet_hours_start
        end_hour = preferences.quiet_hours_end

        if start_hour <= end_hour:
            # Same day quiet hours (e.g., 22:00 to 8:00 next day)
            return start_hour <= current_hour <= end_hour
        else:
            # Crosses midnight (e.g., 22:00 to 8:00)
            return current_hour >= start_hour or current_hour <= end_hour

    async def _next_active_time(self, preferences: NotificationPreferences) -> datetime:
        """Calculate next active time outside quiet hours"""
        now = datetime.now(timezone.utc)

        # TODO: Implement proper timezone conversion
        # For now, just add to end of quiet hours
        next_active = now.replace(hour=preferences.quiet_hours_end, minute=0, second=0)

        if next_active <= now:
            next_active += timedelta(days=1)

        return next_active

    async def _get_delivery_channels(self, preferences: NotificationPreferences, priority: NotificationPriority) -> \
    List[NotificationChannel]:
        """Determine which channels to use for delivery"""
        channels = []

        # Critical/Urgent messages use all available channels
        if priority in [NotificationPriority.CRITICAL, NotificationPriority.URGENT]:
            for channel in preferences.preferred_channels:
                if channel not in preferences.disabled_channels:
                    channels.append(channel)

            # Always add in-app for critical messages
            if NotificationChannel.IN_APP not in channels:
                channels.append(NotificationChannel.IN_APP)

        else:
            # Normal priority uses preferred channels
            for channel in preferences.preferred_channels:
                if channel not in preferences.disabled_channels:
                    channels.append(channel)

        # Fallback to email if no channels available
        if not channels and NotificationChannel.EMAIL not in preferences.disabled_channels:
            channels.append(NotificationChannel.EMAIL)

        return channels

    async def get_notification_stats(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get notification statistics for a user"""
        # TODO: Implement proper stats from database
        logger.info(f"📈 [PLACEHOLDER] Getting notification stats for user {user_id} for {days} days")

        return {
            "total_sent": 42,
            "total_delivered": 40,
            "total_opened": 25,
            "total_clicked": 12,
            "delivery_rate": 0.95,
            "open_rate": 0.63,
            "click_rate": 0.30,
            "by_channel": {
                "email": {"sent": 20, "delivered": 19, "opened": 15},
                "slack": {"sent": 15, "delivered": 15, "opened": 8},
                "in_app": {"sent": 7, "delivered": 6, "opened": 2}
            },
            "by_type": {
                "recommendation": {"sent": 25, "opened": 18},
                "alert": {"sent": 10, "opened": 5},
                "system_update": {"sent": 7, "opened": 2}
            }
        }

    async def mark_notification_opened(self, notification_id: str, user_id: str):
        """Mark notification as opened"""
        # Update database
        await self.db.update_notification_status(notification_id, "opened")

        # Track analytics
        await self.analytics.track_event(
            EventType.NOTIFICATION_OPENED,
            user_id=user_id,
            notification_id=notification_id
        )

        logger.info(f"👀 Notification {notification_id} marked as opened by user {user_id}")

    async def mark_notification_clicked(self, notification_id: str, user_id: str):
        """Mark notification as clicked"""
        # Update database
        await self.db.update_notification_status(notification_id, "clicked")

        # Track analytics
        await self.analytics.track_event(
            EventType.NOTIFICATION_CLICKED,
            user_id=user_id,
            notification_id=notification_id
        )

        logger.info(f"🖱️ Notification {notification_id} marked as clicked by user {user_id}")

    async def get_user_notifications(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's notification history"""
        notifications = await self.db.get_user_notifications(user_id)

        # Apply pagination
        start = offset
        end = offset + limit

        return notifications[start:end]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for notification system"""
        return {
            "status": "healthy" if self.running else "stopped",
            "workers_running": len([t for t in self.worker_tasks if not t.done()]),
            "retry_queue_size": len(self.retry_queue),
            "providers_available": list(self.providers.keys()),
            "cache_connected": await self._check_cache_health(),
            "database_connected": await self._check_database_health()
        }

    async def _check_cache_health(self) -> bool:
        """Check cache connectivity"""
        try:
            await self.cache.set("health_check", "ok", ttl=60)
            return await self.cache.get("health_check") == "ok"
        except:
            return False

    async def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            # TODO: Implement database health check
            return True
        except:
            return False


# Example usage and testing
async def example_usage():
    """Example of how to use the notification system"""

    # Initialize components
    config = NotificationConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_username="your-email@gmail.com",
        smtp_password="your-app-password",
        from_email="noreply@smartdefaults.com",
        slack_bot_token="xoxb-your-slack-token"
    )

    db_manager = DatabaseManager()
    cache_manager = CacheManager()
    analytics_engine = AnalyticsEngine()

    # Initialize and start notification engine
    notification_engine = NotificationEngine(config, db_manager, cache_manager, analytics_engine)
    await notification_engine.start()

    try:
        # Send a recommendation notification
        user_profile = UserProfile(
            id="profile_1",
            user_id="user_123",
            role="data_analyst"
        )

        recommendation = Recommendation(
            id="rec_1",
            user_id="user_123",
            source_id="postgresql_analytics_db",
            confidence_score=0.87
        )

        await notification_engine.send_recommendation_notification(recommendation, user_profile)

        # Send a custom notification
        await notification_engine.send_notification(
            user_id="user_123",
            notification_type=NotificationType.ALERT,
            subject="🚨 Security Alert",
            body="Unusual access pattern detected in your data sources.",
            priority=NotificationPriority.HIGH,
            variables={
                "alert_type": "security",
                "severity": "medium"
            }
        )

        # Update user preferences
        preferences = await notification_engine.get_user_preferences("user_123")
        preferences.preferred_channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
        preferences.quiet_hours_start = 23
        preferences.quiet_hours_end = 7
        await notification_engine.update_user_preferences("user_123", preferences)

        # Get notification stats
        stats = await notification_engine.get_notification_stats("user_123")
        print(f"📊 Notification stats: {stats}")

        # Check health
        health = await notification_engine.health_check()
        print(f"💚 System health: {health}")

        # Let it run for a bit
        await asyncio.sleep(10)

    finally:
        await notification_engine.stop()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(example_usage())