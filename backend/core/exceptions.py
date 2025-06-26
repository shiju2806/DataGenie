# core/exceptions.py - Custom Exceptions for DataGenie
"""
Custom exception classes for DataGenie Multi-Source Analytics
"""

from typing import Optional, Dict, Any


class DataGenieException(Exception):
    """Base exception for all DataGenie errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AccessDeniedException(DataGenieException):
    """Raised when access to a resource is denied"""

    def __init__(self, user_id: str, resource_id: str, action: str, reason: str = "Access denied"):
        self.user_id = user_id
        self.resource_id = resource_id
        self.action = action
        message = f"Access denied for user {user_id} to resource {resource_id} (action: {action}). Reason: {reason}"
        super().__init__(message, {
            "user_id": user_id,
            "resource_id": resource_id,
            "action": action,
            "reason": reason
        })


class InsufficientPermissionsException(DataGenieException):
    """Raised when user lacks required permissions"""

    def __init__(self, user_id: str, required_permissions: list, reason: str = "Insufficient permissions"):
        self.user_id = user_id
        self.required_permissions = required_permissions
        message = f"User {user_id} lacks required permissions: {required_permissions}. Reason: {reason}"
        super().__init__(message, {
            "user_id": user_id,
            "required_permissions": required_permissions,
            "reason": reason
        })


class ResourceNotFoundException(DataGenieException):
    """Raised when a requested resource is not found"""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(message, {
            "resource_type": resource_type,
            "resource_id": resource_id
        })


class DataSourceConnectionException(DataGenieException):
    """Raised when connection to data source fails"""

    def __init__(self, source_id: str, connection_error: str):
        self.source_id = source_id
        self.connection_error = connection_error
        message = f"Failed to connect to data source {source_id}: {connection_error}"
        super().__init__(message, {
            "source_id": source_id,
            "connection_error": connection_error
        })


class DataValidationException(DataGenieException):
    """Raised when data validation fails"""

    def __init__(self, field_name: str, validation_error: str, invalid_value: Any = None):
        self.field_name = field_name
        self.validation_error = validation_error
        self.invalid_value = invalid_value
        message = f"Validation error for field {field_name}: {validation_error}"
        super().__init__(message, {
            "field_name": field_name,
            "validation_error": validation_error,
            "invalid_value": invalid_value
        })


class ConflictResolutionException(DataGenieException):
    """Raised when conflict resolution fails"""

    def __init__(self, conflict_id: str, resolution_error: str):
        self.conflict_id = conflict_id
        self.resolution_error = resolution_error
        message = f"Failed to resolve conflict {conflict_id}: {resolution_error}"
        super().__init__(message, {
            "conflict_id": conflict_id,
            "resolution_error": resolution_error
        })


class CacheException(DataGenieException):
    """Raised when cache operations fail"""

    def __init__(self, operation: str, cache_key: str, error: str):
        self.operation = operation
        self.cache_key = cache_key
        self.error = error
        message = f"Cache {operation} failed for key {cache_key}: {error}"
        super().__init__(message, {
            "operation": operation,
            "cache_key": cache_key,
            "error": error
        })


class MultiSourceJoinException(DataGenieException):
    """Raised when multi-source data joining fails"""

    def __init__(self, source_ids: list, join_error: str):
        self.source_ids = source_ids
        self.join_error = join_error
        message = f"Failed to join sources {source_ids}: {join_error}"
        super().__init__(message, {
            "source_ids": source_ids,
            "join_error": join_error
        })


class QueryPlanningException(DataGenieException):
    """Raised when query planning fails"""

    def __init__(self, query: str, planning_error: str):
        self.query = query
        self.planning_error = planning_error
        message = f"Query planning failed for: {query[:100]}... Error: {planning_error}"
        super().__init__(message, {
            "query": query,
            "planning_error": planning_error
        })


class ConfigurationException(DataGenieException):
    """Raised when configuration is invalid"""

    def __init__(self, config_key: str, config_error: str):
        self.config_key = config_key
        self.config_error = config_error
        message = f"Configuration error for {config_key}: {config_error}"
        super().__init__(message, {
            "config_key": config_key,
            "config_error": config_error
        })