import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional

LOG_FORMAT = os.getenv('LOG_FORMAT', 'standard').lower()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON for cloud observability."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%SZ"),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
            log_data.update(record.extra_data)
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)

def setup_logger(name: str = 'mbmr') -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if LOG_FORMAT == 'json' or os.getenv('RENDER') or os.getenv('VERCEL'):
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# Primary logger instance
logger = setup_logger('mbmr')
security_logger = setup_logger('mbmr.security')

def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redacts sensitive information like PINs and API keys from log records."""
    redacted = {}
    sensitive_keys = {'pin', 'password', 'api_key', 'gemini_key', 'tmdb_key', 'encryption_key', 'session_secret', 'authorization'}
    for k, v in data.items():
        if k.lower() in sensitive_keys:
            redacted[k] = '[REDACTED]'
        elif isinstance(v, dict):
            redacted[k] = sanitize_data(v)
        else:
            redacted[k] = v
    return redacted

def log_security_event(event_type: str, details: Optional[Dict[str, Any]] = None, level: str = 'info'):
    """Logs a structured security audit event."""
    safe_details = sanitize_data(details or {})
    extra = {
        'extra_data': {
            'event_type': event_type,
            'security_audit': True,
            **safe_details
        }
    }
    log_func = getattr(security_logger, level.lower(), security_logger.info)
    log_func(f"[SECURITY] {event_type} - {safe_details}", extra=extra)

def log_auth_attempt(username: str, ip: str, success: bool, reason: Optional[str] = None):
    """Logs authentication success/failure with IP and reason."""
    level = 'info' if success else 'warning'
    details = {
        'username': username,
        'ip': ip,
        'status': 'SUCCESS' if success else 'FAILED'
    }
    if reason:
        details['reason'] = reason
    log_security_event('AUTH_ATTEMPT', details, level=level)

def log_rate_limit_blocked(ip: str, endpoint: str, retry_after: int):
    """Logs rate-limit threshold violations."""
    details = {
        'ip': ip,
        'endpoint': endpoint,
        'retry_after_seconds': retry_after
    }
    log_security_event('RATE_LIMIT_EXCEEDED', details, level='warning')
