import unittest
import json
import logging
from io import StringIO

from backend.logger import (
    JSONFormatter, sanitize_data, log_security_event, 
    log_auth_attempt, log_rate_limit_blocked, security_logger
)

class TestLogger(unittest.TestCase):
    def test_01_json_formatter_structure(self):
        """Test that JSONFormatter outputs structured, parseable JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='mbmr.test',
            level=logging.INFO,
            pathname='test.py',
            lineno=42,
            msg='Test system message',
            args=(),
            exc_info=None
        )
        record.extra_data = {'action': 're-indexing', 'user_id': 123}
        formatted = formatter.format(record)

        data = json.loads(formatted)
        self.assertEqual(data['level'], 'INFO')
        self.assertEqual(data['name'], 'mbmr.test')
        self.assertEqual(data['message'], 'Test system message')
        self.assertEqual(data['action'], 're-indexing')
        self.assertEqual(data['user_id'], 123)
        self.assertIn('timestamp', data)

    def test_02_sensitive_data_sanitization(self):
        """Test that sensitive credentials (PINs, API keys) are redacted."""
        raw_payload = {
            'username': 'cinephile_99',
            'pin': '1234',
            'api_key': 'secret_key_123',
            'gemini_key': 'gemini_secret_xyz',
            'nested': {
                'password': 'mypassword',
                'normal_field': 'hello'
            }
        }
        sanitized = sanitize_data(raw_payload)

        self.assertEqual(sanitized['username'], 'cinephile_99')
        self.assertEqual(sanitized['pin'], '[REDACTED]')
        self.assertEqual(sanitized['api_key'], '[REDACTED]')
        self.assertEqual(sanitized['gemini_key'], '[REDACTED]')
        self.assertEqual(sanitized['nested']['password'], '[REDACTED]')
        self.assertEqual(sanitized['nested']['normal_field'], 'hello')

    def test_03_security_event_logging(self):
        """Test logging of security events with extra metadata."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        security_logger.addHandler(handler)

        try:
            log_security_event('LOGIN_LOCKOUT', {'username': 'test_locked', 'pin': '9999', 'attempts': 5})
            log_auth_attempt('test_user', '127.0.0.1', False, 'Invalid PIN')
            log_rate_limit_blocked('192.168.1.1', '/api/recommend', 60)

            output = stream.getvalue().strip().split('\n')
            self.assertGreaterEqual(len(output), 3)

            # Check 1st event
            event1 = json.loads(output[0])
            self.assertEqual(event1['event_type'], 'LOGIN_LOCKOUT')
            self.assertEqual(event1['pin'], '[REDACTED]')
            self.assertTrue(event1['security_audit'])

            # Check 2nd event
            event2 = json.loads(output[1])
            self.assertEqual(event2['event_type'], 'AUTH_ATTEMPT')
            self.assertEqual(event2['status'], 'FAILED')

            # Check 3rd event
            event3 = json.loads(output[2])
            self.assertEqual(event3['event_type'], 'RATE_LIMIT_EXCEEDED')
            self.assertEqual(event3['retry_after_seconds'], 60)
        finally:
            security_logger.removeHandler(handler)

if __name__ == '__main__':
    unittest.main()
