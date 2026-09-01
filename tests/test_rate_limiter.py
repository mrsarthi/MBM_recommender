import unittest
import time
import json
import threading
from io import BytesIO
from unittest.mock import Mock, patch

from backend.rate_limiter import TokenBucketRateLimiter, rate_limiter
from backend.api import MBMRRequestHandler

class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.limiter = TokenBucketRateLimiter()
        # Fast testing rules:
        self.limiter.rules = {
            '/api/test_burst': (3.0, 1.0),      # 3 burst capacity, 1 token/sec refill
            '/api/recommend': (2.0, 0.5),       # 2 burst capacity
            'default': (5.0, 2.0)
        }

    def test_01_token_bucket_burst_and_exhaustion(self):
        """Test that burst limit is respected and subsequent requests are rejected."""
        # 1st request allowed
        allowed, retry_after = self.limiter.is_allowed('client_1', '/api/test_burst')
        self.assertTrue(allowed)
        self.assertEqual(retry_after, 0)

        # 2nd request allowed
        allowed, retry_after = self.limiter.is_allowed('client_1', '/api/test_burst')
        self.assertTrue(allowed)

        # 3rd request allowed (exhausts capacity of 3)
        allowed, retry_after = self.limiter.is_allowed('client_1', '/api/test_burst')
        self.assertTrue(allowed)

        # 4th request rejected
        allowed, retry_after = self.limiter.is_allowed('client_1', '/api/test_burst')
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)

    def test_02_bucket_isolation_between_clients_and_endpoints(self):
        """Test that different clients and different endpoints have isolated token buckets."""
        # Exhaust client_1 on /api/test_burst
        for _ in range(3):
            self.limiter.is_allowed('client_1', '/api/test_burst')
        self.assertFalse(self.limiter.is_allowed('client_1', '/api/test_burst')[0])

        # client_2 on /api/test_burst should still be allowed
        self.assertTrue(self.limiter.is_allowed('client_2', '/api/test_burst')[0])

        # client_1 on another endpoint should still be allowed
        self.assertTrue(self.limiter.is_allowed('client_1', '/api/recommend')[0])

    def test_03_token_refill_over_time(self):
        """Test that tokens replenish according to the refill rate."""
        # Consume all 3 tokens
        for _ in range(3):
            self.limiter.is_allowed('client_3', '/api/test_burst')
        self.assertFalse(self.limiter.is_allowed('client_3', '/api/test_burst')[0])

        # Wait 1.1 seconds for 1 token to refill
        time.sleep(1.1)
        allowed, _ = self.limiter.is_allowed('client_3', '/api/test_burst')
        self.assertTrue(allowed)

    def test_04_thread_safety_under_concurrent_load(self):
        """Test that rate limiter operates cleanly across multiple concurrent threads."""
        limiter = TokenBucketRateLimiter()
        limiter.rules['default'] = (50.0, 10.0)

        results = []
        def worker():
            for _ in range(10):
                allowed, _ = limiter.is_allowed('concurrent_client', '/api/generic')
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Total 50 requests: exactly 50 should be allowed (capacity = 50)
        self.assertEqual(len(results), 50)
        self.assertTrue(all(results))

    def test_05_http_handler_returns_429(self):
        """Test that MBMRRequestHandler sends HTTP 429 when rate limit is exceeded."""
        rate_limiter.reset()
        # Set a tiny rule for /api/recommend
        rate_limiter.rules['/api/recommend'] = (1.0, 0.1)

        handler = Mock(spec=MBMRRequestHandler)
        handler.headers = {'X-Forwarded-For': '192.168.1.100'}
        handler.client_address = ('192.168.1.100', 12345)
        handler.path = '/api/recommend'
        handler._get_allowed_origin = Mock(return_value='*')
        
        sent_headers = {}
        def mock_send_header(k, v):
            sent_headers[k] = v
        handler.send_header = mock_send_header
        handler.send_response = Mock()
        handler.end_headers = Mock()
        handler.wfile = BytesIO()

        # First call -> passes
        allowed1 = MBMRRequestHandler._enforce_rate_limit(handler, '/api/recommend')
        self.assertTrue(allowed1)

        # Second call -> blocked with 429
        allowed2 = MBMRRequestHandler._enforce_rate_limit(handler, '/api/recommend')
        self.assertFalse(allowed2)
        handler.send_response.assert_called_with(429)
        self.assertIn('Retry-After', sent_headers)
        
        # Verify JSON body
        body = json.loads(handler.wfile.getvalue().decode('utf-8'))
        self.assertIn('error', body)
        self.assertIn('retry_after', body)

if __name__ == '__main__':
    unittest.main()
