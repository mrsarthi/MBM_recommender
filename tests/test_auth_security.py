import os
import sys
import unittest
import json
import urllib.request
import urllib.error
import threading
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import ThreadedHTTPServer, MBMRRequestHandler, create_session_token, verify_session_token
from backend.db import get_or_create_user, verify_user_pin

TEST_PORT = 9981
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

class TestAuthSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedHTTPServer(('127.0.0.1', TEST_PORT), MBMRRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        time.sleep(0.5)
        
        # Create test user with PIN
        cls.test_user = "sec_test_user_77"
        cls.test_pin = "8899"
        get_or_create_user(cls.test_user, pin=cls.test_pin)

    @classmethod
    def tearDownClass(cls):
        try:
            threading.Thread(target=cls.server.shutdown, daemon=True).start()
            cls.server.server_close()
        except Exception:
            pass

    def test_01_token_generation_and_validation(self):
        token = create_session_token("sarthi_watcher")
        self.assertTrue(bool(token))
        valid, user = verify_session_token(token)
        self.assertTrue(valid)
        self.assertEqual(user, "sarthi_watcher")

        # Forged token signature rejection
        bad_token = token[:-4] + "ffff"
        bad_valid, _ = verify_session_token(bad_token)
        self.assertFalse(bad_valid)

    def test_02_unauthenticated_request_rejected(self):
        url = f"{BASE_URL}/api/diary"
        # Calling with spoofed X-Letterboxd-User but no session token
        req = urllib.request.Request(url, headers={'X-Letterboxd-User': self.test_user})
        try:
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 401)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_03_authenticated_request_with_bearer_token(self):
        token = create_session_token(self.test_user)
        url = f"{BASE_URL}/api/diary"
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertIn('films', data)
            self.assertIn('total', data)

    def test_04_login_issues_valid_session_token(self):
        url = f"{BASE_URL}/api/auth/login"
        payload = json.dumps({'username': self.test_user, 'pin': self.test_pin}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertTrue(data.get('success'))
            self.assertIn('session_token', data)
            valid, u = verify_session_token(data['session_token'])
            self.assertTrue(valid)
            self.assertEqual(u, self.test_user)

    def test_05_account_takeover_via_onboarding_prevented(self):
        url = f"{BASE_URL}/api/onboarding/start"
        # Trying to re-onboard existing user with a new PIN
        payload = json.dumps({
            'username': self.test_user,
            'pin': '9999',
            'skip_scrape': True
        }).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as resp:
                self.fail("Expected 409 Conflict")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)

    def test_06_pin_verification_and_lockout(self):
        temp_user = "lockout_user_99"
        get_or_create_user(temp_user, pin="1234")

        # 4 failed attempts
        for _ in range(4):
            ok, msg, _ = verify_user_pin(temp_user, "0000")
            self.assertFalse(ok)

        # 5th failed attempt -> locks account
        ok, msg, _ = verify_user_pin(temp_user, "0000")
        self.assertFalse(ok)
        self.assertIn("locked", msg.lower())

        # 6th attempt with correct PIN is still rejected due to active lockout
        ok, msg, _ = verify_user_pin(temp_user, "1234")
        self.assertFalse(ok)
        self.assertIn("locked", msg.lower())

if __name__ == '__main__':
    unittest.main()
