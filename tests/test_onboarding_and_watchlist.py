import os
import sys
import unittest
import json
import urllib.request
import threading
import time
import shutil

# Ensure workspace root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import ThreadedHTTPServer, CineAIRequestHandler
from backend.config import get_user_profile_path, get_user_watchlist_path
from backend.watchlist import add_to_watchlist, remove_from_watchlist, load_watchlist

TEST_PORT = 9987
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

class TestOnboardingAndWatchlistIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedHTTPServer(('127.0.0.1', TEST_PORT), CineAIRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        try:
            threading.Thread(target=cls.server.shutdown, daemon=True).start()
            cls.server.server_close()
        except Exception:
            pass
        # Clean up test profiles
        for u in ['test_user_alpha', 'test_user_beta']:
            p_prof = get_user_profile_path(u)
            p_wl = get_user_watchlist_path(u)
            if os.path.exists(p_prof):
                try: os.remove(p_prof)
                except Exception: pass
            if os.path.exists(p_wl):
                try: os.remove(p_wl)
                except Exception: pass

    def test_01_new_user_starts_with_zero_watchlist(self):
        """New visitor/user should have 0 watchlist count and 0 films."""
        url = f"{BASE_URL}/api/status"
        req = urllib.request.Request(url, headers={'X-Letterboxd-User': 'test_user_alpha'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data.get('username'), 'test_user_alpha')
            self.assertEqual(data.get('watchlist_count'), 0)
            self.assertEqual(data.get('total_films'), 0)

    def test_02_new_user_empty_watchlist_endpoint(self):
        """GET /api/watchlist for an unsynced user returns an empty list, not default 69 items."""
        url = f"{BASE_URL}/api/watchlist"
        req = urllib.request.Request(url, headers={'X-Letterboxd-User': 'test_user_alpha'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data.get('total'), 0)
            self.assertEqual(data.get('watchlist'), [])

    def test_03_user_specific_watchlist_add_and_remove(self):
        """Adding a movie to user_alpha's watchlist does not affect user_beta."""
        movie = {
            'id': 999999,
            'title': 'Test Alpha Movie',
            'genres': ['Science Fiction', 'Thriller'],
            'overview': 'A test mind-bending film',
            'release_date': '2026-01-01',
            'runtime': 110,
            'vote_average': 8.5
        }

        # Add to Alpha
        url = f"{BASE_URL}/api/watchlist/add"
        payload = json.dumps(movie).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={
            'Content-Type': 'application/json',
            'X-Letterboxd-User': 'test_user_alpha'
        })
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            d = json.loads(resp.read().decode('utf-8'))
            self.assertTrue(d.get('success'))

        # Check Alpha's status
        req_alpha = urllib.request.Request(f"{BASE_URL}/api/status", headers={'X-Letterboxd-User': 'test_user_alpha'})
        with urllib.request.urlopen(req_alpha) as resp:
            d = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(d.get('watchlist_count'), 1)

        # Check Beta's status (should remain 0)
        req_beta = urllib.request.Request(f"{BASE_URL}/api/status", headers={'X-Letterboxd-User': 'test_user_beta'})
        with urllib.request.urlopen(req_beta) as resp:
            d = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(d.get('watchlist_count'), 0)

        # Remove from Alpha
        rem_payload = json.dumps({'movie_id': 999999}).encode('utf-8')
        req_rem = urllib.request.Request(f"{BASE_URL}/api/watchlist/remove", data=rem_payload, headers={
            'Content-Type': 'application/json',
            'X-Letterboxd-User': 'test_user_alpha'
        })
        with urllib.request.urlopen(req_rem) as resp:
            self.assertEqual(resp.status, 200)

        # Alpha is 0 again
        with urllib.request.urlopen(req_alpha) as resp:
            d = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(d.get('watchlist_count'), 0)

    def test_04_sync_endpoint_without_username_fails_gracefully(self):
        """Triggering watchlist sync without a username returns a friendly validation error."""
        url = f"{BASE_URL}/api/watchlist/sync"
        payload = json.dumps({'username': ''}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            d = json.loads(resp.read().decode('utf-8'))
            self.assertFalse(d.get('success'))

if __name__ == '__main__':
    unittest.main()
