import os
import sys
import unittest
import json
import urllib.request
import threading
import time

# Ensure workspace root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import ThreadedHTTPServer, CineAIRequestHandler
from backend.config import LETTERBOXD_USERNAME

TEST_PORT = 9988
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

class TestCineAIAPI(unittest.TestCase):
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

    def test_01_api_status(self):
        url = f"{BASE_URL}/api/status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data.get('username'), LETTERBOXD_USERNAME or 'guest')
            self.assertIn('total_films', data)
            self.assertIn('model_status', data)

    def test_02_api_diary_filters(self):
        url = f"{BASE_URL}/api/diary?rating=All&sort=Newest+Log+First"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertIn('films', data)
            self.assertIn('total', data)

    def test_03_api_taste_radar(self):
        url = f"{BASE_URL}/api/taste_radar"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertIn('radar', data)
            self.assertIn('badges', data)

    def test_04_api_recommendation(self):
        url = f"{BASE_URL}/api/recommend"
        payload = json.dumps({'prompt': 'mind-bending sci-fi', 'context': 'Alone', 'streaming': 'All Platforms'}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertIn('candidates', data)
            self.assertIn('analysis', data)

    def test_05_static_file_serving(self):
        url = f"{BASE_URL}/index.html"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode('utf-8')
            self.assertIn('MBMR', content)
            self.assertIn('Mood-Based Movie Recommender', content)

    def test_06_direct_movie_title_search(self):
        url = f"{BASE_URL}/api/recommend"
        payload = json.dumps({'prompt': 'Inception', 'context': 'Alone', 'streaming': 'All Platforms'}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertIn('candidates', data)
            titles = [c.get('title', '').lower() for c in data['candidates']]
            self.assertTrue(any('inception' in t for t in titles))

if __name__ == '__main__':
    unittest.main()
