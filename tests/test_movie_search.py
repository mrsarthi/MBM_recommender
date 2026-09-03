import os
import sys
import unittest
import json
import urllib.request
import threading
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import ThreadedHTTPServer, CineAIRequestHandler, create_session_token
from backend.recommender import analyze, load_watched_data
from backend.query_parser import interpret_query_with_ai
from backend.predictions import load_ai
from backend.config import MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH

TEST_PORT = 9983
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

class TestMovieTitleSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedHTTPServer(('127.0.0.1', TEST_PORT), CineAIRequestHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

        cls.watched_titles, cls.watched_ids, cls.hated = load_watched_data()
        cls.ai_model, cls.ai_cols, cls.ai_vec, cls.ai_enc = load_ai(
            MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH
        )

        # Seed test user and Inception in database
        from backend.db import get_or_create_user, upsert_user_diary, upsert_movies_batch
        cls.test_username = 'test_search_user'
        user = get_or_create_user(cls.test_username)
        upsert_movies_batch([{
            'movie_id': 27205,
            'title': 'Inception',
            'year': '2010',
            'genres': 'Action, Science Fiction, Adventure',
            'overview': 'Cobb is a skilled thief...'
        }])
        upsert_user_diary(user['id'], [{
            'movie_id': 27205,
            'rating': 4.5,
            'watched_date': '2026-08-26'
        }])

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _query_api(self, prompt):
        url = f"{BASE_URL}/api/recommend"
        payload = json.dumps({
            'prompt': prompt,
            'context': 'Alone',
            'streaming': 'All Platforms'
        }).encode('utf-8')
        token = create_session_token(self.test_username)
        req = urllib.request.Request(url, data=payload, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        })
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode('utf-8'))

    def test_01_search_inception_title(self):
        fresh_user = f"unwatched_user_{int(time.time()*1000)}"
        url = f"{BASE_URL}/api/recommend"
        payload = json.dumps({
            'prompt': 'Inception',
            'context': 'Alone',
            'streaming': 'All Platforms'
        }).encode('utf-8')
        token = create_session_token(fresh_user)
        req = urllib.request.Request(url, data=payload, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        })
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for 'Inception'")
        first_title = candidates[0].get('title', '')
        self.assertEqual(first_title, 'Inception', f"First candidate should be Inception, got '{first_title}'")
        self.assertTrue(candidates[0].get('is_direct_match'), "Inception should be marked as direct match")

    def test_02_search_dune_title(self):
        data = self._query_api('Dune')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for 'Dune'")
        
        # Check top candidates contain Dune
        titles = [c.get('title', '').lower() for c in candidates[:5]]
        self.assertTrue(any('dune' in t for t in titles), f"Top candidates should contain 'Dune', got {titles}")
        
        # Check direct matches have is_direct_match=True
        direct_matches = [c for c in candidates if c.get('is_direct_match')]
        self.assertGreater(len(direct_matches), 0, "Should have direct matches for Dune")

    def test_03_search_interstellar(self):
        data = self._query_api('Interstellar')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for 'Interstellar'")
        
        titles = [c.get('title', '') for c in candidates]
        self.assertIn('Interstellar', titles, "Interstellar should be in the search results")

    def test_04_search_fight_club(self):
        data = self._query_api('Fight Club')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for 'Fight Club'")
        first = candidates[0]
        self.assertEqual(first.get('title'), 'Fight Club')
        self.assertTrue(first.get('is_direct_match'))

    def test_05_search_partial_title_spider_man(self):
        data = self._query_api('Spider-Man')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for 'Spider-Man'")
        titles = [c.get('title', '').lower() for c in candidates]
        self.assertTrue(any('spider' in t for t in titles), "Should find Spider-Man films")

    def test_06_search_director_christopher_nolan(self):
        data = self._query_api('Christopher Nolan')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 0, "Should return candidates for Christopher Nolan")

    def test_07_movie_recommendations_ripple(self):
        # Searching a specific movie should also bring up recommendations
        data = self._query_api('Parasite')
        candidates = data.get('candidates', [])
        self.assertGreater(len(candidates), 1, "Should return Parasite and ripple recommendations")
        first = candidates[0]
        self.assertEqual(first.get('title'), 'Parasite')

    def test_08_direct_analyze_function(self):
        ai_analysis = {'genres': [], 'search_query': '', 'suggested_titles': []}
        picks = analyze(
            [], [], self.hated,
            ai_analysis, self.ai_model, self.ai_cols, self.ai_vec, self.ai_enc,
            raw_prompt='The Matrix'
        )
        self.assertGreater(len(picks), 0)
        first_title = picks[0].get('title', '')
        self.assertIn('Matrix', first_title)
        self.assertTrue(picks[0].get('is_direct_match'))

if __name__ == '__main__':
    unittest.main()
