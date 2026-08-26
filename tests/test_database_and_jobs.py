import unittest
import time
import requests
import pandas as pd
from backend.db import (
    init_db, get_or_create_user, verify_user_pin,
    upsert_movies_batch, get_existing_movie_ids,
    upsert_user_diary, get_user_diary,
    upsert_user_watchlist, get_user_watchlist,
    remove_from_user_watchlist, add_to_user_watchlist
)
from backend.in_memory_model import train_user_model_in_memory, get_or_train_user_model
from backend.jobs import start_onboarding_job, get_job_status
from backend.recommender import analyze, titleNormalize

import threading
from backend.api import ThreadedHTTPServer, MBMRRequestHandler

# Own port and own server: the suite must not depend on a dev server already running.
API_PORT = 9899
API_URL = "http://localhost:{0}".format(API_PORT)


class TestNeonDatabaseAndJobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.server = ThreadedHTTPServer(('127.0.0.1', API_PORT), MBMRRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        try:
            threading.Thread(target=cls.server.shutdown, daemon=True).start()
            cls.server.server_close()
        except Exception:
            pass

    def test_01_user_creation_and_pin_auth(self):
        test_user = "test_cinephile_99"
        user = get_or_create_user(test_user, pin="4321", tmdb_key="test_tmdb", gemini_key="test_gemini")
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], test_user)

        # Correct PIN
        ok, msg, u = verify_user_pin(test_user, "4321")
        self.assertTrue(ok)
        self.assertEqual(u['tmdb_key'], "test_tmdb")

        # Wrong PIN
        ok_wrong, msg_wrong, _ = verify_user_pin(test_user, "0000")
        self.assertFalse(ok_wrong)
        self.assertIn("Invalid", msg_wrong)

    def test_02_shared_movies_deduplication_and_nan_sanitization(self):
        # Insert Swing Girls & Interstellar
        movies = [
            {
                'movie_id': 36592,
                'title': 'Swing Girls',
                'year': '2004',
                'genres': 'Comedy, Music',
                'overview': 'Delinquent high school girls form a big band.',
                'director': 'Shinobu Yaguchi',
                'runtime': 105,
                'vote_average': 7.9,
                'poster_path': '/5pAfMXqcaUjr0dBuOhLpTRbtZqa.jpg'
            },
            {
                'movie_id': 157336,
                'title': 'Interstellar',
                'year': '2014',
                'genres': 'Adventure, Drama, Science Fiction',
                'overview': 'A team of explorers travel through a wormhole in space.',
                'director': 'Christopher Nolan',
                'runtime': 169,
                'vote_average': 8.4,
                'poster_path': '/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg'
            }
        ]
        upsert_movies_batch(movies)
        existing = get_existing_movie_ids([36592, 157336, 99999999])
        self.assertIn(36592, existing)
        self.assertIn(157336, existing)
        self.assertNotIn(99999999, existing)

    def test_03_multi_user_isolation(self):
        user_a = "user_alpha_neontest"
        user_b = "user_beta_neontest"

        u_a = get_or_create_user(user_a)
        u_b = get_or_create_user(user_b)

        # Upsert movies into shared table first
        upsert_movies_batch([
            {'movie_id': 36592, 'title': 'Swing Girls', 'year': '2004', 'genres': 'Comedy, Music'},
            {'movie_id': 157336, 'title': 'Interstellar', 'year': '2014', 'genres': 'Sci-Fi'}
        ])

        # User A adds Swing Girls (36592) to watchlist
        upsert_user_watchlist(u_a['id'], [{'movie_id': 36592}])
        
        # User B logs Interstellar (157336) to diary
        upsert_user_diary(u_b['id'], [{'movie_id': 157336, 'rating': 5.0, 'watched_date': '2026-08-25'}])

        wl_a = get_user_watchlist(user_a)
        wl_b = get_user_watchlist(user_b)

        self.assertEqual(len(wl_a), 1)
        self.assertEqual(wl_a[0]['movie_id'], 36592)
        self.assertEqual(wl_a[0]['title'], 'Swing Girls')
        self.assertNotEqual(wl_a[0]['title'].lower(), 'nan')

        self.assertEqual(len(wl_b), 0)

        diary_b, total_b, avg_b = get_user_diary(user_b)
        self.assertEqual(total_b, 1)
        self.assertEqual(diary_b[0]['title'], 'Interstellar')

    def test_04_in_memory_model_training_speed(self):
        # Create user with mock diary films for in-memory training
        train_user = "train_speed_user"
        u = get_or_create_user(train_user)

        movies_mock = []
        diary_mock = []
        for i in range(1, 15):
            m_id = 900000 + i
            movies_mock.append({
                'movie_id': m_id,
                'title': f'Test Film {i}',
                'year': '2020',
                'genres': 'Sci-Fi, Action' if i % 2 == 0 else 'Comedy, Romance',
                'overview': f'Overview for film {i}',
                'director': 'Director X' if i % 3 == 0 else 'Director Y',
                'runtime': 110,
                'vote_average': 7.5
            })
            diary_mock.append({
                'movie_id': m_id,
                'rating': 4.5 if i % 2 == 0 else 2.5,
                'watched_date': '2026-08-20'
            })
        
        upsert_movies_batch(movies_mock)
        upsert_user_diary(u['id'], diary_mock)

        # Measure in-memory training speed (DB fetch + fit)
        t0 = time.time()
        model, cols, vec, enc = train_user_model_in_memory(train_user)
        elapsed = time.time() - t0

        self.assertIsNotNone(model)
        self.assertGreater(len(cols), 0)
        self.assertLess(elapsed, 10.0, f"Training took {elapsed:.3f}s")

    def test_05_http_api_search_and_recommendation_robustness(self):
        # Test direct search for "swing girl" via API
        resp = requests.post(f"{API_URL}/api/recommend", json={
            'prompt': 'swing girl',
            'context': 'Alone',
            'streaming': 'All Platforms'
        }, timeout=35)
        
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('candidates', data)
        self.assertGreater(len(data['candidates']), 0)
        
        # Check that top candidate title is Swing Girls and fields are not nan
        top_cand = data['candidates'][0]
        self.assertNotIn('nan', str(top_cand.get('title', '')).lower())
        self.assertNotIn('nan', str(top_cand.get('year', '')).lower())

    def test_06_http_api_auth_login(self):
        # Test Login API
        resp = requests.post(f"{API_URL}/api/auth/login", json={
            'username': 'test_cinephile_99',
            'pin': '4321'
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['user']['username'], 'test_cinephile_99')

    def test_07_async_onboarding_job_status(self):
        # Start onboarding job for guest
        resp = requests.post(f"{API_URL}/api/onboarding/start", json={
            'username': 'test_job_runner',
            'pin': '9999'
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        job_id = data['job_id']

        # Check status
        st_resp = requests.get(f"{API_URL}/api/onboarding/status?job_id={job_id}")
        self.assertEqual(st_resp.status_code, 200)
        st_data = st_resp.json()
        self.assertIn('status', st_data)
        self.assertIn('progress', st_data)

    def test_08_import_csv_full_history(self):
        # Test importing a 5-film Letterboxd export CSV
        unique_csv_user = f"csv_user_{int(time.time())}"
        csv_sample = """Date,Name,Year,Letterboxd URI,Rating
2026-08-01,Inception,2010,https://boxd.it/1sz2,5.0
2026-08-02,Blade Runner 2049,2017,https://boxd.it/c6c2,4.5
2026-08-03,Parasite,2019,https://boxd.it/h5fa,5.0
2026-08-04,Whiplash,2014,https://boxd.it/7iT8,4.5
2026-08-05,La La Land,2016,https://boxd.it/bWzA,4.0
"""
        resp = requests.post(f"{API_URL}/api/import_csv", json={
            'username': unique_csv_user,
            'csv_content': csv_sample,
            'is_watchlist': False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])

        # The import runs as a background job so a full export cannot time out the
        # request, so the response carries a job_id rather than the finished result.
        job_id = data.get('job_id')
        self.assertIsNotNone(job_id, "import_csv should return a job_id")

        deadline = time.time() + 120
        status = {}
        while time.time() < deadline:
            status = requests.get(
                f"{API_URL}/api/onboarding/status", params={'job_id': job_id}, timeout=10
            ).json()
            if status.get('status') in ('completed', 'failed'):
                break
            time.sleep(1)

        self.assertEqual(status.get('status'), 'completed',
                         f"Import job did not complete: {status}")

        # Verify diary has exactly 5 films in Neon DB
        diary, total, avg_r = get_user_diary(unique_csv_user)
        self.assertEqual(total, 5)
        self.assertGreater(avg_r, 4.0)

    def test_06_watchlist_mood_recommendation(self):
        """Test that /api/recommend with source='watchlist' correctly filters and returns watchlist films."""
        from backend.db import add_to_user_watchlist, get_or_create_user
        from unittest.mock import patch, Mock

        test_username = "rec_test_user_99"
        u = get_or_create_user(test_username)
        
        # Add a specific sci-fi movie and a horror movie to watchlist
        m_scifi = {
            'movie_id': 27205, # Inception
            'title': 'Inception',
            'genres': 'Science Fiction, Action, Thriller',
            'overview': 'Cobb steals information from deep within the subconscious.',
            'director': 'Christopher Nolan',
            'runtime': 148,
            'vote_average': 8.3,
            'poster_path': '/qmDpN6Ud1A1BUp9zJmUkzICJuUk.jpg',
            'backdrop_path': '/s3Tld83RKY5zZNV68HCk76zLEBh.jpg'
        }
        m_horror = {
            'movie_id': 138843, # The Conjuring
            'title': 'The Conjuring',
            'genres': 'Horror, Thriller',
            'overview': 'Paranormal investigators work to help a family.',
            'director': 'James Wan',
            'runtime': 112,
            'vote_average': 7.5,
            'poster_path': '/ff906214.jpg',
            'backdrop_path': '/bb906214.jpg'
        }
        
        add_to_user_watchlist(test_username, m_scifi)
        add_to_user_watchlist(test_username, m_horror)

        # Patch interpret_query_with_ai directly to bypass live API calls
        with patch('backend.api.interpret_query_with_ai') as mock_interpret:
            mock_interpret.return_value = {
                'genres': ['Horror'],
                'search_query': 'paranormal',
                'suggested_titles': ['The Conjuring']
            }

            # Post recommendation request with source='watchlist'
            resp = requests.post(f"{API_URL}/api/recommend", json={
                'username': test_username,
                'prompt': 'something scary and paranormal',
                'context': 'Alone',
                'streaming': 'All Platforms',
                'source': 'watchlist'
            })
            
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn('candidates', data)
            cands = data['candidates']
            
            # Should have returned The Conjuring as the top pick (Horror match)
            self.assertTrue(len(cands) > 0)
            self.assertEqual(cands[0]['title'], 'The Conjuring')

    def test_09_letterboxd_paginated_scraping(self):
        """Test that scrape_letterboxd_diary correctly fetches and compiles paginated diary rows."""
        from backend.jobs import scrape_letterboxd_diary
        from unittest.mock import Mock

        mock_session = Mock()
        
        def mock_get(url, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            
            if 'page/1/' in url:
                mock_resp.text = "".join([
                    f'<tr class="diary-entry-row">data-item-slug="movie-{i}" data-item-name="Movie {i} (2020)" rated-8 /for/2020/01/01/</tr>'
                    for i in range(50)
                ])
            elif 'page/2/' in url:
                mock_resp.text = "".join([
                    f'<tr class="diary-entry-row">data-item-slug="movie-{i}" data-item-name="Movie {i} (2020)" rated-8 /for/2020/01/01/</tr>'
                    for i in range(50, 70)
                ])
            else:
                mock_resp.text = ""
            return mock_resp
            
        mock_session.get.side_effect = mock_get

        entries = scrape_letterboxd_diary('test_user', session=mock_session, max_pages=5)
        
        self.assertEqual(len(entries), 70)
        self.assertEqual(entries[0]['slug'], 'movie-0')
        self.assertEqual(entries[69]['slug'], 'movie-69')

    def test_10_letterboxd_watchlist_paginated_scraping(self):
        """Test that scrape_letterboxd_watchlist correctly fetches and compiles paginated watchlist posters."""
        from backend.jobs import scrape_letterboxd_watchlist
        from unittest.mock import Mock

        mock_session = Mock()
        
        def mock_get(url, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            
            if 'page/1/' in url:
                mock_resp.text = "".join([
                    f'<div class="poster">data-item-slug="movie-{i}" data-item-name="Movie {i} (2020)"</div>'
                    for i in range(50)
                ]) + 'href="/watchlist/page/2/"'
            elif 'page/2/' in url:
                mock_resp.text = "".join([
                    f'<div class="poster">data-item-slug="movie-{i}" data-item-name="Movie {i} (2020)"</div>'
                    for i in range(50, 70)
                ])
            else:
                mock_resp.text = ""
            return mock_resp
            
        mock_session.get.side_effect = mock_get

        entries = scrape_letterboxd_watchlist('test_user', session=mock_session, max_pages=5)
        
        self.assertEqual(len(entries), 70)
        self.assertEqual(entries[0]['slug'], 'movie-0')
        self.assertEqual(entries[69]['slug'], 'movie-69')

if __name__ == '__main__':
    unittest.main()
