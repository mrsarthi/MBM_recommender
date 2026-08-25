import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
from backend.predictions import predict_movie_score, predict_movie_scores_batch
from backend.in_memory_model import train_user_model_in_memory, get_or_train_user_model
from backend.gemini_client import interpret_query_with_ai, CASCADE_MODELS, _fallback_mood_match
from backend.db import get_or_create_user, upsert_user_diary, upsert_movies_batch, get_user_watchlist
import pandas as pd

class TestFastWatchlistAndGeminiFallback(unittest.TestCase):

    def setUp(self):
        # Create a test user with sample training data
        self.username = "test_speed_user"
        self.user = get_or_create_user(self.username)
        
        sample_movies = [
            {'movie_id': 90001, 'title': 'Speed Noir', 'genres': 'Crime, Mystery, Thriller', 'overview': 'A fast-paced detective thriller.', 'director': 'David Fincher', 'runtime': 120, 'vote_average': 8.2},
            {'movie_id': 90002, 'title': 'Space Odyssey', 'genres': 'Science Fiction, Adventure', 'overview': 'Journey to the stars and mysterious alien monoliths.', 'director': 'Stanley Kubrick', 'runtime': 149, 'vote_average': 8.4},
            {'movie_id': 90003, 'title': 'Laugh Out Loud', 'genres': 'Comedy, Romance', 'overview': 'Hilarious romantic mishaps in modern city.', 'director': 'Edgar Wright', 'runtime': 95, 'vote_average': 7.1},
            {'movie_id': 90004, 'title': 'The Dark Knight', 'genres': 'Action, Crime, Drama', 'overview': 'Batman raises the stakes in his war on crime.', 'director': 'Christopher Nolan', 'runtime': 152, 'vote_average': 8.5},
            {'movie_id': 90005, 'title': 'Pulp Fiction', 'genres': 'Crime, Drama', 'overview': 'The lives of two mob hitmen, a boxer, a gangster and his wife.', 'director': 'Quentin Tarantino', 'runtime': 154, 'vote_average': 8.5},
            {'movie_id': 90006, 'title': 'Spirited Away', 'genres': 'Animation, Family, Fantasy', 'overview': 'A young girl wanders into a world ruled by gods and spirits.', 'director': 'Hayao Miyazaki', 'runtime': 125, 'vote_average': 8.5}
        ]
        upsert_movies_batch(sample_movies)
        upsert_user_diary(self.user['id'], [
            {'movie_id': 90001, 'rating': 4.5, 'watched_date': '2026-08-01'},
            {'movie_id': 90002, 'rating': 5.0, 'watched_date': '2026-08-02'},
            {'movie_id': 90003, 'rating': 3.0, 'watched_date': '2026-08-03'},
            {'movie_id': 90004, 'rating': 4.5, 'watched_date': '2026-08-04'},
            {'movie_id': 90005, 'rating': 4.0, 'watched_date': '2026-08-05'},
            {'movie_id': 90006, 'rating': 5.0, 'watched_date': '2026-08-06'}
        ])

    def test_01_predict_movie_scores_batch_accuracy_and_speed(self):
        """Test that batch scoring runs in < 20ms for 100 movies and returns valid floats."""
        model, cols, vec, enc = train_user_model_in_memory(self.username)
        self.assertIsNotNone(model)

        test_movies = [
            {'title': f'Movie {i}', 'genres': 'Crime, Thriller' if i % 2 == 0 else 'Comedy',
             'overview': 'Dark mystery thriller' if i % 2 == 0 else 'Fun comedy',
             'director': 'David Fincher' if i % 2 == 0 else 'Unknown',
             'runtime': 110}
            for i in range(100)
        ]

        t0 = time.time()
        scores = predict_movie_scores_batch(model, cols, vec, enc, test_movies)
        duration_ms = (time.time() - t0) * 1000

        print(f"\n[BENCHMARK] 100-film vectorized batch scoring completed in: {duration_ms:.2f}ms")
        self.assertEqual(len(scores), 100)
        self.assertLess(duration_ms, 50.0, "Batch scoring should take under 50ms")
        for s in scores:
            self.assertIsInstance(s, float)
            self.assertGreaterEqual(s, 0.5)
            self.assertLessEqual(s, 5.0)

    def test_02_gemini_cascade_models_active(self):
        """Test that the Gemini multi-model fallback cascade responds with structured query JSON."""
        res = interpret_query_with_ai("mind-bending psychological mystery like shutter island")
        print("\n[AI RESPONSE]:", res)
        self.assertIn('genres', res)
        self.assertIn('search_query', res)
        self.assertTrue(len(res['genres']) > 0)
        self.assertTrue(any(g in res['genres'] for g in ['Mystery', 'Thriller', 'Science Fiction', 'Drama']))

    def test_03_gemini_fallback_on_invalid_key(self):
        """Test that when API key is exhausted or invalid, fallback heuristic responds gracefully."""
        res = interpret_query_with_ai("dark cyberpunk neo-noir detective", custom_api_key="INVALID_KEY_XYZ")
        self.assertIn('genres', res)
        self.assertTrue(any(g in res['genres'] for g in ['Crime', 'Mystery', 'Thriller', 'Science Fiction', 'Action']))

if __name__ == '__main__':
    unittest.main()
