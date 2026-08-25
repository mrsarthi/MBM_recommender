import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

# Ensure workspace root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

from backend.sync_letterboxd import sync_rss
from backend.recommender import titleNormalize, load_watched_data
from backend.feature_engineering import feature_engineering
from backend.model_train import train_personal_model
from backend.predictions import predict_movie_score, get_post_watch_recommendations, get_watch_providers, load_ai
from backend.gemini_client import _fallback_mood_match

class TestMBMRecommender(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_csv = os.path.join(self.test_dir, 'test_profile.csv')
        self.sample_features = os.path.join(self.test_dir, 'test_features.csv')
        self.sample_model = os.path.join(self.test_dir, 'test_model.pkl')
        self.sample_cols = os.path.join(self.test_dir, 'test_cols.pkl')
        self.sample_vec = os.path.join(self.test_dir, 'test_vec.pkl')
        self.sample_enc = os.path.join(self.test_dir, 'test_enc.pkl')
        self.sample_memory = os.path.join(self.test_dir, 'test_memory.csv')

        # Create dummy sample user dataset (25 movies to satisfy >= 15 train requirement)
        data = {
            'Date': [f'2025-01-{i+1:02d}' for i in range(25)],
            'Name': [f'Movie {i}' for i in range(25)],
            'Year': [2000 + i for i in range(25)],
            'Rating': [5.0 if i % 2 == 0 else 2.0 for i in range(25)], # Mix of loved and hated
            'movie_id': [1000 + i for i in range(25)],
            'genres': ['Action, Sci-Fi' if i % 2 == 0 else 'Drama, Romance' for i in range(25)],
            'director': ['Christopher Nolan' if i < 5 else 'Denis Villeneuve' if i < 10 else 'Other Director' for i in range(25)],
            'cast': ['Leonardo DiCaprio, Joseph Gordon-Levitt' if i % 2 == 0 else 'Timothée Chalamet, Zendaya' for i in range(25)],
            'keywords': ['dream, subconscious' if i % 2 == 0 else 'desert, spice' for i in range(25)],
            'overview': ['A thief who steals corporate secrets through the use of dream-sharing technology.' if i % 2 == 0 else 'A noble family becomes embroiled in a war for control over the galaxy\'s most valuable asset.' for i in range(25)]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_title_normalization(self):
        print("\n[Test 1] Testing titleNormalize...")
        self.assertEqual(titleNormalize("Spider-Man: No Way Home (2021)!"), "spidermannowayhome2021")
        self.assertEqual(titleNormalize("  Everything Everywhere All at Once  "), "everythingeverywhereallatonce")
        self.assertEqual(titleNormalize("OMG: Oh My God!"), "omgohmygod")
        print("  -> PASSED: Title normalization accurately strips non-alphanumeric chars and whitespace.")

    def test_02_watched_movies_and_veto_system(self):
        print("\n[Test 2] Testing watchedMovies & Veto parsing...")
        watched_titles, watched_ids, hated_movies = load_watched_data(self.sample_csv, self.sample_memory)
        self.assertGreater(len(watched_titles), 0)
        self.assertGreater(len(watched_ids), 0)
        self.assertGreater(len(hated_movies), 0)
        # Check that low rated movies (<= 2.5) are in hated list
        self.assertIn("movie1", hated_movies) # index 1 had 2.0 rating
        self.assertNotIn("movie0", hated_movies) # index 0 had 5.0 rating
        print(f"  -> PASSED: Loaded {len(watched_titles)} watched titles, {len(hated_movies)} vetoed movies.")

    def test_03_feature_engineering_pipeline(self):
        print("\n[Test 3] Testing Feature Engineering with Director, Cast & NLP keywords...")
        ok = feature_engineering(
            input_file=self.sample_csv,
            output_file=self.sample_features,
            vectorizer_path=self.sample_vec,
            encoders_path=self.sample_enc
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.sample_features))
        self.assertTrue(os.path.exists(self.sample_vec))

        feat_df = pd.read_csv(self.sample_features)
        self.assertIn('Rating', feat_df.columns)
        self.assertTrue(any(c.startswith('genre_') for c in feat_df.columns))
        print(f"  -> PASSED: Successfully engineered {feat_df.shape[1]} features across {feat_df.shape[0]} rows.")

    def test_04_model_training_and_recency_weights(self):
        print("\n[Test 4] Testing Model Training with Recency Weights...")
        feature_engineering(
            input_file=self.sample_csv,
            output_file=self.sample_features,
            vectorizer_path=self.sample_vec,
            encoders_path=self.sample_enc
        )
        ok = train_personal_model(
            input_file=self.sample_features,
            model_path=self.sample_model,
            columns_path=self.sample_cols
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.sample_model))
        self.assertTrue(os.path.exists(self.sample_cols))
        print("  -> PASSED: Random Forest regressor trained and saved with feature columns.")

    def test_05_prediction_scoring(self):
        print("\n[Test 5] Testing Prediction Scoring & AI Match...")
        feature_engineering(
            input_file=self.sample_csv,
            output_file=self.sample_features,
            vectorizer_path=self.sample_vec,
            encoders_path=self.sample_enc
        )
        train_personal_model(
            input_file=self.sample_features,
            model_path=self.sample_model,
            columns_path=self.sample_cols
        )
        model, cols, vec, enc = load_ai(
            model_path=self.sample_model,
            cols_path=self.sample_cols,
            vec_path=self.sample_vec,
            enc_path=self.sample_enc
        )

        high_score = predict_movie_score(
            model, cols, vec, enc,
            genres=['Action', 'Sci-Fi'],
            director='Christopher Nolan',
            overview='A thief who enters subconscious minds to steal secrets.'
        )
        low_score = predict_movie_score(
            model, cols, vec, enc,
            genres=['Drama', 'Romance'],
            director='Unknown',
            overview='A slow romantic drama set in a small village.'
        )

        self.assertGreaterEqual(high_score, 0.5)
        self.assertLessEqual(high_score, 5.0)
        self.assertGreaterEqual(low_score, 0.5)
        self.assertLessEqual(low_score, 5.0)
        self.assertGreater(high_score, low_score)
        print(f"  -> PASSED: Model predicted {high_score:.2f}★ for High-Affinity vs {low_score:.2f}★ for Low-Affinity.")

    def test_06_post_watch_ripple_recommendations(self):
        print("\n[Test 6] Testing Post-Watch Ripple Recommendations...")
        recs = get_post_watch_recommendations(27205, watched_titles=set(), watched_ids={27205}, top_n=3)
        self.assertIsInstance(recs, list)
        if recs:
            self.assertLessEqual(len(recs), 3)
            self.assertNotIn(27205, [r['id'] for r in recs])
            print(f"  -> PASSED: Returned {len(recs)} unlocked recommendations (e.g. '{recs[0]['title']}').")
        else:
            print("  -> SKIPPED (Network/API limited): Returned empty list safely.")

    def test_07_watch_providers(self):
        print("\n[Test 7] Testing TMDB Watch Providers query...")
        providers = get_watch_providers(27205, region='US')
        self.assertIsInstance(providers, list)
        print(f"  -> PASSED: Watch providers query executed safely (returned: {providers[:3] if providers else 'None'}).")

    def test_08_fallback_mood_matching(self):
        print("\n[Test 8] Testing Fallback Mood Matching...")
        happy_genres = _fallback_mood_match("I am feeling happy and want a comedy")
        self.assertIn("Comedy", happy_genres)

        tense_genres = _fallback_mood_match("Give me something tense and scary")
        self.assertTrue(any(g in tense_genres for g in ["Horror", "Thriller", "Mystery"]))
        print("  -> PASSED: Fallback mood parser accurately mapped mood keywords to genres.")

    def test_09_letterboxd_rss_parser_structure(self):
        print("\n[Test 9] Testing Letterboxd RSS Parser structure...")
        ok, msg = sync_rss("non_existent_dummy_user_123456789", self.sample_csv)
        self.assertFalse(ok)
        print("  -> PASSED: Letterboxd RSS parser correctly handles 404/invalid user gracefully.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
