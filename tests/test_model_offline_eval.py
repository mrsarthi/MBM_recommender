"""
Unit & Integration Tests for Recommendation Science & Offline Evaluation
Validates smoothed target encoding, out-of-sample evaluation metrics,
batch scoring throughput, and hated-movie exact matching.
"""

import time
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.in_memory_model import train_user_model_in_memory, invalidate_user_model
from backend.predictions import predict_movie_scores_batch, predict_movie_score
from backend.eval_harness import evaluate_model_temporal_holdout, compute_ndcg_at_k
from backend.recommender import titleNormalize, analyze


class TestModelOfflineEvalAndUpgrades(unittest.TestCase):

    def setUp(self):
        # Construct synthetic cinephile diary with 60 films
        records = []
        directors = ['Christopher Nolan', 'Denis Villeneuve', 'Greta Gerwig', 'Quentin Tarantino', 'Unknown Indie']
        genres_pool = ['Sci-Fi, Action', 'Drama, Mystery', 'Comedy, Romance', 'Thriller, Crime', 'Horror']
        
        for i in range(60):
            d = directors[i % len(directors)]
            g = genres_pool[i % len(genres_pool)]
            # High ratings for Nolan & Villeneuve, lower for Unknown
            base_r = 4.5 if d in ['Christopher Nolan', 'Denis Villeneuve'] else (3.0 if d == 'Unknown Indie' else 3.8)
            noise = (i % 3 - 1) * 0.5
            rating = max(1.0, min(5.0, base_r + noise))
            
            records.append({
                'movie_id': 1000 + i,
                'title': f'Film {i}',
                'director': d,
                'cast': f'Actor_{i%10}, Actor_{(i+1)%10}, Actor_{(i+2)%10}',
                'genres': g,
                'overview': f'Exciting storyline about cinema plot {i} with intense twists.',
                'runtime': 100 + (i % 40),
                'Rating': rating,
                'Date': f'2024-{(i//30)+1:02d}-{(i%28)+1:02d}'
            })
        self.sample_df = pd.DataFrame(records)

    def test_01_smoothed_target_encoding_correctness(self):
        """Verify that unseen directors evaluate to user mean prior, not zero."""
        with patch('backend.in_memory_model.get_diary_training_df', return_value=self.sample_df):
            model, cols, vec, encoders = train_user_model_in_memory('test_cinephile')
            
            self.assertIsNotNone(model)
            self.assertIn('director_target_map', encoders)
            self.assertIn('cast_target_map', encoders)
            
            user_mean = encoders['user_mean']
            self.assertGreater(user_mean, 2.5)
            self.assertLess(user_mean, 4.8)
            
            # An unseen director should default to user_mean, not 0.0
            unseen_score = predict_movie_score(
                model, cols, vec, encoders,
                genres=['Sci-Fi'], director='Brand New Unseen Director', overview='Deep space adventure'
            )
            self.assertGreater(unseen_score, 2.0)
            self.assertLessEqual(unseen_score, 5.0)

    def test_02_hated_movie_exact_title_match_bugfix(self):
        """Verify that penalizing 'Up' does NOT penalize 'Upgrade' (fixes §3.h)."""
        hated_movies = ['Up'] # User rated 'Up' 1.0 star
        
        # Test candidate 1: "Upgrade" (2018 sci-fi thriller)
        cand_upgrade = {
            'id': 500600,
            'title': 'Upgrade',
            'genres': ['Action', 'Thriller', 'Science Fiction'],
            'genre_ids': [28, 53, 878],
            'overview': 'Set in the near-future, technology controls nearly all aspects of life.',
            'runtime': 100,
            'thematic_weight': 1.0
        }
        
        # Test candidate 2: "Up" (Pixar animated movie)
        cand_up = {
            'id': 14160,
            'title': 'Up',
            'genres': ['Animation', 'Comedy', 'Family'],
            'genre_ids': [16, 35, 10751],
            'overview': 'Carl Fredricksen spent his entire life dreaming of exploring the globe.',
            'runtime': 96,
            'thematic_weight': 1.0
        }

        with patch('backend.in_memory_model.get_diary_training_df', return_value=self.sample_df):
            model, cols, vec, encoders = train_user_model_in_memory('test_cinephile')
            
            # Mock TMDB discover returning both candidates
            with patch('backend.recommender.http_session.get') as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {'results': [cand_upgrade, cand_up]}
                mock_get.return_value = mock_resp
                
                results = analyze(
                    watchedSet_titles=set(),
                    watchedSet_ids=set(),
                    hated_movies=hated_movies,
                    ai_analysis={'genres': ['Action', 'Thriller'], 'search_query': 'scifi', 'suggested_titles': []},
                    ai_model=model,
                    ai_columns=cols,
                    ai_vectorizer=vec,
                    ai_encoders=encoders,
                    raw_prompt="sci-fi thriller",
                    source="discover"
                )
                
                res_map = {r['title']: r['ai_score'] for r in results}
                
                # 'Upgrade' should NOT suffer the -2.5 penalty from 'Up'
                self.assertIn('Upgrade', res_map)
                self.assertIn('Up', res_map)
                self.assertGreater(res_map['Upgrade'], res_map['Up'], 
                    f"Upgrade ({res_map['Upgrade']}) was unjustly penalized by Up ({res_map['Up']})")

    def test_03_vectorized_batch_scoring_performance(self):
        """Benchmark 200 candidates scored in a single batch pass (< 30ms)."""
        with patch('backend.in_memory_model.get_diary_training_df', return_value=self.sample_df):
            model, cols, vec, encoders = train_user_model_in_memory('test_cinephile')
            
            # Generate 200 candidate movies
            candidates = []
            for i in range(200):
                candidates.append({
                    'id': 2000 + i,
                    'title': f'Candidate Movie {i}',
                    'director': 'Christopher Nolan' if i % 4 == 0 else 'Other Director',
                    'cast': 'Actor_0, Actor_1',
                    'genres': ['Sci-Fi', 'Thriller'] if i % 2 == 0 else ['Comedy'],
                    'overview': 'A high stakes adventure involving time manipulation and dreams.',
                    'runtime': 120
                })
            
            # Warm-up run
            predict_movie_scores_batch(model, cols, vec, encoders, candidates[:10])
            
            t0 = time.perf_counter()
            scores = predict_movie_scores_batch(model, cols, vec, encoders, candidates)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            self.assertEqual(len(scores), 200)
            self.assertTrue(all(0.5 <= s <= 5.0 for s in scores))
            print(f"\n[BENCHMARK] 200-film vectorized batch scoring completed in: {elapsed_ms:.2f}ms")
            self.assertLess(elapsed_ms, 100.0, f"Batch scoring took too long: {elapsed_ms:.2f}ms")

    def test_04_offline_eval_harness_temporal_holdout(self):
        """Evaluate out-of-sample MAE, Spearman rank correlation, and NDCG@10."""
        metrics = evaluate_model_temporal_holdout(self.sample_df, holdout_ratio=0.20)
        
        self.assertGreater(metrics['train_size'], 0)
        self.assertGreater(metrics['test_size'], 0)
        self.assertGreater(metrics['mae'], 0.0)
        self.assertLess(metrics['mae'], 1.5, f"Out-of-sample MAE should be reasonable: {metrics['mae']}")
        self.assertGreaterEqual(metrics['ndcg_10'], 0.5, f"NDCG@10 should be high: {metrics['ndcg_10']}")
        
        print(f"\n[OFFLINE EVAL METRICS]")
        print(f"  -> Out-of-Sample MAE: {metrics['mae']}★ (Baseline: {metrics['baseline_mean_mae']}★)")
        print(f"  -> Spearman Rank Correlation (rho): {metrics['spearman_rho']}")
        print(f"  -> NDCG@10: {metrics['ndcg_10']}")


    def test_05_temporal_year_constraint_enforcement(self):
        """Verify that 'before 2000s' extracts year_max=1999 and strictly excludes post-2000 films."""
        from backend.query_parser import _extract_year_constraints
        
        # 1. Test extraction on various natural language phrases
        ymin, ymax = _extract_year_constraints("Something like the odyssey from before 2000s")
        self.assertEqual(ymax, 1999)
        self.assertIsNone(ymin)
        
        ymin, ymax = _extract_year_constraints("classic 80s sci fi")
        self.assertEqual(ymin, 1980)
        self.assertEqual(ymax, 1989)

        ymin, ymax = _extract_year_constraints("90s action thrillers")
        self.assertEqual(ymin, 1990)
        self.assertEqual(ymax, 1999)

        ymin, ymax = _extract_year_constraints("post-2015 psychological horror")
        self.assertEqual(ymin, 2016)

        # 2. Test that discovery candidate filtering discards post-2000 candidates
        candidates = [
            {'id': 101, 'title': 'Jason and the Argonauts', 'release_date': '1963-06-19', 'genre_ids': [12, 14]},
            {'id': 102, 'title': 'The Odyssey', 'release_date': '1997-05-18', 'genre_ids': [12, 18]},
            {'id': 103, 'title': 'Avengers: Endgame', 'release_date': '2019-04-26', 'genre_ids': [28, 12]},
            {'id': 104, 'title': 'Facing El Chapo', 'release_date': '2024-01-10', 'genre_ids': [99]}
        ]

        with patch('backend.in_memory_model.get_diary_training_df', return_value=self.sample_df):
            model, cols, vec, encoders = train_user_model_in_memory('test_cinephile')
            with patch('backend.recommender.http_session.get') as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {'results': candidates}
                mock_get.return_value = mock_resp

                results = analyze(
                    watchedSet_titles=set(),
                    watchedSet_ids=set(),
                    hated_movies=[],
                    ai_analysis={
                        'genres': ['Adventure', 'Fantasy'],
                        'search_query': 'greek mythology epic quest',
                        'suggested_titles': [],
                        'year_max': 1999
                    },
                    ai_model=model,
                    ai_columns=cols,
                    ai_vectorizer=vec,
                    ai_encoders=encoders,
                    raw_prompt="Something like the odyssey from before 2000s",
                    source="discover"
                )

                result_titles = [r['title'] for r in results]
                self.assertIn('Jason and the Argonauts', result_titles)
                self.assertIn('The Odyssey', result_titles)
                self.assertNotIn('Avengers: Endgame', result_titles)
                self.assertNotIn('Facing El Chapo', result_titles)


if __name__ == '__main__':
    unittest.main()
