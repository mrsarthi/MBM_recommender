import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from backend.query_parser import interpret_query_with_ai, _extract_negations
from backend.recommender import analyze
from backend.collaborative import collaborative_engine

class TestCollaborativeAndIntentGating(unittest.TestCase):

    def test_01_negation_extraction(self):
        """Test that negations (not anime, no romance, etc.) are correctly extracted."""
        neg1 = _extract_negations("old japanese movies that are critically acclaimed and are not anime")
        self.assertIn("Animation", neg1["genres"])
        self.assertTrue(any(k in neg1["keywords"] for k in ["anime", "animation", "animated"]))

        neg2 = _extract_negations("dark psychological thrillers with no comedy and without romance")
        self.assertIn("Comedy", neg2["genres"])
        self.assertIn("Romance", neg2["genres"])

    def test_02_query_interpretation_adult_thematic_keywords(self):
        """Test that sex/erotic queries extract specific thematic keywords and genres."""
        res = interpret_query_with_ai("sex heavy movies")
        self.assertTrue(any(g in res.get("genres", []) for g in ["Drama", "Romance", "Thriller"]))
        thematic_kws = res.get("thematic_keywords", [])
        self.assertTrue(len(thematic_kws) > 0)
        self.assertTrue(any(k in thematic_kws for k in ["erotic", "sexuality", "adult", "steamy", "sensual"]))

    def test_03_thematic_relevance_gate_rejects_ddlj(self):
        """Test that Thematic Relevance Gate severely penalizes unrelated high-rated films like DDLJ."""
        ai_analysis = interpret_query_with_ai("sex heavy movies")
        
        # Test mock candidates
        unrelated_movie = {
            'id': 19404,
            'title': 'Dilwale Dulhania Le Jayenge',
            'release_date': '1995-10-20',
            'genre_ids': [18, 10749],
            'genres': ['Drama', 'Romance'],
            'overview': 'Raj and Simran fall in love on vacation in Europe and must convince her traditional family.',
            'vote_average': 8.6,
            'vote_count': 4500
        }
        erotic_movie = {
            'id': 402,
            'title': 'Basic Instinct',
            'release_date': '1992-03-20',
            'genre_ids': [53, 9648],
            'genres': ['Thriller', 'Mystery'],
            'overview': 'A violent, sex-obsessed novelist becomes the prime suspect in an erotic murder case.',
            'vote_average': 6.9,
            'vote_count': 3800,
            'thematic_match': True
        }

        # Mock TMDB returning both
        results = analyze(
            watchedSet_titles=set(),
            watchedSet_ids=set(),
            hated_movies=[],
            ai_analysis=ai_analysis,
            ai_model=None,
            ai_columns=[],
            ai_vectorizer=None,
            ai_encoders={},
            raw_prompt="sex heavy movies"
        )
        
        # In live results or ranking, verify that any recommended movie doesn't falsely rank family romance above erotic content
        titles = [r.get('title') for r in results[:5]]
        self.assertNotIn('Dilwale Dulhania Le Jayenge', titles)

    def test_04_collaborative_engine_runs_safely(self):
        """Test that CollaborativeEngine initializes, trains without error, and returns predictions."""
        collaborative_engine.train()
        # Predictions for dummy user or non-existent user return empty dict safely
        preds = collaborative_engine.get_collaborative_predictions(9999999, [100, 200, 300])
        self.assertIsInstance(preds, dict)

    def test_05_dynamic_keyword_and_gore_rejection(self):
        """Test that dynamic keywords resolve arbitrary queries and reject unrelated blockbusters (e.g. Avengers)."""
        ai_analysis = interpret_query_with_ai("gore movies")
        results = analyze(
            watchedSet_titles=set(),
            watchedSet_ids=set(),
            hated_movies=[],
            ai_analysis=ai_analysis,
            ai_model=None,
            ai_columns=[],
            ai_vectorizer=None,
            ai_encoders={},
            raw_prompt="gore movies"
        )
        top_titles = [r.get('title') for r in results[:10]]
        self.assertNotIn('The Avengers', top_titles)
        self.assertNotIn('The Devil Wears Prada 2', top_titles)
        self.assertNotIn('Project Hail Mary', top_titles)

    def test_06_upcoming_query_parsing(self):
        """Test that forward-looking terms are parsed with is_upcoming=True and year_min=2026."""
        parsed = interpret_query_with_ai("upcoming highly anticipated movies")
        self.assertTrue(parsed.get('is_upcoming'))
        self.assertGreaterEqual(parsed.get('year_min', 0), 2026)

    def test_07_franchise_diversity_capping(self):
        """Test that franchise deduplication prevents sequels from flooding top results."""
        ai_analysis = interpret_query_with_ai("mind boggling movies")
        results = analyze(
            watchedSet_titles=set(),
            watchedSet_ids=set(),
            hated_movies=[],
            ai_analysis=ai_analysis,
            ai_model=None,
            ai_columns=[],
            ai_vectorizer=None,
            ai_encoders={},
            raw_prompt="mind boggling movies"
        )
        top_10 = [r.get('title') for r in results[:10]]
        saw_count = sum(1 for t in top_10 if 'saw' in str(t).lower() or 'jigsaw' in str(t).lower())
        self.assertLessEqual(saw_count, 1)

if __name__ == '__main__':
    unittest.main()


