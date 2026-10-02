import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from backend.query_parser import (
    _extract_year_constraints,
    _extract_reference_entity,
    interpret_query
)
from backend.recommender import analyze
from backend.in_memory_model import train_user_model_in_memory


class TestDiscoverySearchPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {'Name': 'Blade Runner', 'Year': 1982, 'Rating': 5.0, 'Directors': 'Ridley Scott', 'Cast': 'Harrison Ford, Rutger Hauer', 'Genres': 'Science Fiction, Drama', 'Date': '2024-01-01'},
            {'Name': 'Alien', 'Year': 1979, 'Rating': 4.5, 'Directors': 'Ridley Scott', 'Cast': 'Sigourney Weaver, Tom Skerritt', 'Genres': 'Horror, Science Fiction', 'Date': '2024-01-02'},
            {'Name': '2001: A Space Odyssey', 'Year': 1968, 'Rating': 5.0, 'Directors': 'Stanley Kubrick', 'Cast': 'Keir Dullea, Gary Lockwood', 'Genres': 'Science Fiction, Mystery', 'Date': '2024-01-03'},
            {'Name': 'Jason and the Argonauts', 'Year': 1963, 'Rating': 4.5, 'Directors': 'Don Chaffey', 'Cast': 'Todd Armstrong, Nancy Kovack', 'Genres': 'Adventure, Fantasy', 'Date': '2024-01-04'},
            {'Name': 'Clash of the Titans', 'Year': 1981, 'Rating': 4.0, 'Directors': 'Desmond Davis', 'Cast': 'Harry Hamlin, Laurence Olivier', 'Genres': 'Adventure, Fantasy', 'Date': '2024-01-05'},
            {'Name': 'The Room', 'Year': 2003, 'Rating': 0.5, 'Directors': 'Tommy Wiseau', 'Cast': 'Tommy Wiseau, Greg Sestero', 'Genres': 'Drama, Romance', 'Date': '2024-01-06'}
        ])

    def test_01_query_decomposition_entity_and_years(self):
        """Test deterministic regex extraction for reference entities and era bounds."""
        # Odyssey query
        prompt1 = "Something like the odyssey from before 2000s"
        ymin1, ymax1 = _extract_year_constraints(prompt1)
        ref1 = _extract_reference_entity(prompt1)
        self.assertEqual(ymax1, 1999)
        self.assertIsNone(ymin1)
        self.assertEqual(ref1.lower(), "the odyssey")

        # Shutter island query
        prompt2 = "Movies like Shutter Island"
        ymin2, ymax2 = _extract_year_constraints(prompt2)
        ref2 = _extract_reference_entity(prompt2)
        self.assertIsNone(ymin2)
        self.assertIsNone(ymax2)
        self.assertEqual(ref2, "Shutter Island")

        # 80s sci fi query
        prompt3 = "classic 80s sci-fi"
        ymin3, ymax3 = _extract_year_constraints(prompt3)
        self.assertEqual(ymin3, 1980)
        self.assertEqual(ymax3, 1989)

    def test_02_ai_interpretation_fallback_structure(self):
        """Test interpret_query returns a clean structured payload."""
        result = interpret_query("Something like the odyssey from before 2000s")
        self.assertEqual(result.get('year_max'), 1999)
        self.assertEqual(result.get('reference_entity', '').lower(), "the odyssey")
        self.assertIn('Adventure', result.get('genres', []))

    def test_04_reference_entity_ripple(self):
        """Test reference entity 'the odyssey' expands via TMDB recommendations."""
        fallback_analysis = {
            'genres': ['Adventure', 'Fantasy'],
            'search_query': 'Something like the odyssey from before 2000s',
            'reference_entity': 'the odyssey',
            'year_max': 1999
        }

        def mock_tmdb_get(url, params=None, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            p = params or {}
            
            if 'search/movie' in url and 'the odyssey' in p.get('query', '').lower():
                mock_resp.json.return_value = {'results': [{'id': 2001, 'title': 'The Odyssey', 'release_date': '1997-05-18', 'genre_ids': [12, 18]}]}
            elif 'movie/2001/recommendations' in url:
                mock_resp.json.return_value = {'results': [
                    {'id': 2002, 'title': 'Clash of the Titans', 'release_date': '1981-06-12', 'genre_ids': [12, 14]},
                    {'id': 2003, 'title': 'Hercules', 'release_date': '1997-06-27', 'genre_ids': [12, 16]},
                    {'id': 2004, 'title': 'Immortals', 'release_date': '2011-11-11', 'genre_ids': [14, 28]} # Filtered out!
                ]}
            else:
                mock_resp.json.return_value = {'results': []}
            return mock_resp

        with patch('backend.in_memory_model.get_diary_training_df', return_value=self.sample_df):
            model, cols, vec, encoders = train_user_model_in_memory('test_cinephile')
            with patch('backend.recommender.http_session.get', side_effect=mock_tmdb_get):
                results = analyze(
                    watchedSet_titles=set(),
                    watchedSet_ids=set(),
                    hated_movies=[],
                    ai_analysis=fallback_analysis,
                    ai_model=model,
                    ai_columns=cols,
                    ai_vectorizer=vec,
                    ai_encoders=encoders,
                    raw_prompt="Something like the odyssey from before 2000s",
                    source="discover"
                )

                result_titles = [r['title'] for r in results]
                self.assertIn('The Odyssey', result_titles)
                self.assertIn('Clash of the Titans', result_titles)
                self.assertIn('Hercules', result_titles)
                self.assertNotIn('Immortals', result_titles)


if __name__ == '__main__':
    unittest.main()
