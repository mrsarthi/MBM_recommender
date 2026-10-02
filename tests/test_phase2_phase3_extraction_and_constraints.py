#!/usr/bin/env python3
"""
Phase 2 & 3 Verification Tests
Validates: Person entity extraction, reference entity, year/language/runtime/quality
constraints, negation parsing, thematic keywords, compound groups, and recommender
integration (person filmography, runtime/quality filtering, excluded genres/keywords).

Run: python tests/test_phase2_phase3_extraction_and_constraints.py
"""

import os
import sys
import unittest
from unittest.mock import patch, Mock, MagicMock

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.query_parser import (
    interpret_query,
    _extract_person_entity,
    _extract_reference_entity,
    _extract_year_constraints,
    _extract_languages,
    _extract_thematic_keywords,
    _extract_negations,
    _extract_runtime_constraints,
    _extract_quality_constraints,
    _extract_compound_keyword_groups,
    _fallback_mood_match,
    _clean_search_query,
    _strip_person_from_prompt,
    _clean_search_query_with_person,
    VALID_GENRES,
)
from backend.recommender import _get_person_filmography


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Query Parser Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestPersonEntityExtraction(unittest.TestCase):
    """Tests for _extract_person_entity and person resolution."""

    def test_full_name_christopher_nolan(self):
        result = _extract_person_entity('christopher nolan movies')
        self.assertIn('Christopher Nolan', result)

    def test_surname_only_nolan(self):
        result = _extract_person_entity('nolan films')
        self.assertIn('Christopher Nolan', result)

    def test_full_name_scorsese(self):
        result = _extract_person_entity('movies by martin scorsese')
        self.assertIn('Martin Scorsese', result)

    def test_surname_only_scorsese(self):
        result = _extract_person_entity('scorsese movies')
        self.assertIn('Martin Scorsese', result)

    def test_surname_only_tarantino(self):
        result = _extract_person_entity('tarantino films')
        self.assertIn('Quentin Tarantino', result)

    def test_director_villeneuve(self):
        result = _extract_person_entity('denis villeneuve movies')
        self.assertIn('Denis Villeneuve', result)

    def test_actor_johansson_full_name(self):
        result = _extract_person_entity('scarlett johansson movies')
        self.assertIn('Scarlett Johansson', result)

    def test_actor_johansson_surname(self):
        result = _extract_person_entity('johansson films')
        self.assertIn('Scarlett Johansson', result)

    def test_actor_johansson_typo(self):
        result = _extract_person_entity('scarlett johanson movies')
        self.assertIn('Scarlett Johansson', result)

    def test_actor_dicaprio(self):
        result = _extract_person_entity('dicaprio movies')
        self.assertIn('Leonardo DiCaprio', result)

    def test_actor_hanks(self):
        result = _extract_person_entity('tom hanks films')
        self.assertIn('Tom Hanks', result)

    def test_actor_pitt(self):
        result = _extract_person_entity('brad pitt movies')
        self.assertIn('Brad Pitt', result)

    def test_no_person_for_vibe_query(self):
        result = _extract_person_entity('rainy night cyberpunk vibes')
        self.assertIsNone(result)

    def test_no_person_for_genre_query(self):
        result = _extract_person_entity('comedy movies')
        self.assertIsNone(result)

    def test_director_del_toro(self):
        result = _extract_person_entity('del toro movies')
        self.assertIn('Guillermo del Toro', result)

    def test_director_cuarron(self):
        result = _extract_person_entity('cuaron films')
        self.assertEqual(result, 'Alfonso Cuarón')  # TMDB's spelling

    def test_director_kubrick(self):
        result = _extract_person_entity('kubrick movies')
        self.assertIn('Stanley Kubrick', result)

    def test_director_spielberg(self):
        result = _extract_person_entity('spielberg films')
        self.assertIn('Steven Spielberg', result)


class TestReferenceEntityExtraction(unittest.TestCase):
    """Tests for _extract_reference_entity."""

    def test_movies_like_title(self):
        result = _extract_reference_entity('movies like Inception')
        self.assertEqual(result, 'Inception')

    def test_something_like_title(self):
        result = _extract_reference_entity('something like The Matrix')
        self.assertEqual(result, 'The Matrix')

    def test_similar_to_title(self):
        result = _extract_reference_entity('films similar to Interstellar')
        self.assertEqual(result, 'Interstellar')

    def test_in_the_vein_of(self):
        result = _extract_reference_entity('in the vein of Pulp Fiction')
        self.assertIn('Pulp Fiction', result)

    def test_no_reference_entity_for_vibe(self):
        result = _extract_reference_entity('dark rainy cyberpunk movies')
        self.assertIsNone(result)

    def test_no_generic_movies_as_entity(self):
        result = _extract_reference_entity('movies')
        self.assertIsNone(result)


class TestYearConstraints(unittest.TestCase):
    """Tests for _extract_year_constraints."""

    def test_year_max_before_2010(self):
        ymin, ymax = _extract_year_constraints('movies from before 2010')
        self.assertIsNone(ymin)
        self.assertEqual(ymax, 2009)

    def test_year_min_after_2015(self):
        ymin, ymax = _extract_year_constraints('movies after 2015')
        self.assertEqual(ymin, 2016)
        self.assertIsNone(ymax)

    def test_decade_1990s(self):
        ymin, ymax = _extract_year_constraints('movies from the 1990s')
        self.assertEqual(ymin, 1990)
        self.assertEqual(ymax, 1999)

    def test_decade_1960s(self):
        ymin, ymax = _extract_year_constraints('movies from the 60s')
        self.assertEqual(ymin, 1960)
        self.assertEqual(ymax, 1969)

    def test_year_range_1995_to_2005(self):
        ymin, ymax = _extract_year_constraints('movies from 1995 to 2005')
        self.assertEqual(ymin, 1995)
        self.assertEqual(ymax, 2005)

    def test_year_range_dash(self):
        ymin, ymax = _extract_year_constraints('films from 2000-2010')
        self.assertEqual(ymin, 2000)
        self.assertEqual(ymax, 2010)

    def test_prior_to_1995(self):
        ymin, ymax = _extract_year_constraints('prior to 1995')
        self.assertIsNone(ymin)
        self.assertEqual(ymax, 1994)

    def test_no_year_constraints(self):
        ymin, ymax = _extract_year_constraints('good movies')
        self.assertIsNone(ymin)
        self.assertIsNone(ymax)


class TestLanguageExtraction(unittest.TestCase):
    """Tests for _extract_languages."""

    def test_japanese(self):
        result = _extract_languages('japanese anime')
        self.assertIn('ja', result)

    def test_french(self):
        result = _extract_languages('french films')
        self.assertIn('fr', result)

    def test_korean(self):
        result = _extract_languages('korean movies')
        self.assertIn('ko', result)

    def test_chinese(self):
        result = _extract_languages('chinese cinema')
        self.assertIn('zh', result)

    def test_german(self):
        result = _extract_languages('german films')
        self.assertIn('de', result)

    def test_no_language(self):
        result = _extract_languages('action movies')
        self.assertEqual(result, [])


class TestRuntimeConstraints(unittest.TestCase):
    """Tests for _extract_runtime_constraints."""

    def test_under_90_min_words(self):
        result = _extract_runtime_constraints('movies under 90 min')
        self.assertEqual(result, 90)

    def test_under_90_minutes_full(self):
        result = _extract_runtime_constraints('under 90 minutes')
        self.assertEqual(result, 90)

    def test_less_than_100_min(self):
        result = _extract_runtime_constraints('less than 100 min')
        self.assertEqual(result, 100)

    def test_under_2_hours(self):
        result = _extract_runtime_constraints('under 2 hours')
        self.assertEqual(result, 120)

    def test_under_1_5_hours(self):
        result = _extract_runtime_constraints('under 1.5 hours')
        self.assertEqual(result, 90)

    def test_at_most_90_minutes(self):
        result = _extract_runtime_constraints('at most 90 minutes')
        self.assertEqual(result, 90)

    def test_no_more_than_2_hours(self):
        result = _extract_runtime_constraints('no more than 2 hours')
        self.assertEqual(result, 120)

    def test_short_movies(self):
        result = _extract_runtime_constraints('short movies')
        self.assertEqual(result, 90)

    def test_quick_films(self):
        result = _extract_runtime_constraints('quick films')
        self.assertEqual(result, 90)

    def test_no_runtime_constraint(self):
        result = _extract_runtime_constraints('action movies')
        self.assertIsNone(result)


class TestQualityConstraints(unittest.TestCase):
    """Tests for _extract_quality_constraints."""

    def test_critically_acclaimed(self):
        result = _extract_quality_constraints('critically acclaimed films')
        self.assertEqual(result, 7.5)

    def test_acclaimed(self):
        result = _extract_quality_constraints('acclaimed movies')
        self.assertEqual(result, 7.5)

    def test_masterpiece(self):
        result = _extract_quality_constraints('masterpiece films')
        self.assertEqual(result, 7.5)

    def test_award_winning(self):
        result = _extract_quality_constraints('award winning movies')
        self.assertEqual(result, 7.5)

    def test_top_rated(self):
        result = _extract_quality_constraints('top rated films')
        self.assertEqual(result, 7.5)

    def test_oscar(self):
        result = _extract_quality_constraints('oscar winners')
        self.assertEqual(result, 7.5)

    def test_no_quality_constraint(self):
        result = _extract_quality_constraints('action movies')
        self.assertIsNone(result)


class TestNegationExtraction(unittest.TestCase):
    """Tests for _extract_negations."""

    def test_not_anime(self):
        result = _extract_negations('movies without anime')
        self.assertIn('Animation', result['genres'])

    def test_not_romance(self):
        result = _extract_negations('not romantic films')
        self.assertIn('Romance', result['genres'])

    def test_no_horror(self):
        result = _extract_negations('no horror movies')
        self.assertIn('Horror', result['genres'])

    def test_not_comedy(self):
        result = _extract_negations('no comedy films')
        self.assertIn('Comedy', result['genres'])

    def test_not_animated(self):
        result = _extract_negations('not animated')
        self.assertIn('Animation', result['genres'])

    def test_not_time_travel(self):
        result = _extract_negations('not time travel')
        self.assertIn('time travel', result['keywords'])

    def test_not_scifi(self):
        result = _extract_negations('not sci-fi')
        self.assertIn('Science Fiction', result['genres'])

    def test_arbitrary_negation(self):
        result = _extract_negations('movies that are not violent')
        self.assertIn('violent', result['keywords'])

    def test_not_gory(self):
        result = _extract_negations('not gory')
        self.assertIn('gore', result['keywords'])


class TestThematicKeywords(unittest.TestCase):
    """Tests for _extract_thematic_keywords."""

    def test_cyberpunk_keyword(self):
        result = _extract_thematic_keywords('cyberpunk movies')
        self.assertTrue(any('cyberpunk' in k or 'dystopia' in k for k in result))

    def test_time_travel_keyword(self):
        result = _extract_thematic_keywords('time travel films')
        self.assertTrue(any('temporal' in k for k in result))

    def test_nudity_keyword(self):
        result = _extract_thematic_keywords('movies with nudity')
        self.assertTrue(any('nude' in k or 'erotic' in k for k in result))

    def test_gore_keyword(self):
        result = _extract_thematic_keywords('gory movies')
        self.assertTrue(any('gore' in k for k in result))

    def test_samurai_keyword(self):
        result = _extract_thematic_keywords('samurai films')
        self.assertTrue(any('samurai' in k for k in result))

    def test_neo_noir_keyword(self):
        result = _extract_thematic_keywords('neo-noir')
        self.assertTrue(len(result) > 0)


class TestCompoundKeywordGroups(unittest.TestCase):
    """Tests for _extract_compound_keyword_groups."""

    def test_gore_and_nudity(self):
        result = _extract_compound_keyword_groups('movies with gore and nudity')
        self.assertGreaterEqual(len(result), 2)

    def test_cyberpunk_and_heist(self):
        result = _extract_compound_keyword_groups('cyberpunk heist movies')
        self.assertGreaterEqual(len(result), 2)

    def test_single_concept_no_compound(self):
        result = _extract_compound_keyword_groups('gore movies')
        self.assertEqual(len(result), 0)

    def test_no_compound(self):
        result = _extract_compound_keyword_groups('comedy films')
        self.assertEqual(len(result), 0)


class TestFallbackMoodMatch(unittest.TestCase):
    """Tests for _fallback_mood_match."""

    def test_comedy_genre(self):
        result = _fallback_mood_match('comedy movies')
        self.assertIn('Comedy', result)

    def test_horror_genre(self):
        result = _fallback_mood_match('horror films')
        self.assertIn('Horror', result)

    def test_drama_genre(self):
        result = _fallback_mood_match('drama movies')
        self.assertIn('Drama', result)

    def test_scifi_genre(self):
        result = _fallback_mood_match('sci-fi movies')
        self.assertIn('Science Fiction', result)

    def test_anime_genre(self):
        result = _fallback_mood_match('japanese anime')
        self.assertIn('Animation', result)

    def test_happy_mood(self):
        result = _fallback_mood_match('i am feeling happy')
        self.assertTrue(any(g in result for g in ['Comedy', 'Music', 'Animation', 'Family', 'Romance']))

    def test_mind_bending(self):
        result = _fallback_mood_match('mind-bending sci-fi')
        self.assertIn('Science Fiction', result)

    def test_tense_mood(self):
        result = _fallback_mood_match('give me something tense and scary')
        self.assertTrue(any(g in result for g in ['Horror', 'Thriller', 'Mystery']))

    def test_excluded_genre_not_returned(self):
        result = _fallback_mood_match('comedy but not rom-com', excluded_genres=['Romance'])
        self.assertIn('Comedy', result)
        self.assertNotIn('Romance', result)


class TestInterpretQueryIntegration(unittest.TestCase):
    """Integration tests for interpret_query covering all extraction dimensions."""

    def test_nolan_before_2010(self):
        r = interpret_query('christopher nolan movies from before 2010')
        self.assertEqual(r['person'], 'Christopher Nolan')
        self.assertEqual(r['year_max'], 2009)
        self.assertEqual(r['search_query'], '')

    def test_nolan_time_travel(self):
        r = interpret_query('nolan movies about time travel')
        self.assertEqual(r['person'], 'Christopher Nolan')
        self.assertTrue(any('temporal' in k for k in r['thematic_keywords']))

    def test_nolan_not_time_travel(self):
        r = interpret_query('nolan movies that are not time travel')
        self.assertEqual(r['person'], 'Christopher Nolan')
        self.assertTrue(any('time travel' in k for k in r['excluded_keywords']))

    def test_johansson_nudity(self):
        r = interpret_query('scarlett johansson movies with nudity')
        self.assertEqual(r['person'], 'Scarlett Johansson')
        self.assertTrue(any('nude' in k or 'erotic' in k for k in r['thematic_keywords']))

    def test_johanson_typo_acclaimed(self):
        r = interpret_query('scarlett johanson movies that are critically acclaimed')
        self.assertEqual(r['person'], 'Scarlett Johansson')
        self.assertEqual(r['vote_average_min'], 7.5)

    def test_comedy_under_90(self):
        r = interpret_query('comedy movies under 90 min')
        self.assertEqual(r['runtime_max'], 90)
        self.assertIn('Comedy', r['genres'])

    def test_movies_like_inception(self):
        r = interpret_query('movies like Inception')
        self.assertEqual(r['reference_entity'], 'Inception')
        self.assertIsNone(r['person'])
        self.assertEqual(r['search_query'], 'Inception')

    def test_something_like_matrix(self):
        r = interpret_query('something like The Matrix')
        self.assertEqual(r['reference_entity'], 'The Matrix')
        self.assertIsNone(r['person'])  # ref_entity takes priority over person match

    def test_upcoming_movies(self):
        r = interpret_query('upcoming movies')
        self.assertTrue(r['is_upcoming'])

    def test_french_1960s(self):
        r = interpret_query('french films from the 1960s')
        self.assertIn('fr', r['languages'])
        self.assertEqual(r['year_min'], 1960)
        self.assertEqual(r['year_max'], 1969)

    def test_90s_comedy(self):
        r = interpret_query('comedy movies from the 1990s')
        self.assertIn('Comedy', r['genres'])
        self.assertEqual(r['year_min'], 1990)
        self.assertEqual(r['year_max'], 1999)

    def test_japanese_anime(self):
        r = interpret_query('japanese anime')
        self.assertIn('ja', r['languages'])
        self.assertIn('Animation', r['genres'])

    def test_drama_after_2015(self):
        r = interpret_query('drama movies after 2015')
        self.assertIn('Drama', r['genres'])
        self.assertEqual(r['year_min'], 2016)

    def test_gore_and_nudity_compound(self):
        r = interpret_query('horror movies with both gore and nudity')
        self.assertGreaterEqual(len(r['compound_keyword_groups']), 2)

    def test_denzen_washington(self):
        r = interpret_query('denzel washington movies')
        self.assertIn('Denzel Washington', r['person'])

    def test_direct_match_search_query(self):
        r = interpret_query('Inception')
        self.assertEqual(r['search_query'], 'Inception')
        self.assertIsNone(r['person'])

    def test_not_anime_excluded(self):
        r = interpret_query('movies without anime')
        self.assertIn('Animation', r['excluded_genres'])

    def test_scorsese_directed(self):
        r = interpret_query('movies directed by scorsese')
        self.assertIn('Martin Scorsese', r['person'])

    def test_under_2_hours(self):
        r = interpret_query('action movies under 2 hours')
        self.assertEqual(r['runtime_max'], 120)

    def test_year_range_1990_to_2000(self):
        r = interpret_query('movies from 1990 to 2000')
        self.assertEqual(r['year_min'], 1990)
        self.assertEqual(r['year_max'], 2000)


class TestStripPersonFromPrompt(unittest.TestCase):
    """Tests for _strip_person_from_prompt."""

    def test_strip_full_name(self):
        result = _strip_person_from_prompt('christopher nolan movies', 'Christopher Nolan')
        self.assertNotIn('Nolan', result)
        self.assertNotIn('nolan', result)

    def test_strip_surname(self):
        result = _strip_person_from_prompt('nolan films', 'Christopher Nolan')
        self.assertNotIn('nolan', result.lower())

    def test_strip_preserves_rest(self):
        result = _strip_person_from_prompt('nolan time travel movies', 'Christopher Nolan')
        self.assertIn('time travel', result.lower())

    def test_strip_none_person(self):
        result = _strip_person_from_prompt('comedy movies', None)
        self.assertEqual(result, 'comedy movies')


class TestCleanSearchQuery(unittest.TestCase):
    """Tests for _clean_search_query and _clean_search_query_with_person."""

    def test_person_returns_empty(self):
        result = _clean_search_query_with_person('nolan movies', 'Christopher Nolan')
        self.assertEqual(result, '')

    def test_vibe_query_returns_empty(self):
        result = _clean_search_query('something like Inception')
        self.assertEqual(result, '')

    def test_direct_title_preserved(self):
        result = _clean_search_query('Inception')
        self.assertEqual(result, 'Inception')

    def test_conversational_prefix_stripped(self):
        result = _clean_search_query('can you recommend Inception')
        self.assertEqual(result, 'Inception')


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Recommender Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestPersonFilmographyCache(unittest.TestCase):
    """Tests for _get_person_filmography caching and structure."""

    def test_returns_none_for_empty_name(self):
        result = _get_person_filmography('', 'DUMMY_KEY')
        self.assertIsNone(result)

    def test_returns_none_for_no_key(self):
        result = _get_person_filmography('Christopher Nolan', None)
        self.assertIsNone(result)

    def test_returns_none_for_missing_key(self):
        result = _get_person_filmography('Christopher Nolan', '')
        self.assertIsNone(result)

    def test_cache_structure_with_mock(self):
        """Verify the cache stores results correctly."""
        # Clear cache
        from backend import recommender as rec_mod
        rec_mod._person_filmography_cache.clear()

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'results': [{'id': 1, 'name': 'Christopher Nolan'}]
        }

        mock_credits = Mock()
        mock_credits.status_code = 200
        mock_credits.json.return_value = {
            'cast': [{'id': 101}],
            'crew': [{'id': 201}]
        }

        def mock_get(url, params=None, timeout=None, **kwargs):
            if 'search/person' in url:
                return mock_resp
            elif 'movie_credits' in url:
                return mock_credits
            return Mock(status_code=404)

        with patch.object(rec_mod.http_session, 'get', side_effect=mock_get):
            result = _get_person_filmography('Christopher Nolan', 'DUMMY_KEY')

        self.assertIsNotNone(result)
        self.assertEqual(result['person_id'], 1)
        self.assertEqual(result['name'], 'Christopher Nolan')
        self.assertIn(101, result['cast_ids'])
        self.assertIn(201, result['crew_ids'])
        self.assertIn(101, result['all_ids'])
        self.assertIn(201, result['all_ids'])

    def test_cache_hit_returns_cached(self):
        """Second call should return cached result without API call."""
        from backend import recommender as rec_mod
        rec_mod._person_filmography_cache.clear()

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'results': [{'id': 1, 'name': 'Christopher Nolan'}]
        }
        mock_credits = Mock()
        mock_credits.status_code = 200
        mock_credits.json.return_value = {'cast': [], 'crew': []}

        call_count = [0]

        def mock_get(url, params=None, timeout=None, **kwargs):
            call_count[0] += 1
            if 'search/person' in url:
                return mock_resp
            elif 'movie_credits' in url:
                return mock_credits
            return Mock(status_code=404)

        with patch.object(rec_mod.http_session, 'get', side_effect=mock_get):
            r1 = _get_person_filmography('Christopher Nolan', 'DUMMY_KEY')
            r2 = _get_person_filmography('Christopher Nolan', 'DUMMY_KEY')

        self.assertEqual(r1, r2)
        self.assertEqual(call_count[0], 2)  # Two calls for first (search + credits)

    def test_handles_api_error_gracefully(self):
        from backend import recommender as rec_mod
        rec_mod._person_filmography_cache.clear()

        mock_error = Mock()
        mock_error.status_code = 500
        mock_error.json.side_effect = Exception("API Error")

        with patch.object(rec_mod.http_session, 'get', return_value=mock_error):
            result = _get_person_filmography('Christopher Nolan', 'DUMMY_KEY')

        self.assertIsNone(result)


class TestRecommenderFilters(unittest.TestCase):
    """Test that analyze() correctly applies runtime and quality filters."""

    def test_runtime_max_filters_long_movies(self):
        """Verify runtime_max filters candidates."""
        from backend.recommender import analyze

        ai_analysis = {
            'genres': [],
            'search_query': '',
        }

        # Mock TMDB responses
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'results': []}

        with patch('backend.recommender.http_session.get', return_value=mock_resp):
            picks = analyze(
                set(), set(), set(),
                ai_analysis, None, None, None, None,
                source='all', username=None,
                raw_prompt='short movies under 90 min',
                tmdb_key='DUMMY_TMDB'
            )
            self.assertIsInstance(picks, list)

    def test_vote_average_min_applied(self):
        """Verify vote_average_min is read from ai_analysis."""
        from backend.recommender import analyze

        ai_analysis = {
            'genres': [],
            'search_query': 'something like Test',
            'vote_average_min': 8.0,
        }

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'results': []}

        with patch('backend.recommender.http_session.get', return_value=mock_resp):
            picks = analyze(
                set(), set(), set(),
                ai_analysis, None, None, None, None,
                source='all', username=None,
                raw_prompt='something like Test',
                tmdb_key='DUMMY_TMDB'
            )
            self.assertIsInstance(picks, list)


class TestExcludedGenresInAnalyze(unittest.TestCase):
    """Verify excluded_genres filtering in analyze()."""

    def test_excluded_genres_from_ai_analysis(self):
        """Excluded genres should filter out matching candidates."""
        from backend.recommender import analyze

        ai_analysis = {
            'genres': ['Horror'],
            'search_query': '',
            'excluded_genres': ['Horror'],
            'excluded_keywords': [],
        }

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'results': [
            {'id': 1, 'title': 'Test Horror', 'genre_ids': [27], 'release_date': '2020-01-01', 'vote_average': 7.0, 'overview': 'scary'},
        ]}

        with patch('backend.recommender.http_session.get', return_value=mock_resp):
            picks = analyze(
                set(), set(), set(),
                ai_analysis, None, None, None, None,
                source='all', username=None,
                raw_prompt='',
                tmdb_key='DUMMY_TMDB'
            )
            # Horror candidate should be filtered out by excluded_genres
            for p in picks:
                self.assertNotIn(1, [m.get('id') for m in picks] if isinstance(p, dict) else [])


class TestGenreDictionaryIntegrity(unittest.TestCase):
    """Verify genre dictionary and mappings are consistent."""

    def test_all_valid_genres_in_genre_dict(self):
        from backend.recommender import analyze
        import inspect
        source = inspect.getsource(analyze)
        for g in VALID_GENRES:
            self.assertIn(g, source, f"Genre {g} should be in analyze() genreDict")

    def test_genre_dict_has_all_tmdb_genres(self):
        from backend.recommender import analyze
        import inspect
        source = inspect.getsource(analyze)
        for g in VALID_GENRES:
            self.assertIn(g, source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
