"""
Regression tests for negated prompts ("movies without nudity").

Negated words must never become search terms, and exclusions must be enforced
against TMDB keyword tags rather than overview text (overviews rarely say "nudity").
"""
import os
import sys
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.recommender as recommender
from backend.query_parser import interpret_query, _extract_negated_terms

NUDITY_TAGS = [
    {'id': 281741, 'name': 'nudity'},
    {'id': 359980, 'name': 'female nudity'},
    {'id': 54613, 'name': 'purity'},       # fuzzy search noise: must not be excluded
    {'id': 10031, 'name': 'nudism'},       # not a whole-word match
]
GORE_TAGS = [
    {'id': 10292, 'name': 'gore'},
    {'id': 210024, 'name': 'extreme gore'},
]
TAGGED_NUDITY = {'id': 3, 'title': 'Tagged Film', 'genre_ids': [18], 'release_date': '2010-01-01',
                 'vote_average': 7.0, 'overview': 'A quiet drama.'}
CLEAN = {'id': 4, 'title': 'Clean Film', 'genre_ids': [18], 'release_date': '2011-01-01',
         'vote_average': 7.2, 'overview': 'Another quiet drama.'}
MOVIE_TAGS = {3: [{'id': 359980, 'name': 'female nudity'}], 4: [{'id': 999, 'name': 'friendship'}],
              5: [{'id': 10292, 'name': 'gore'}], 6: []}


class FakeTmdb:
    """Routes TMDB URLs to canned responses and records every discover call."""

    def __init__(self):
        self.discover_calls = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        resp = Mock()
        resp.status_code = 200
        path = url.split('/3', 1)[-1]
        if path == '/search/keyword':
            q = params.get('query', '')
            results = NUDITY_TAGS if q == 'nudity' else GORE_TAGS if q == 'gore' else []
            resp.json.return_value = {'results': results, 'total_pages': 1}
        elif path == '/discover/movie':
            self.discover_calls.append(dict(params))
            # Real TMDB drops films carrying any without_keywords tag; emulate it.
            excluded = set(str(params.get('without_keywords', '')).split('|'))
            pool = [TAGGED_NUDITY, CLEAN]
            resp.json.return_value = {'results': [dict(m) for m in pool
                                                  if not {str(t['id']) for t in MOVIE_TAGS[m['id']]} & excluded]}
        elif path.startswith('/movie/') and path.split('/')[2].isdigit() and len(path.split('/')) == 3:
            mid = int(path.split('/')[2])
            resp.json.return_value = {'id': mid, 'original_language': 'en',
                                      'keywords': {'keywords': MOVIE_TAGS.get(mid, [])}}
        elif path == '/search/movie':
            q = params.get('query', '')
            titles = {'Reference Film': {'id': 10, 'title': 'Reference Film', 'genre_ids': [18], 'release_date': '2000-01-01'},
                      'No Time to Die': {'id': 7, 'title': 'No Time to Die', 'genre_ids': [28], 'release_date': '2021-09-29'}}
            resp.json.return_value = {'results': [dict(titles[q])] if q in titles else []}
        elif path == '/movie/10/recommendations':
            # Recommendations can't be filtered server-side: tagged film comes through here
            resp.json.return_value = {'results': [dict(TAGGED_NUDITY), dict(CLEAN)]}
        else:
            resp.json.return_value = {'results': []}
        return resp


def run_analyze(prompt, source='all', username=None, watchlist=None):
    fake = FakeTmdb()
    recommender._exclusion_tag_cache.clear()
    recommender._movie_tag_cache.clear()
    patches = [patch.object(recommender.http_session, 'get', side_effect=fake.get)]
    if username:
        patches += [patch('backend.db.get_user', return_value=None),
                    patch('backend.db.get_user_watchlist', return_value=watchlist or []),
                    patch('backend.db.get_user_diary', return_value=([], 0, None))]
    for p in patches:
        p.start()
    try:
        picks = recommender.analyze(set(), set(), [], interpret_query(prompt), None, None, None, None,
                                    raw_prompt=prompt, source=source, username=username, tmdb_key='KEY')
    finally:
        for p in patches:
            p.stop()
    return picks, fake


class TestNegationParsing(unittest.TestCase):

    def test_negated_terms_and_positive_text(self):
        self.assertEqual(_extract_negated_terms('Movies without nudity'), (['nudity'], 'Movies'))
        self.assertEqual(_extract_negated_terms('thrillers without gore or nudity'), (['gore', 'nudity'], 'thrillers'))
        self.assertEqual(_extract_negated_terms('dark sci-fi with no romance'), (['romance'], 'dark sci-fi'))
        self.assertEqual(_extract_negated_terms('non-anime fantasy'), (['anime'], 'fantasy'))

    def test_degree_words_are_not_exclusions(self):
        terms, positive = _extract_negated_terms('not too long comedy')
        self.assertEqual(terms, [])
        self.assertIn('comedy', positive)

    def test_negated_word_never_drives_search(self):
        a = interpret_query('Movies without nudity')
        self.assertEqual(a['genres'], [], 'negated "nudity" must not map to Romance/Drama/Thriller')
        self.assertEqual(a['search_query'], '')
        self.assertEqual(a['thematic_keywords'], [])
        self.assertTrue(a['exclusion_only'])
        self.assertIn('nudity', a['excluded_keywords'])

    def test_reference_title_drops_trailing_exclusion(self):
        a = interpret_query('movies like Drive but without gore')
        self.assertEqual(a['reference_entity'], 'Drive')
        self.assertEqual(a['negated_terms'], ['gore'])

    def test_negation_word_inside_reference_title_is_not_an_exclusion(self):
        a = interpret_query('something like No Country for Old Men')
        self.assertEqual(a['reference_entity'], 'No Country for Old Men')
        self.assertEqual(a['negated_terms'], [])

    def test_negation_word_title_is_not_a_person(self):
        self.assertIsNone(interpret_query('No Time to Die')['person'])
        self.assertIsNone(interpret_query('Not Okay')['person'])

    def test_positive_part_still_parsed(self):
        a = interpret_query('scary movies without gore')
        self.assertIn('Horror', a['genres'])
        self.assertEqual(a['negated_terms'], ['gore'])
        self.assertFalse(a['exclusion_only'])


class TestTagExclusions(unittest.TestCase):

    def test_without_nudity_never_searches_for_nudity_tags(self):
        picks, fake = run_analyze('Movies without nudity')
        self.assertTrue(fake.discover_calls, 'exclusion-only prompt should still discover films')
        for params in fake.discover_calls:
            self.assertNotIn('281741', str(params.get('with_keywords', '')))
            self.assertNotIn('359980', str(params.get('with_keywords', '')))

    def test_without_nudity_sends_tags_as_exclusions(self):
        picks, fake = run_analyze('Movies without nudity')
        for params in fake.discover_calls:
            excluded = set(params.get('without_keywords', '').split('|'))
            self.assertTrue({'281741', '359980'} <= excluded)
            self.assertNotIn('54613', excluded, '"purity" is search noise, not an exclusion')
            self.assertNotIn('10031', excluded)
        ids = [p['id'] for p in picks]
        self.assertNotIn(3, ids)
        self.assertIn(4, ids)

    def test_tagged_film_from_recommendations_is_dropped(self):
        picks, fake = run_analyze('something like Reference Film without nudity')
        ids = [p['id'] for p in picks]
        self.assertNotIn(3, ids, 'tagged film arriving via TMDB recommendations must be filtered')
        self.assertIn(4, ids)

    def test_title_starting_with_negation_word_still_found(self):
        picks, _ = run_analyze('No Time to Die')
        directs = [p for p in picks if p.get('is_direct_match')]
        self.assertTrue(directs and directs[0]['id'] == 7, 'exact title match must win over the parsed exclusion')

    def test_without_gore_excludes_gore_tags(self):
        picks, fake = run_analyze('scary movies without gore')
        self.assertTrue(fake.discover_calls)
        for params in fake.discover_calls:
            self.assertNotIn('10292', str(params.get('with_keywords', '')))
            excluded = set(params.get('without_keywords', '').split('|'))
            self.assertTrue({'10292', '210024'} <= excluded)

    def test_watchlist_mode_drops_tagged_films(self):
        watchlist = [
            {'movie_id': 5, 'title': 'Gory Horror', 'genres': 'Horror', 'overview': 'A haunted house.'},
            {'movie_id': 6, 'title': 'Quiet Horror', 'genres': 'Horror', 'overview': 'A haunted house.'},
        ]
        picks, _ = run_analyze('scary movies without gore', source='watchlist', username='tester', watchlist=watchlist)
        ids = [p['id'] for p in picks]
        self.assertNotIn(5, ids)
        self.assertIn(6, ids)


if __name__ == '__main__':
    unittest.main()
