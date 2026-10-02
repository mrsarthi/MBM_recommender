"""
Unit tests for prompt understanding (backend.concepts) and relevance scoring.

No network: these pin the behaviour that the live benchmark (tests/query_benchmark.py)
measures end to end.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.recommender as recommender
from backend.concepts import analyze_concepts
from backend.query_parser import interpret_query


def intent_for(prompt):
    """Intent without TMDB: tag ids are left empty."""
    a = interpret_query(prompt)
    return recommender._build_intent(a, a['positive_query'], None)


class TestConcepts(unittest.TestCase):

    def test_named_genre_is_required(self):
        c = analyze_concepts('dark sci-fi')
        self.assertIn(['Science Fiction'], c['genre_groups'])
        # "dark" is a preference, never a requirement
        self.assertNotIn(['Thriller'], c['genre_groups'])
        self.assertIn('Thriller', c['soft_genres'])

    def test_combined_genres_are_all_required(self):
        # Separate groups: a film must be Romance AND Comedy
        self.assertEqual(analyze_concepts('romantic comedies')['genre_groups'], [['Romance'], ['Comedy']])
        groups = analyze_concepts('zombie comedies')['genre_groups']
        self.assertIn(['Comedy'], groups)

    def test_mood_avoids_genres(self):
        c = analyze_concepts('feel good movies')
        self.assertIn('Horror', c['avoid_genres'])
        self.assertEqual(c['genre_groups'], [])

    def test_named_genre_overrides_mood_avoidance(self):
        # "dark" avoids Comedy, but the user asked for comedy
        c = analyze_concepts('dark comedy')
        self.assertNotIn('Comedy', c['avoid_genres'])
        self.assertIn(['Comedy'], c['genre_groups'])

    def test_moods_have_no_theme_defining_tags(self):
        self.assertEqual(analyze_concepts('dark gritty movies')['core_tags'], [])

    def test_specific_trigger_defines_its_own_theme(self):
        self.assertEqual(analyze_concepts('boxing movies')['core_tags'][0], 'boxing')
        self.assertEqual(analyze_concepts('dinosaur movies')['core_tags'][0], 'dinosaur')
        # A generic trigger keeps the whole family of tags
        self.assertIn('basketball', analyze_concepts('sports movies')['core_tags'])

    def test_broad_tags_are_not_core(self):
        c = analyze_concepts('christmas movies')
        self.assertIn('christmas', c['core_tags'])
        self.assertNotIn('holiday', c['core_tags'])
        self.assertIn('holiday', c['tags'])

    def test_excluded_genre_never_required(self):
        c = analyze_concepts('romantic comedies', excluded_genres=['Romance'])
        self.assertEqual(c['genre_groups'], [['Comedy']])

    def test_covered_words(self):
        self.assertTrue({'heist'} <= analyze_concepts('heist movies')['covered'])
        self.assertFalse({'lighthouse'} & analyze_concepts('movies set in a lighthouse')['covered'])


class TestIntent(unittest.TestCase):

    def test_concept_prompt_has_nothing_uncovered(self):
        i = intent_for('christmas movies')
        self.assertEqual(i['uncovered'], [])
        self.assertTrue(i['framed'])

    def test_title_prompt_keeps_title_words(self):
        self.assertEqual(intent_for('The Grand Budapest Hotel')['uncovered'], ['budapest', 'hotel'])

    def test_bare_concept_word_is_not_framed(self):
        i = intent_for('Alien')
        self.assertEqual(i['uncovered'], [])
        self.assertFalse(i['framed'])

    def test_documentaries_avoided_unless_requested(self):
        self.assertIn('Documentary', intent_for('war movies')['avoid_genres'])
        self.assertNotIn('Documentary', intent_for('war documentaries')['avoid_genres'])


class TestRelevance(unittest.TestCase):

    def test_missing_required_genre_fails(self):
        i = intent_for('dark sci-fi')
        rel, ok = recommender._relevance({'Thriller', 'Crime'}, 'a dark crime story', i)
        self.assertFalse(ok)
        rel2, ok2 = recommender._relevance({'Science Fiction', 'Thriller'}, 'a dark future', i)
        self.assertTrue(ok2)
        self.assertGreater(rel2, rel)

    def test_core_tag_beats_supporting_tag(self):
        i = intent_for('christmas movies')
        i['tag_ids'], i['core_tag_ids'] = ['1', '2'], ['1']
        core, _ = recommender._relevance({'Comedy'}, '', i, tag_hit=2)
        supporting, _ = recommender._relevance({'Comedy'}, '', i, tag_hit=1)
        self.assertGreater(core, supporting)

    def test_avoided_genre_lowers_relevance(self):
        i = intent_for('feel good movies')
        happy, _ = recommender._relevance({'Comedy', 'Family'}, '', i)
        grim, _ = recommender._relevance({'Horror'}, '', i)
        self.assertGreater(happy, grim)

    def test_quality_prior_distrusts_few_votes(self):
        few = recommender._quality({'vote_average': 9.5, 'vote_count': 8})
        many = recommender._quality({'vote_average': 8.0, 'vote_count': 20000})
        self.assertGreater(many, few)


class TestPersonDetection(unittest.TestCase):

    def test_common_words_are_not_people(self):
        from backend.query_parser import _detect_person
        for prompt in ['jurassic park vibes', 'fallen angels type movie', 'stoner comedy',
                       'a bleak hardy outsider drama', 'haunted house movies', 'heist movies',
                       'korean revenge thrillers', 'No Time to Die', 'The Matrix', 'movies with nudity']:
            self.assertIsNone(_detect_person(prompt), prompt)

    def test_surname_needs_person_position(self):
        from backend.query_parser import _detect_person
        self.assertEqual(_detect_person('nolan movies about time travel')['name'], 'Christopher Nolan')
        self.assertEqual(_detect_person("nolan's films")['name'], 'Christopher Nolan')
        self.assertIsNone(_detect_person('a nolan-esque puzzle box'))

    def test_unknown_names_after_cues_with_role(self):
        from backend.query_parser import _detect_person
        found = _detect_person('movies starring anya taylor-joy')
        self.assertEqual((found['name'], found['role']), ('Anya Taylor-joy', 'actor'))
        found = _detect_person('directed by sean baker')
        self.assertEqual((found['name'], found['role']), ('Sean Baker', 'director'))
        self.assertEqual(_detect_person('david lynch movies')['name'], 'David Lynch')

    def test_typo_is_stripped_from_prompt(self):
        a = interpret_query('scarlett johanson movies that are critically acclaimed')
        self.assertEqual(a['person'], 'Scarlett Johansson')
        self.assertEqual(recommender._build_intent(a, a['positive_query'], None)['uncovered'], [])

    def test_role_selects_credits(self):
        data = {'directed_ids': {1}, 'cast_ids': {2}, 'signature_ids': {1, 2}}
        self.assertEqual(recommender._person_ids_for_role(data, 'director'), {1})
        self.assertEqual(recommender._person_ids_for_role(data, 'actor'), {2})
        self.assertEqual(recommender._person_ids_for_role(data, None), {1, 2})


class TestPersonValidation(unittest.TestCase):

    def test_name_matching(self):
        self.assertTrue(recommender._person_name_matches('Christopher Nolan', 'Christopher Nolan'))
        self.assertTrue(recommender._person_name_matches('Nolan', 'Christopher Nolan'))
        self.assertTrue(recommender._person_name_matches('Bong Joon-ho', 'Bong Joon Ho'))
        self.assertFalse(recommender._person_name_matches('Blade Runner', 'Rutger Hauer'))
        self.assertFalse(recommender._person_name_matches('Fight Club', 'Fightclub Jones'))


class TestExactTagMatching(unittest.TestCase):

    def test_only_exact_names_accepted(self):
        recommender._exact_tag_cache.clear()
        fake = {'results': [{'id': 1, 'name': 'dark'}, {'id': 2, 'name': 'dark comedy'},
                            {'id': 3, 'name': 'tokyo, japan'}]}
        with patch.object(recommender.http_session, 'get') as get:
            get.return_value.json.return_value = fake
            self.assertEqual(recommender._get_exact_keyword_ids('dark', 'KEY'), ['1'])
            # Location tags carry a country suffix
            self.assertEqual(recommender._get_exact_keyword_ids('tokyo', 'KEY'), ['3'])
        recommender._exact_tag_cache.clear()


if __name__ == '__main__':
    unittest.main()
