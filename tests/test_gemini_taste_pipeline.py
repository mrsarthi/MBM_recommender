import os
import sys
import unittest
import json
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import (
    get_or_create_user, upsert_movies_batch, upsert_user_diary,
    get_user_taste_anchors, upsert_user_watchlist
)
from backend.query_parser import (
    interpret_query_with_ai, generate_matchmaker_pitch, _fallback_mood_match
)
from backend.recommender import analyze
from backend.watchlist import pick_movie_for_tonight
from backend.in_memory_model import train_user_model_in_memory

import time

class TestGeminiTastePipeline(unittest.TestCase):

    def setUp(self):
        self.username = f"test_taste_cinephile_{int(time.time() * 1000)}"
        self.user = get_or_create_user(self.username)
        
        # Insert sample movies with directors, genres, and years
        sample_movies = [
            {
                'movie_id': 80001, 'title': 'Blade Runner 2049', 'year': '2017',
                'genres': 'Science Fiction, Mystery, Thriller',
                'overview': 'A young blade runner discovers a long-buried secret.',
                'director': 'Denis Villeneuve', 'runtime': 164, 'vote_average': 8.0
            },
            {
                'movie_id': 80002, 'title': 'Arrival', 'year': '2016',
                'genres': 'Science Fiction, Drama, Mystery',
                'overview': 'Linguist works with the military to communicate with alien lifeforms.',
                'director': 'Denis Villeneuve', 'runtime': 116, 'vote_average': 7.9
            },
            {
                'movie_id': 80003, 'title': 'Se7en', 'year': '1995',
                'genres': 'Crime, Mystery, Thriller',
                'overview': 'Two detectives hunt a serial killer based on seven deadly sins.',
                'director': 'David Fincher', 'runtime': 127, 'vote_average': 8.3
            },
            {
                'movie_id': 80004, 'title': 'Fight Club', 'year': '1999',
                'genres': 'Drama, Thriller',
                'overview': 'An insomniac office worker forms an underground fight club.',
                'director': 'David Fincher', 'runtime': 139, 'vote_average': 8.4
            },
            {
                'movie_id': 80005, 'title': 'Generic Slasher', 'year': '2005',
                'genres': 'Horror',
                'overview': 'Teens in the woods.',
                'director': 'Unknown Director', 'runtime': 90, 'vote_average': 5.0
            }
        ]
        upsert_movies_batch(sample_movies)

        # Log diary entries
        upsert_user_diary(self.user['id'], [
            {'movie_id': 80001, 'rating': 5.0, 'watched_date': '2026-08-10'},
            {'movie_id': 80002, 'rating': 4.5, 'watched_date': '2026-08-11'},
            {'movie_id': 80003, 'rating': 5.0, 'watched_date': '2026-08-12'},
            {'movie_id': 80004, 'rating': 4.5, 'watched_date': '2026-08-13'},
            {'movie_id': 80005, 'rating': 1.5, 'watched_date': '2026-08-14'}
        ])

    def test_01_get_user_taste_anchors(self):
        """Test extraction of top directors, 5★ favorites, and preferred genres."""
        anchors = get_user_taste_anchors(self.username)
        self.assertIsNotNone(anchors)
        self.assertIn('top_directors', anchors)
        self.assertIn('favorite_movies', anchors)
        self.assertIn('top_genres', anchors)
        self.assertIn('preferred_decades', anchors)

        # Denis Villeneuve and David Fincher should be top directors
        self.assertTrue(any('Denis Villeneuve' in d for d in anchors['top_directors']))
        self.assertTrue(any('David Fincher' in d for d in anchors['top_directors']))
        
        # 5-star favorites should include Blade Runner 2049 and Se7en
        self.assertIn('Blade Runner 2049', anchors['favorite_movies'])
        self.assertIn('Se7en', anchors['favorite_movies'])
        
        # Sci-Fi / Thriller / Mystery should be top genres
        self.assertTrue(any(g in anchors['top_genres'] for g in ['Science Fiction', 'Mystery', 'Thriller', 'Drama']))

    def test_02_interpret_query_with_taste_context_mock(self):
        """Test Gemini query parsing with taste context and structured JSON."""
        anchors = get_user_taste_anchors(self.username)
        
        mock_response = {
            "genres": ["Science Fiction", "Thriller"],
            "search_query": "atmospheric neo noir",
            "suggested_titles": [
                {
                    "title": "Gattaca",
                    "year": "1997",
                    "vibe_pitch": "Matches your affinity for smart, atmospheric 90s dystopian sci-fi."
                },
                {
                    "title": "Dark City",
                    "year": "1998",
                    "vibe_pitch": "Dark neo-noir aesthetic that aligns with your love for mystery and Se7en."
                }
            ]
        }

        mock_client = Mock()
        mock_resp = Mock()
        mock_resp.text = json.dumps(mock_response)
        mock_client.models.generate_content.return_value = mock_resp

        with patch('backend.query_parser._get_genai_client', return_value=mock_client):
            res = interpret_query_with_ai(
                "atmospheric rainy neo noir sci-fi",
                custom_api_key="TEST_KEY",
                taste_context=anchors
            )

            self.assertIn('genres', res)
            self.assertEqual(res['genres'], ["Science Fiction", "Thriller"])
            self.assertEqual(len(res['suggested_titles']), 2)
            self.assertEqual(res['suggested_titles'][0]['title'], "Gattaca")
            self.assertEqual(res['suggested_titles'][0]['year'], "1997")
            self.assertTrue("dystopian" in res['suggested_titles'][0]['vibe_pitch'])

    def test_03_recommender_deduplication_and_ranking(self):
        """Test that already-watched movies are filtered out and remaining candidates are ranked by AI score."""
        ai_model, ai_cols, ai_vec, ai_enc = train_user_model_in_memory(self.username)

        # Watched set contains movie_id 80001 (Blade Runner 2049) and title 'se7en'
        watched_titles = {'bladerunner2049', 'se7en', 'arrival', 'fightclub', 'genericslasher'}
        watched_ids = {80001, 80002, 80003, 80004, 80005}

        ai_analysis = {
            'genres': ['Science Fiction', 'Thriller'],
            'search_query': 'neo noir',
            'suggested_titles': [
                {'title': 'Blade Runner 2049', 'year': '2017', 'vibe_pitch': 'Already watched'},
                {'title': 'Gattaca', 'year': '1997', 'vibe_pitch': 'Atmospheric sci-fi mystery'},
                {'title': 'Dark City', 'year': '1998', 'vibe_pitch': 'Noir thriller with great mood'}
            ]
        }

        # Mock TMDB movie lookups
        def mock_http_get(url, params=None, timeout=None):
            resp = Mock()
            resp.status_code = 200
            q = (params or {}).get('query', '')
            if 'Gattaca' in q:
                resp.json.return_value = {
                    'results': [{
                        'id': 782, 'title': 'Gattaca', 'genre_ids': [878, 53, 18],
                        'release_date': '1997-10-24', 'vote_average': 7.6,
                        'overview': 'In a future society, genetically engineered individuals rule.',
                        'director': 'Andrew Niccol'
                    }]
                }
            elif 'Dark City' in q:
                resp.json.return_value = {
                    'results': [{
                        'id': 2666, 'title': 'Dark City', 'genre_ids': [878, 9648],
                        'release_date': '1998-02-27', 'vote_average': 7.3,
                        'overview': 'A man struggles with memories in a world that never sees sunlight.',
                        'director': 'Alex Proyas'
                    }]
                }
            elif 'Blade Runner' in q:
                resp.json.return_value = {
                    'results': [{
                        'id': 80001, 'title': 'Blade Runner 2049', 'genre_ids': [878, 53],
                        'release_date': '2017-10-06', 'vote_average': 8.0
                    }]
                }
            else:
                resp.json.return_value = {'results': []}
            return resp

        with patch('backend.recommender.http_session.get', side_effect=mock_http_get):
            picks = analyze(
                watched_titles, watched_ids, ['genericslasher'],
                ai_analysis, ai_model, ai_cols, ai_vec, ai_enc,
                user_context='Alone', streaming_filter='All Platforms',
                raw_prompt='atmospheric neo noir', source='all', username=self.username,
                tmdb_key='DUMMY_TMDB'
            )

            # Blade Runner 2049 must NOT be in discovery picks because it was watched
            pick_ids = [p.get('id') for p in picks]
            self.assertNotIn(80001, pick_ids)
            
            # Gattaca and Dark City should be present
            self.assertTrue(782 in pick_ids or 2666 in pick_ids)

            # Check that vibe_pitch was preserved
            for p in picks:
                if p.get('id') == 782:
                    self.assertEqual(p.get('vibe_pitch'), 'Atmospheric sci-fi mystery')

            # Verify picks are sorted descending by predicted ai_score
            for i in range(len(picks) - 1):
                self.assertGreaterEqual(picks[i].get('ai_score', 0), picks[i+1].get('ai_score', 0))

    def test_04_generate_matchmaker_pitch(self):
        """Test AI pitch generation for watchlist matchmaker."""
        anchors = get_user_taste_anchors(self.username)
        sample_winner = {
            'movie_id': 782, 'title': 'Gattaca', 'year': '1997', 'runtime': 106,
            'genres': ['Science Fiction', 'Thriller'], 'ai_score': 4.5,
            'overview': 'A genetically imperfect man dreams of traveling to the stars.'
        }

        mock_client = Mock()
        mock_resp = Mock()
        mock_resp.text = '{"pitch": "Since you loved Blade Runner 2049, Gattaca gives you that same smart, elegant dystopian vision in just 106 minutes."}'
        mock_client.models.generate_content.return_value = mock_resp

        with patch('backend.query_parser._get_genai_client', return_value=mock_client):
            pitch = generate_matchmaker_pitch(
                sample_winner,
                user_taste=anchors,
                duration_pref='< 2 hours',
                mood_pref='Mind-Bending',
                custom_api_key='TEST_KEY'
            )

            self.assertIn("Blade Runner 2049", pitch)
            self.assertIn("Gattaca", pitch)

    def test_05_matchmaker_fallback_without_key(self):
        """Test fallback pitch when API key is not provided."""
        sample_winner = {
            'movie_id': 782, 'title': 'Gattaca', 'year': '1997', 'runtime': 106,
            'genres': ['Science Fiction', 'Thriller'], 'ai_score': 4.5,
            'clusters': ['Mind-Bending', 'Quick Watch']
        }
        with patch('backend.query_parser.GEMINI_API_KEY', None):
            pitch = generate_matchmaker_pitch(
                sample_winner,
                user_taste=None,
                duration_pref='< 2 hours',
                mood_pref='Mind-Bending',
                custom_api_key=None
            )
            self.assertIn("affinity score", pitch)
            self.assertIn("106 min", pitch)

    def test_06_watchlist_search_differentiation(self):
        """Test that distinct queries on watchlist return distinct, relevant rankings."""
        ai_model, ai_cols, ai_vec, ai_enc = train_user_model_in_memory(self.username)

        # Setup diverse watchlist movies
        wl_sample = [
            {
                'movie_id': 90001, 'title': 'Eraserhead', 'year': '1977',
                'genres': 'Horror, Fantasy',
                'overview': 'Henry Spencer tries to survive his industrial environment and a bizarre mutant child.',
                'director': 'David Lynch', 'runtime': 89, 'vote_count': 1800, 'vote_average': 7.4
            },
            {
                'movie_id': 90002, 'title': 'The Conjuring', 'year': '2013',
                'genres': 'Horror, Thriller',
                'overview': 'Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence.',
                'director': 'James Wan', 'runtime': 112, 'vote_count': 11000, 'vote_average': 7.5
            },
            {
                'movie_id': 90003, 'title': 'Everything Everywhere All at Once', 'year': '2022',
                'genres': 'Action, Adventure, Science Fiction',
                'overview': 'An aging Chinese immigrant is swept up in an insane, bizarre, weird multiverse adventure.',
                'director': 'Daniel Kwan', 'runtime': 139, 'vote_count': 6000, 'vote_average': 7.8
            },
            {
                'movie_id': 90004, 'title': 'Before Sunrise', 'year': '1995',
                'genres': 'Drama, Romance',
                'overview': 'A young man and woman meet on a train in Europe and spend one evening together in Vienna.',
                'director': 'Richard Linklater', 'runtime': 101, 'vote_count': 4200, 'vote_average': 8.2
            },
            {
                'movie_id': 90005, 'title': 'Avengers: Endgame', 'year': '2019',
                'genres': 'Action, Science Fiction',
                'overview': 'The grave course of events set in motion by Thanos that wiped out half the universe.',
                'director': 'Anthony Russo', 'runtime': 181, 'vote_count': 25000, 'vote_average': 8.3
            }
        ]
        upsert_movies_batch(wl_sample)
        upsert_user_watchlist(self.user['id'], [m['movie_id'] for m in wl_sample])

        # Query 1: Some weird films
        picks_weird = analyze(
            set(), set(), set(),
            {'genres': ['Fantasy', 'Science Fiction'], 'search_query': 'surreal weird', 'suggested_titles': []},
            ai_model, ai_cols, ai_vec, ai_enc,
            source='watchlist', username=self.username, raw_prompt='Some weird films'
        )
        top_weird_ids = [p['id'] for p in picks_weird[:2]]
        # Eraserhead or EEAAO must be in top weird
        self.assertTrue(90001 in top_weird_ids or 90003 in top_weird_ids)

        # Query 2: Some weird horror films
        picks_weird_horror = analyze(
            set(), set(), set(),
            {'genres': ['Horror'], 'search_query': 'weird surreal horror', 'suggested_titles': []},
            ai_model, ai_cols, ai_vec, ai_enc,
            source='watchlist', username=self.username, raw_prompt='Some weird horror films'
        )
        # Eraserhead (Horror + Weird) must be #1
        self.assertEqual(picks_weird_horror[0]['id'], 90001)

        # Query 3: horror
        picks_horror = analyze(
            set(), set(), set(),
            {'genres': ['Horror'], 'search_query': 'horror', 'suggested_titles': []},
            ai_model, ai_cols, ai_vec, ai_enc,
            source='watchlist', username=self.username, raw_prompt='horror'
        )
        horror_ids = {p['id'] for p in picks_horror}
        # Non-horror films like Before Sunrise and Avengers must NOT be in horror top picks
        self.assertNotIn(90004, [p['id'] for p in picks_horror[:2]])
        self.assertNotIn(90005, [p['id'] for p in picks_horror[:2]])

        # Verify that all 4 queries produce DIFFERENT top recommendations
        self.assertNotEqual([p['id'] for p in picks_weird], [p['id'] for p in picks_weird_horror])

if __name__ == '__main__':
    unittest.main()

