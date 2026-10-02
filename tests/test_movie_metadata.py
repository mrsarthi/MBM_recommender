"""
Stored film metadata used by watchlist search: original_language and TMDB tags.
Uses the configured database, with throwaway movie ids that are removed afterwards.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import upsert_movies_batch, fill_movie_metadata, get_connection, release_connection

TEST_IDS = (999990001, 999990002)


def stored(movie_id):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT original_language, keywords FROM movies WHERE movie_id = %s", (movie_id,))
            return cur.fetchone()
    finally:
        release_connection(conn)


class TestMovieMetadata(unittest.TestCase):

    def tearDown(self):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM movies WHERE movie_id = ANY(%s)", (list(TEST_IDS),))
                conn.commit()
        finally:
            release_connection(conn)

    def test_upsert_stores_language_and_keeps_it(self):
        upsert_movies_batch([{'movie_id': TEST_IDS[0], 'title': 'Lang Test', 'original_language': 'ko'}])
        self.assertEqual(stored(TEST_IDS[0])[0], 'ko')
        # A later upsert without a language (e.g. a Letterboxd sync) must not erase it
        upsert_movies_batch([{'movie_id': TEST_IDS[0], 'title': 'Lang Test'}])
        self.assertEqual(stored(TEST_IDS[0])[0], 'ko')

    def test_backfill_fills_gaps_only(self):
        upsert_movies_batch([
            {'movie_id': TEST_IDS[0], 'title': 'Empty'},
            {'movie_id': TEST_IDS[1], 'title': 'Known', 'original_language': 'fr', 'keywords': 'heist'},
        ])
        fill_movie_metadata([(TEST_IDS[0], 'ja', 'anime, robot'), (TEST_IDS[1], 'en', 'other')])
        self.assertEqual(stored(TEST_IDS[0]), ('ja', 'anime, robot'))
        self.assertEqual(stored(TEST_IDS[1]), ('fr', 'heist'))


if __name__ == '__main__':
    unittest.main()
