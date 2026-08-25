import os
import sys
import unittest
import tempfile
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.watchlist import add_to_watchlist, remove_from_watchlist, load_watchlist, pick_movie_for_tonight, get_mood_cluster

class TestWatchlist(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_wl = os.path.join(self.test_dir, 'test_watchlist.csv')

    def test_01_add_and_load_watchlist(self):
        movie1 = {
            'id': 27205,
            'title': 'Inception',
            'genres': 'Action, Science Fiction, Mystery',
            'overview': 'A thief who steals corporate secrets through dream-sharing technology.',
            'year': '2010',
            'runtime': 148,
            'vote_average': 8.4
        }
        ok, msg = add_to_watchlist(movie1, self.sample_wl)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.sample_wl))

        items = load_watchlist(self.sample_wl)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['title'], 'Inception')
        self.assertIn('Mind-Bending', items[0]['clusters'])

    def test_02_remove_from_watchlist(self):
        movie1 = {'id': 27205, 'title': 'Inception', 'year': '2010', 'runtime': 148}
        add_to_watchlist(movie1, self.sample_wl)
        
        ok, msg = remove_from_watchlist(27205, self.sample_wl)
        self.assertTrue(ok)
        items = load_watchlist(self.sample_wl)
        self.assertEqual(len(items), 0)

    def test_03_pick_for_tonight(self):
        movie1 = {'id': 1, 'title': 'Short Noir', 'genres': 'Crime, Drama', 'runtime': 95, 'vote_average': 8.0}
        movie2 = {'id': 2, 'title': 'Sci-Fi Epic', 'genres': 'Science Fiction, Mystery', 'runtime': 160, 'vote_average': 8.5}
        add_to_watchlist(movie1, self.sample_wl)
        add_to_watchlist(movie2, self.sample_wl)

        # Test quick duration filter
        winner, msg = pick_movie_for_tonight({'duration': '< 100 mins', 'mood': 'Any'}, None, None, None, None, watchlist_path=self.sample_wl)
        self.assertIsNotNone(winner)
        self.assertIn('pitch', winner)

    def test_04_mood_cluster_classification(self):
        c1 = get_mood_cluster('Science Fiction, Thriller', 140)
        self.assertIn('Mind-Bending', c1)

        c2 = get_mood_cluster('Comedy, Family', 90)
        self.assertIn('Comfort', c2)
        self.assertIn('Quick Watch', c2)

if __name__ == '__main__':
    unittest.main()
