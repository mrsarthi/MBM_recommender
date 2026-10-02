import os
import sys
import unittest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.watchlist import get_mood_cluster

class TestWatchlist(unittest.TestCase):
    def test_04_mood_cluster_classification(self):
        c1 = get_mood_cluster('Science Fiction, Thriller', 140)
        self.assertIn('Mind-Bending', c1)

        c2 = get_mood_cluster('Comedy, Family', 90)
        self.assertIn('Comfort', c2)
        self.assertIn('Quick Watch', c2)

if __name__ == '__main__':
    unittest.main()
