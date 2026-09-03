import time
import threading
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend.db import get_connection, release_connection
from backend.logger import logger

class CollaborativeEngine:
    """
    User-User Collaborative Filtering Recommender Engine.
    Leverages cross-user rating history from Neon DB user_diary to find taste twins
    and generate collaborative score predictions for candidate movies.
    """
    def __init__(self, cache_ttl_seconds=3600):
        self._cache_ttl = cache_ttl_seconds
        self._last_trained = 0.0
        self._user_item_matrix = None  # DataFrame with user_id index and movie_id columns
        self._user_sim_matrix = None   # DataFrame with user_id index and user_id columns
        self._user_mean_ratings = {}   # dict of user_id -> mean rating
        self._lock = threading.Lock()

    def train(self, force=False):
        """
        Loads all user ratings from user_diary and builds the user similarity matrix.
        Caches the matrix in memory for ultra-fast lookups (< 1ms).
        """
        now = time.time()
        with self._lock:
            if not force and (now - self._last_trained < self._cache_ttl) and self._user_sim_matrix is not None:
                return

            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_id, movie_id, rating
                        FROM user_diary
                        WHERE rating IS NOT NULL AND rating > 0
                    """)
                    rows = cur.fetchall()
            except Exception as e:
                logger.warning(f"[CollaborativeEngine] Failed to load user_diary: {e}")
                return
            finally:
                release_connection(conn)

            if not rows or len(rows) < 10:
                self._user_item_matrix = None
                self._user_sim_matrix = None
                self._last_trained = now
                return

            df = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating'])
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])

            # Minimum ratings threshold per user to be included in similarity calculation
            user_counts = df['user_id'].value_counts()
            active_users = user_counts[user_counts >= 3].index
            if len(active_users) < 2:
                # Fallback to all users if few have >= 3 ratings
                active_users = user_counts.index

            df_filtered = df[df['user_id'].isin(active_users)]
            if df_filtered.empty:
                return

            # Compute mean rating per user for mean-centering (handles harsh vs generous raters)
            self._user_mean_ratings = df_filtered.groupby('user_id')['rating'].mean().to_dict()

            # Build User-Item Pivot Table
            user_item = df_filtered.pivot_table(index='user_id', columns='movie_id', values='rating')
            
            # Mean-center the ratings (subtract user mean, fill unrated with 0)
            user_item_centered = user_item.sub(user_item.mean(axis=1), axis=0).fillna(0)

            # Compute Cosine Similarity between users
            sim_scores = cosine_similarity(user_item_centered)
            sim_df = pd.DataFrame(sim_scores, index=user_item.index, columns=user_item.index)

            self._user_item_matrix = user_item
            self._user_sim_matrix = sim_df
            self._last_trained = now
            logger.info(f"[CollaborativeEngine] Trained CF matrix across {len(user_item)} users and {user_item.shape[1]} movies.")

    def get_collaborative_predictions(self, target_user_id: int, movie_ids: list, top_k_neighbors: int = 15):
        """
        Calculates predicted ratings for candidate movies based on taste-similar users.
        Returns dict: {movie_id: predicted_rating (float)}
        """
        if not target_user_id or not movie_ids:
            return {}

        self.train()

        with self._lock:
            if self._user_sim_matrix is None or self._user_item_matrix is None:
                return {}
            if target_user_id not in self._user_sim_matrix.index:
                return {}

            user_sims = self._user_sim_matrix.loc[target_user_id].drop(target_user_id, errors='ignore')
            # Select neighbors with positive taste correlation
            positive_neighbors = user_sims[user_sims > 0.05].sort_values(ascending=False).head(top_k_neighbors)

            if positive_neighbors.empty:
                return {}

            neighbor_ids = positive_neighbors.index
            neighbor_sim_weights = positive_neighbors.values

            predictions = {}
            target_user_mean = self._user_mean_ratings.get(target_user_id, 3.5)

            for mid in movie_ids:
                try:
                    mid_int = int(mid)
                except (ValueError, TypeError):
                    continue

                if mid_int not in self._user_item_matrix.columns:
                    continue

                # Get ratings given to this movie by the top similar neighbors
                neighbor_ratings = self._user_item_matrix.loc[neighbor_ids, mid_int]
                valid_mask = neighbor_ratings.notna()

                if not valid_mask.any():
                    continue

                valid_sims = neighbor_sim_weights[valid_mask]
                valid_rats = neighbor_ratings[valid_mask].values

                # Weighted rating deviation formula
                sim_sum = np.sum(np.abs(valid_sims))
                if sim_sum > 0:
                    neighbor_means = np.array([self._user_mean_ratings.get(nid, 3.5) for nid in neighbor_ids[valid_mask]])
                    rating_diffs = valid_rats - neighbor_means
                    predicted_rating = target_user_mean + (np.dot(valid_sims, rating_diffs) / sim_sum)
                    # Bound rating between 0.5 and 5.0
                    bounded_pred = round(float(np.clip(predicted_rating, 0.5, 5.0)), 2)
                    predictions[mid_int] = bounded_pred

            return predictions

# Global singleton instance
collaborative_engine = CollaborativeEngine()
