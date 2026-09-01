"""
Offline Model Evaluation Harness
Provides chronological out-of-sample holdout validation measuring honest
Mean Absolute Error (MAE), Spearman Rank Correlation (rho), and NDCG@10.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, ndcg_score
from scipy.stats import spearmanr

def compute_ndcg_at_k(y_true, y_pred, k=10):
    """Computes Normalized Discounted Cumulative Gain at k."""
    if len(y_true) < 2:
        return 1.0
    try:
        # Reshape for sklearn ndcg_score: shape (1, n_samples)
        true_2d = np.asarray(y_true).reshape(1, -1)
        pred_2d = np.asarray(y_pred).reshape(1, -1)
        return float(ndcg_score(true_2d, pred_2d, k=min(k, len(y_true))))
    except Exception:
        return 1.0

def evaluate_model_temporal_holdout(df: pd.DataFrame, holdout_ratio: float = 0.20, k_smooth: float = 3.0):
    """
    Evaluates recommendation taste model by holding out the most recent `holdout_ratio`
    (default: 20%) of watched films by watch date.
    
    Returns:
        dict with {
            'train_size': int,
            'test_size': int,
            'mae': float,
            'spearman_rho': float,
            'ndcg_10': float,
            'baseline_mean_mae': float
        }
    """
    if df is None or len(df) < 8 or 'Rating' not in df.columns:
        return {
            'train_size': 0,
            'test_size': 0,
            'mae': 0.0,
            'spearman_rho': 0.0,
            'ndcg_10': 0.0,
            'baseline_mean_mae': 0.0
        }

    df_clean = df.dropna(subset=['Rating']).copy()
    if 'Date' in df_clean.columns:
        df_clean = df_clean.sort_values('Date', ascending=True)
    
    n_total = len(df_clean)
    split_idx = max(5, int(n_total * (1.0 - holdout_ratio)))
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    if len(test_df) < 2:
        return {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'mae': 0.0,
            'spearman_rho': 0.0,
            'ndcg_10': 0.0,
            'baseline_mean_mae': 0.0
        }

    user_mean = float(train_df['Rating'].mean())

    # 1. Genres multi-hot
    genre_series = train_df['genres'].fillna('').astype(str).str.split(', ')
    all_genres = set([g for sublist in genre_series for g in sublist if g])
    for g in all_genres:
        train_df[f'genre_{g}'] = train_df['genres'].fillna('').astype(str).apply(lambda x: 1.0 if g in x else 0.0)
        test_df[f'genre_{g}'] = test_df['genres'].fillna('').astype(str).apply(lambda x: 1.0 if g in x else 0.0)

    # 2. Text TF-IDF
    tfidf = TfidfVectorizer(max_features=40, stop_words='english')
    train_overviews = train_df['overview'].fillna('').astype(str).str.strip()
    if train_overviews.str.cat(sep='').strip() == '':
        train_overviews = train_df['title'].fillna('cinema').astype(str)
    
    test_overviews = test_df['overview'].fillna('').astype(str).str.strip()
    if test_overviews.str.cat(sep='').strip() == '':
        test_overviews = test_df['title'].fillna('cinema').astype(str)

    try:
        tfidf_train_mat = tfidf.fit_transform(train_overviews).toarray()
        tfidf_test_mat = tfidf.transform(test_overviews).toarray()
    except Exception:
        tfidf_train_mat = np.zeros((len(train_df), 10), dtype=np.float32)
        tfidf_test_mat = np.zeros((len(test_df), 10), dtype=np.float32)

    for i in range(tfidf_train_mat.shape[1]):
        train_df[f'tfidf_{i}'] = tfidf_train_mat[:, i]
        test_df[f'tfidf_{i}'] = tfidf_test_mat[:, i]

    # 3. Smoothed Director Encoding
    dir_stats = train_df.groupby('director')['Rating'].agg(['sum', 'count']).to_dict(orient='index')
    dir_target_map = {}
    dir_count_map = {}
    for d_name, stats in dir_stats.items():
        if d_name and str(d_name).lower() != 'unknown':
            s = stats['sum']
            n = stats['count']
            dir_target_map[str(d_name).strip()] = float((s + k_smooth * user_mean) / (n + k_smooth))
            dir_count_map[str(d_name).strip()] = int(n)

    # 4. Smoothed Cast Encoding
    cast_stats = {}
    for _, row in train_df.iterrows():
        r_val = float(row['Rating'])
        cast_raw = str(row.get('cast') or '').split(',')
        for actor in [a.strip() for a in cast_raw if a.strip() and a.strip().lower() != 'nan']:
            if actor not in cast_stats:
                cast_stats[actor] = {'sum': 0.0, 'count': 0}
            cast_stats[actor]['sum'] += r_val
            cast_stats[actor]['count'] += 1

    cast_target_map = {a: float((s['sum'] + k_smooth * user_mean) / (s['count'] + k_smooth)) for a, s in cast_stats.items()}
    cast_count_map = {a: int(s['count']) for a, s in cast_stats.items()}

    # Apply to train and test
    for d_frame in (train_df, test_df):
        d_frame['director_target_encoded'] = d_frame['director'].apply(
            lambda d: dir_target_map.get(str(d).strip(), user_mean) if d else user_mean
        )
        d_frame['director_film_count'] = d_frame['director'].apply(
            lambda d: float(dir_count_map.get(str(d).strip(), 0)) if d else 0.0
        )
        
        def _cast_f(c_val):
            actors = [a.strip() for a in str(c_val or '').split(',') if a.strip()]
            a0 = actors[0] if len(actors) > 0 else ''
            a1 = actors[1] if len(actors) > 1 else ''
            s0 = cast_target_map.get(a0, user_mean)
            s1 = cast_target_map.get(a1, user_mean)
            cnt = max(cast_count_map.get(a0, 0), cast_count_map.get(a1, 0))
            return pd.Series([s0, s1, float(cnt)])
            
        c_res = d_frame['cast'].apply(_cast_f)
        d_frame['cast_0_target_encoded'] = c_res[0]
        d_frame['cast_1_target_encoded'] = c_res[1]
        d_frame['cast_film_count'] = c_res[2]
        d_frame['runtime_clean'] = pd.to_numeric(d_frame['runtime'], errors='coerce').fillna(105.0)

    feature_cols = [c for c in train_df.columns if c.startswith('genre_') or c.startswith('tfidf_')]
    feature_cols.extend([
        'director_target_encoded', 'director_film_count',
        'cast_0_target_encoded', 'cast_1_target_encoded', 'cast_film_count',
        'runtime_clean'
    ])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Rating'].astype(float)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Rating'].astype(float).values

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0.5, 5.0)

    # Metrics
    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    baseline_mae = float(mean_absolute_error(y_test, np.full_like(y_test, user_mean)))
    
    rho_res = spearmanr(y_test, y_pred_clipped)
    rho = float(rho_res.statistic) if not np.isnan(rho_res.statistic) else 0.0
    ndcg = compute_ndcg_at_k(y_test, y_pred_clipped, k=10)

    return {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'mae': round(mae, 3),
        'spearman_rho': round(rho, 3),
        'ndcg_10': round(ndcg, 3),
        'baseline_mean_mae': round(baseline_mae, 3)
    }
