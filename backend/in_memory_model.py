import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from backend.db import get_diary_training_df

# In-memory model cache: username -> (model, columns, vectorizer, encoders, timestamp)
_user_models = {}

def train_user_model_in_memory(username: str):
    """
    Trains a personalized AI taste model in RAM directly from database rows.
    Completes in < 150ms without writing any .pkl files to disk.
    """
    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user or clean_user == 'guest':
        return None, [], None, {}

    df = get_diary_training_df(clean_user)
    if df.empty or len(df) < 5 or 'Rating' not in df.columns:
        return None, [], None, {}

    df = df.copy()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').astype(float)
    df = df.dropna(subset=['Rating'])
    if len(df) < 5:
        return None, [], None, {}

    # 1. Genres multi-hot encoding
    genre_series = df['genres'].fillna('').astype(str).str.split(', ')
    all_genres = set([g for sublist in genre_series for g in sublist if g])
    for g in all_genres:
        df[f'genre_{g}'] = df['genres'].fillna('').astype(str).apply(lambda x: 1.0 if g in x else 0.0)

    # 2. Text TF-IDF on overview (with title & genre fallback if empty)
    tfidf = TfidfVectorizer(max_features=40, stop_words='english')
    overviews = df['overview'].fillna('').astype(str).str.strip()
    if overviews.str.cat(sep='').strip() == '':
        overviews = df['title'].fillna('cinema film').astype(str) + ' ' + df['genres'].fillna('movie').astype(str)
    
    try:
        tfidf_mat = tfidf.fit_transform(overviews).toarray()
    except Exception:
        tfidf = TfidfVectorizer(max_features=40)
        tfidf_mat = tfidf.fit_transform(['cinema film motion picture' for _ in range(len(df))]).toarray()

    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_mat.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_mat, columns=tfidf_cols, index=df.index)
    df = pd.concat([df, tfidf_df], axis=1)

    # 3. Smoothed Bayesian Target Encoding for Directors & Top-2 Cast
    user_mean = float(df['Rating'].mean()) if len(df) > 0 else 3.5
    user_std = float(df['Rating'].std()) if len(df) > 1 and df['Rating'].std() > 0 else 1.0
    k_smooth = 3.0  # Smoothing weight

    # Director target map
    dir_stats = df.groupby('director')['Rating'].agg(['sum', 'count']).to_dict(orient='index')
    dir_target_map = {}
    dir_count_map = {}
    for d_name, stats in dir_stats.items():
        if d_name and str(d_name).lower() != 'unknown' and str(d_name).lower() != 'nan':
            s = stats['sum']
            n = stats['count']
            dir_target_map[str(d_name).strip()] = float((s + k_smooth * user_mean) / (n + k_smooth))
            dir_count_map[str(d_name).strip()] = int(n)

    # Cast target map
    cast_stats = {}
    for _, row in df.iterrows():
        r_val = float(row['Rating'])
        cast_raw = str(row.get('cast') or '').split(',')
        for actor in [a.strip() for a in cast_raw if a.strip() and a.strip().lower() != 'nan']:
            if actor not in cast_stats:
                cast_stats[actor] = {'sum': 0.0, 'count': 0}
            cast_stats[actor]['sum'] += r_val
            cast_stats[actor]['count'] += 1

    cast_target_map = {}
    cast_count_map = {}
    for a_name, stats in cast_stats.items():
        s = stats['sum']
        n = stats['count']
        cast_target_map[a_name] = float((s + k_smooth * user_mean) / (n + k_smooth))
        cast_count_map[a_name] = int(n)

    # Apply smoothed features to DataFrame
    df['director_target_encoded'] = df['director'].apply(
        lambda d: dir_target_map.get(str(d).strip(), user_mean) if d else user_mean
    )
    df['director_film_count'] = df['director'].apply(
        lambda d: float(dir_count_map.get(str(d).strip(), 0)) if d else 0.0
    )

    def extract_top2_cast_features(cast_val):
        actors = [a.strip() for a in str(cast_val or '').split(',') if a.strip()]
        a0 = actors[0] if len(actors) > 0 else ''
        a1 = actors[1] if len(actors) > 1 else ''
        score0 = cast_target_map.get(a0, user_mean)
        score1 = cast_target_map.get(a1, user_mean)
        count_max = max(cast_count_map.get(a0, 0), cast_count_map.get(a1, 0))
        return pd.Series([score0, score1, float(count_max)])

    cast_feats = df['cast'].apply(extract_top2_cast_features)
    cast_feats.columns = ['cast_0_target_encoded', 'cast_1_target_encoded', 'cast_film_count']
    df = pd.concat([df, cast_feats], axis=1)

    # 4. Runtime
    df['runtime_clean'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(105.0)

    # Package encoders dictionary
    encoders = {
        'director_target_map': dir_target_map,
        'dir_count_map': dir_count_map,
        'cast_target_map': cast_target_map,
        'cast_count_map': cast_count_map,
        'user_mean': user_mean,
        'user_std': user_std,
        'k_smooth': k_smooth
    }

    # Build feature columns
    feature_cols = [c for c in df.columns if c.startswith('genre_') or c.startswith('tfidf_')]
    feature_cols.extend([
        'director_target_encoded', 'director_film_count',
        'cast_0_target_encoded', 'cast_1_target_encoded', 'cast_film_count',
        'runtime_clean'
    ])

    X = df[feature_cols].fillna(0)
    y = df['Rating'].astype(float)

    # High quality tuned Random Forest regressor with leaf regularization
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    cached_data = (model, feature_cols, tfidf, encoders, time.time())
    _user_models[clean_user] = cached_data
    return model, feature_cols, tfidf, encoders

def get_or_train_user_model(username: str, force_retrain: bool = False):
    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user or clean_user == 'guest':
        return None, [], None, {}

    if not force_retrain and clean_user in _user_models:
        model, cols, vec, enc, ts = _user_models[clean_user]
        # Valid for 3 hours in RAM
        if time.time() - ts < 10800:
            return model, cols, vec, enc

    return train_user_model_in_memory(clean_user)

def invalidate_user_model(username: str):
    clean_user = (username or '').strip().lstrip('@').lower()
    if clean_user in _user_models:
        del _user_models[clean_user]
