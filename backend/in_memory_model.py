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
    df = df.dropna(subset=['Rating'])
    if len(df) < 5:
        return None, [], None, {}

    # 1. Genres multi-hot encoding
    genre_series = df['genres'].fillna('').astype(str).str.split(', ')
    all_genres = set([g for sublist in genre_series for g in sublist if g])
    for g in all_genres:
        df[f'genre_{g}'] = df['genres'].fillna('').astype(str).apply(lambda x: 1 if g in x else 0)

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

    # 3. Director Encoding
    le_dir = LabelEncoder()
    df['director_encoded'] = le_dir.fit_transform(df['director'].fillna('Unknown').astype(str))
    encoders = {'director': le_dir}

    # 4. Runtime
    df['runtime_clean'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(105)

    # Build feature columns
    feature_cols = [c for c in df.columns if c.startswith('genre_') or c.startswith('tfidf_')]
    feature_cols.extend(['director_encoded', 'runtime_clean'])

    X = df[feature_cols].fillna(0)
    y = df['Rating'].astype(float)

    # Fast lightweight Random Forest regressor
    model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
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
