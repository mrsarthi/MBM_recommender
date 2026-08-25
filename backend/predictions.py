import os
import joblib
import pandas as pd
import numpy as np
import requests
from backend.config import MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH, TMDB_KEY, TMDB_BASE_URL

def load_ai(model_path=MODEL_PATH, cols_path=COLUMNS_PATH, vec_path=VECTORIZER_PATH, enc_path=ENCODERS_PATH):
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    cols = joblib.load(cols_path) if os.path.exists(cols_path) else None
    vec = joblib.load(vec_path) if os.path.exists(vec_path) else None
    encoders = joblib.load(enc_path) if os.path.exists(enc_path) else None
    return model, cols, vec, encoders

def predict_movie_score(model, feature_cols, vectorizer, encoders, genres=None, director=None,
                        keywords=None, context="Alone", overview="", runtime=None):
    """
    Predicts a personal rating (0.5-5.0) for candidate movie metadata.

    The feature vector built here MUST mirror the one built during training in
    backend/in_memory_model.py, which produces:
        genre_<Name>      multi-hot genre flags
        tfidf_<i>         i-th component of the fitted TF-IDF vector over the overview
        director_encoded  LabelEncoder index for the director
        runtime_clean     runtime in minutes (105 when unknown)

    Older on-disk models used `ov_<word>` / `dir_<name>` instead, so both layouts are
    filled in and whichever the model actually declares is the one that gets used.
    """
    if model is None or not feature_cols:
        return 3.5

    genres = genres or []
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(',') if g.strip()]
    row = {c: 0.0 for c in feature_cols}
    col_set = row.keys()

    # 1. Genres (same column name in both layouts)
    for g in genres:
        col = f'genre_{str(g).strip()}'
        if col in col_set:
            row[col] = 1.0

    # 2. Overview text features
    if vectorizer is not None and overview:
        try:
            vec_vals = vectorizer.transform([str(overview)]).toarray()[0]
            # Current layout: positional tfidf_<i>
            for i, val in enumerate(vec_vals):
                col = f'tfidf_{i}'
                if col in col_set:
                    row[col] = float(val)
            # Legacy layout: ov_<term>
            if any(c.startswith('ov_') for c in col_set):
                for w, val in zip(vectorizer.get_feature_names_out(), vec_vals):
                    col = f'ov_{w}'
                    if col in col_set:
                        row[col] = float(val)
        except Exception:
            pass

    # 3. Director
    if director:
        d = str(director).strip()
        if 'director_encoded' in col_set:
            le = (encoders or {}).get('director')
            try:
                # LabelEncoder.transform raises on unseen labels, so look it up first.
                classes = list(getattr(le, 'classes_', []))
                if d in classes:
                    row['director_encoded'] = float(classes.index(d))
            except Exception:
                pass
        col = f'dir_{d}'
        if col in col_set:
            row[col] = 1.0

    # 4. Runtime — training filled unknowns with 105, so 0 would be far out of distribution.
    if 'runtime_clean' in col_set:
        try:
            rt = float(runtime) if runtime not in (None, '', 0) else 105.0
        except (TypeError, ValueError):
            rt = 105.0
        row['runtime_clean'] = rt

    df = pd.DataFrame([row], columns=feature_cols)
    try:
        pred = float(model.predict(df)[0])
    except Exception:
        return 3.5

    # Context adjustments
    if context == "With Partner":
        if any(g in ['Romance', 'Comedy', 'Drama'] for g in genres): pred += 0.2
        if any(g in ['Horror', 'Documentary'] for g in genres): pred -= 0.15
    elif context == "Friends Night":
        if any(g in ['Action', 'Comedy', 'Horror', 'Adventure'] for g in genres): pred += 0.25
        if any(g in ['Drama', 'Documentary'] for g in genres): pred -= 0.2
    elif context == "Family":
        if any(g in ['Animation', 'Family', 'Adventure'] for g in genres): pred += 0.3
        if any(g in ['Crime', 'Horror', 'Thriller'] for g in genres): pred -= 0.5

    return float(np.clip(pred, 0.5, 5.0))

def get_post_watch_recommendations(movie_id, watched_titles=None, watched_ids=None, ai_model=None,
                                   ai_columns=None, ai_vectorizer=None, ai_encoders=None, top_n=6):
    """
    Fetches ripple recommendations on TMDB for a watched movie.
    """
    if not TMDB_KEY or not movie_id: return []

    watched_titles = set(watched_titles or [])
    watched_ids = set(watched_ids or [])

    genre_dict = {
        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
        9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
        10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
    }
    
    candidates = []
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
        resp = requests.get(url, params={'api_key': TMDB_KEY, 'language': 'en-US'}, timeout=8).json()
        results = resp.get('results', []) if isinstance(resp, dict) else []
        
        if len(results) < 5:
            sim_url = f"{TMDB_BASE_URL}/movie/{movie_id}/similar"
            sim_resp = requests.get(sim_url, params={'api_key': TMDB_KEY, 'language': 'en-US'}, timeout=8).json()
            if isinstance(sim_resp, dict):
                results.extend(sim_resp.get('results', []))
                
        seen = set()
        for m in results:
            m_id = m.get('id')
            title = m.get('title', '')
            norm_title = title.lower().replace(' ', '')
            
            if not m_id or m_id in seen or m_id in watched_ids or norm_title in watched_titles:
                continue
            seen.add(m_id)
            
            genres = [genre_dict[g] for g in m.get('genre_ids', []) if g in genre_dict]
            overview = m.get('overview', '')
            
            score = 3.5
            if ai_model:
                score = predict_movie_score(ai_model, ai_columns, ai_vectorizer, ai_encoders, genres=genres, overview=overview)
            m['ai_score'] = score
            candidates.append(m)
            
        candidates.sort(key=lambda x: (x.get('ai_score', 0), x.get('vote_average', 0)), reverse=True)
        return candidates[:top_n]
    except Exception as e:
        print(f"Error getting ripple recommendations: {e}")
        return []

def get_watch_providers(movie_id, region='US'):
    """
    Fetches streaming flatrate providers from TMDB.
    """
    if not TMDB_KEY or not movie_id: return []
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}/watch/providers"
        resp = requests.get(url, params={'api_key': TMDB_KEY}, timeout=6).json()
        if isinstance(resp, dict):
            flatrate = resp.get('results', {}).get(region, {}).get('flatrate', [])
            return [p.get('provider_name') for p in flatrate if p.get('provider_name')]
    except Exception: pass
    return []
