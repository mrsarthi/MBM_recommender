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

def predict_movie_scores_batch(model, feature_cols, vectorizer, encoders, movies_list, context="Alone"):
    """
    Vectorized batch inference: computes AI match scores for an entire list of movies
    in a single 2D matrix pass (< 5ms for 200 films) instead of slow sequential loops.
    """
    if model is None or not feature_cols or not movies_list:
        return [3.8] * len(movies_list)

    n = len(movies_list)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    X = np.zeros((n, len(feature_cols)), dtype=np.float32)

    # Prior defaults from training
    user_mean = float((encoders or {}).get('user_mean', 3.5)) if encoders else 3.5
    dir_target_map = (encoders or {}).get('director_target_map', {}) if encoders else {}
    dir_count_map = (encoders or {}).get('dir_count_map', {}) if encoders else {}
    cast_target_map = (encoders or {}).get('cast_target_map', {}) if encoders else {}
    cast_count_map = (encoders or {}).get('cast_count_map', {}) if encoders else {}
    le_dir = (encoders or {}).get('director') if encoders else None
    dir_classes = set(getattr(le_dir, 'classes_', [])) if le_dir else set()

    # Pre-fill target encoded columns with user_mean prior baseline
    if 'director_target_encoded' in col_idx:
        X[:, col_idx['director_target_encoded']] = user_mean
    if 'cast_0_target_encoded' in col_idx:
        X[:, col_idx['cast_0_target_encoded']] = user_mean
    if 'cast_1_target_encoded' in col_idx:
        X[:, col_idx['cast_1_target_encoded']] = user_mean
    if 'runtime_clean' in col_idx:
        X[:, col_idx['runtime_clean']] = 105.0

    # 1. Extract and transform all overviews in one TF-IDF batch
    if vectorizer is not None:
        overviews = [str(m.get('overview') or '') for m in movies_list]
        try:
            tfidf_mat = vectorizer.transform(overviews).toarray()
            for i in range(min(tfidf_mat.shape[1], len(feature_cols))):
                col_name = f'tfidf_{i}'
                if col_name in col_idx:
                    X[:, col_idx[col_name]] = tfidf_mat[:, i]
        except Exception:
            pass

    # 2. Extract genres, directors, cast, and runtimes
    for row_i, m in enumerate(movies_list):
        # Genres
        genres_raw = m.get('genres', [])
        if isinstance(genres_raw, str):
            genres_raw = [g.strip() for g in genres_raw.split(',') if g.strip()]
        for g in genres_raw:
            col_name = f'genre_{str(g).strip()}'
            if col_name in col_idx:
                X[row_i, col_idx[col_name]] = 1.0

        # Director
        d = str(m.get('director') or '').strip()
        if d:
            if 'director_target_encoded' in col_idx and d in dir_target_map:
                X[row_i, col_idx['director_target_encoded']] = dir_target_map[d]
            if 'director_film_count' in col_idx:
                X[row_i, col_idx['director_film_count']] = float(dir_count_map.get(d, 0))
            if 'director_encoded' in col_idx and le_dir and d in dir_classes:
                try:
                    X[row_i, col_idx['director_encoded']] = float(le_dir.transform([d])[0])
                except Exception:
                    pass

        # Cast
        cast_raw = m.get('cast', '')
        if isinstance(cast_raw, list):
            actors = [str(a).strip() for a in cast_raw if str(a).strip()]
        else:
            actors = [a.strip() for a in str(cast_raw or '').split(',') if a.strip()]
        
        a0 = actors[0] if len(actors) > 0 else ''
        a1 = actors[1] if len(actors) > 1 else ''
        if 'cast_0_target_encoded' in col_idx and a0 in cast_target_map:
            X[row_i, col_idx['cast_0_target_encoded']] = cast_target_map[a0]
        if 'cast_1_target_encoded' in col_idx and a1 in cast_target_map:
            X[row_i, col_idx['cast_1_target_encoded']] = cast_target_map[a1]
        if 'cast_film_count' in col_idx:
            X[row_i, col_idx['cast_film_count']] = float(max(cast_count_map.get(a0, 0), cast_count_map.get(a1, 0)))

        # Runtime
        if 'runtime_clean' in col_idx:
            try:
                r = float(m.get('runtime', 0) or 0)
                if r > 0:
                    X[row_i, col_idx['runtime_clean']] = r
            except Exception:
                pass

    try:
        preds = model.predict(X)
        results = []
        for i, p in enumerate(preds):
            val = float(p)
            # Context adjustments
            m_genres = movies_list[i].get('genres', [])
            if isinstance(m_genres, str):
                m_genres = [g.strip() for g in m_genres.split(',')]
            if context == "With Partner":
                if any(g in ['Romance', 'Comedy', 'Drama'] for g in m_genres): val += 0.2
                if any(g in ['Horror', 'Documentary'] for g in m_genres): val -= 0.15
            elif context == "Friends Night":
                if any(g in ['Action', 'Comedy', 'Horror', 'Adventure'] for g in m_genres): val += 0.25
                if any(g in ['Drama', 'Documentary'] for g in m_genres): val -= 0.2
            elif context == "Family":
                if any(g in ['Animation', 'Family', 'Adventure'] for g in m_genres): val += 0.3
                if any(g in ['Horror', 'Crime', 'Thriller'] for g in m_genres): val -= 0.4
            
            results.append(round(min(5.0, max(0.5, val)), 1))
        return results
    except Exception:
        return [3.8] * n

def predict_movie_score(model, feature_cols, vectorizer, encoders, genres=None, director=None,
                        keywords=None, context="Alone", overview="", runtime=None, cast=None):
    """
    Predicts a personal rating (0.5-5.0) for candidate movie metadata.
    """
    if model is None or not feature_cols:
        return 3.5

    user_mean = float((encoders or {}).get('user_mean', 3.5)) if encoders else 3.5
    dir_target_map = (encoders or {}).get('director_target_map', {}) if encoders else {}
    dir_count_map = (encoders or {}).get('dir_count_map', {}) if encoders else {}
    cast_target_map = (encoders or {}).get('cast_target_map', {}) if encoders else {}
    cast_count_map = (encoders or {}).get('cast_count_map', {}) if encoders else {}

    genres = genres or []
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(',') if g.strip()]
    row = {c: 0.0 for c in feature_cols}
    col_set = set(row.keys())

    # Pre-fill priors
    if 'director_target_encoded' in col_set:
        row['director_target_encoded'] = user_mean
    if 'cast_0_target_encoded' in col_set:
        row['cast_0_target_encoded'] = user_mean
    if 'cast_1_target_encoded' in col_set:
        row['cast_1_target_encoded'] = user_mean
    if 'runtime_clean' in col_set:
        row['runtime_clean'] = 105.0

    # 1. Genres
    for g in genres:
        col = f'genre_{str(g).strip()}'
        if col in col_set:
            row[col] = 1.0

    # 2. Overview text features
    if vectorizer is not None and overview:
        try:
            vec_vals = vectorizer.transform([str(overview)]).toarray()[0]
            for i, val in enumerate(vec_vals):
                col = f'tfidf_{i}'
                if col in col_set:
                    row[col] = float(val)
        except Exception:
            pass

    # 3. Director
    if director:
        d = str(director).strip()
        if 'director_target_encoded' in col_set and d in dir_target_map:
            row['director_target_encoded'] = dir_target_map[d]
        if 'director_film_count' in col_set:
            row['director_film_count'] = float(dir_count_map.get(d, 0))
        if 'director_encoded' in col_set:
            le = (encoders or {}).get('director')
            classes = list(getattr(le, 'classes_', [])) if le else []
            if d in classes:
                row['director_encoded'] = float(classes.index(d))

    # 4. Cast
    if cast:
        if isinstance(cast, list):
            actors = [str(a).strip() for a in cast if str(a).strip()]
        else:
            actors = [a.strip() for a in str(cast or '').split(',') if a.strip()]
        a0 = actors[0] if len(actors) > 0 else ''
        a1 = actors[1] if len(actors) > 1 else ''
        if 'cast_0_target_encoded' in col_set and a0 in cast_target_map:
            row['cast_0_target_encoded'] = cast_target_map[a0]
        if 'cast_1_target_encoded' in col_set and a1 in cast_target_map:
            row['cast_1_target_encoded'] = cast_target_map[a1]
        if 'cast_film_count' in col_set:
            row['cast_film_count'] = float(max(cast_count_map.get(a0, 0), cast_count_map.get(a1, 0)))

    # 5. Runtime
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
                                   ai_columns=None, ai_vectorizer=None, ai_encoders=None, top_n=6, tmdb_key=None):
    """
    Fetches ripple recommendations on TMDB for a watched movie.
    """
    active_tmdb = (tmdb_key or TMDB_KEY or '').strip()
    if not active_tmdb or not movie_id: return []

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
        resp = requests.get(url, params={'api_key': active_tmdb, 'language': 'en-US'}, timeout=8).json()
        results = resp.get('results', []) if isinstance(resp, dict) else []
        
        if len(results) < 5:
            sim_url = f"{TMDB_BASE_URL}/movie/{movie_id}/similar"
            sim_resp = requests.get(sim_url, params={'api_key': active_tmdb, 'language': 'en-US'}, timeout=8).json()
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

_providers_cache = {}

def get_watch_providers(movie_id, region='US', tmdb_key=None):
    """
    Fetches streaming flatrate providers from TMDB (with fast RAM cache).
    """
    active_tmdb = (tmdb_key or TMDB_KEY or '').strip()
    if not active_tmdb or not movie_id: return []
    cache_key = f"{movie_id}_{region}_{active_tmdb[:6]}"
    if cache_key in _providers_cache:
        return _providers_cache[cache_key]

    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}/watch/providers"
        resp = requests.get(url, params={'api_key': active_tmdb}, timeout=4).json()
        if isinstance(resp, dict):
            flatrate = resp.get('results', {}).get(region, {}).get('flatrate', [])
            provs = [p.get('provider_name') for p in flatrate if p.get('provider_name')]
            _providers_cache[cache_key] = provs
            return provs
    except Exception: pass
    return []
