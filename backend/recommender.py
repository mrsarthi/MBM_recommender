import os
import re
import pandas as pd
import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from backend.config import TMDB_KEY, TMDB_BASE_URL, PROFILE_PATH, APP_MEMORY_FILE
from backend.predictions import predict_movie_score, get_watch_providers

# Cached HTTP session with automated connection retry
http_session = requests_cache.CachedSession('tmdb_cache', backend='sqlite', expire_after=604800)
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

def titleNormalize(title):
    clean = re.sub(r'^Poster for\s+', '', str(title), flags=re.IGNORECASE).strip()
    clean = re.sub(r'\s*\(\d{4}\)$', '', clean).strip()
    return re.sub(r'[^a-z0-9]', '', clean.lower())

def load_watched_data(profile_path=PROFILE_PATH, memory_path=APP_MEMORY_FILE):
    watched_titles = set()
    watched_ids = set()
    hated_movies = set()
    
    if os.path.exists(profile_path):
        try:
            df = pd.read_csv(profile_path)
            df.columns = [c.strip() for c in df.columns]
            col_name = 'Name' if 'Name' in df.columns else 'Title'
            
            for _, row in df.iterrows():
                if col_name in row and pd.notna(row[col_name]):
                    t_norm = titleNormalize(row[col_name])
                    watched_titles.add(t_norm)
                    
                    if 'movie_id' in row and pd.notna(row['movie_id']):
                        try: watched_ids.add(int(float(row['movie_id'])))
                        except: pass
                        
                    if 'Rating' in row and pd.notna(row['Rating']):
                        try:
                            if float(row['Rating']) <= 2.5:
                                hated_movies.add(t_norm)
                        except: pass
        except Exception as e:
            print(f"Warning reading watched profile: {e}")
            
    if os.path.exists(memory_path) and os.path.getsize(memory_path) > 0:
        try:
            mem = pd.read_csv(memory_path)
            if 'movie_id' in mem.columns:
                watched_ids.update(mem['movie_id'].dropna().astype(int))
        except Exception: pass
        
    return watched_titles, watched_ids, hated_movies

from concurrent.futures import ThreadPoolExecutor

def analyze(watchedSet_titles, watchedSet_ids, hated_movies, ai_analysis, ai_model, ai_columns, ai_vectorizer, ai_encoders, user_context="Alone", streaming_filter="All Platforms", raw_prompt=""):
    """
    Finds candidates matching direct movie name, query/mood, or similar films,
    scores candidates using personal AI model, and returns curated recommendations.
    """
    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }
    idToGenre = {v: k for k, v in genreDict.items()}
    
    direct_matches = []
    results = []
    seen_ids = set()
    
    clean_raw = raw_prompt.strip() if raw_prompt else ''
    norm_raw = titleNormalize(clean_raw)

    # 1. Direct Movie Name Search on TMDB
    if clean_raw:
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': clean_raw}, timeout=6).json()
            matches = resp.get('results', []) if isinstance(resp, dict) else []
            for m in matches[:10]:
                m_id = m.get('id')
                if m_id and m_id not in seen_ids:
                    seen_ids.add(m_id)
                    m_norm = titleNormalize(m.get('title', ''))
                    # Check if exact or close title match
                    if norm_raw and (norm_raw == m_norm or norm_raw in m_norm or m_norm in norm_raw):
                        m['is_direct_match'] = True
                        if (m_norm in watchedSet_titles) or (m_id in watchedSet_ids):
                            m['is_watched'] = True
                        direct_matches.append(m)
                    else:
                        results.append(m)

            # Sort direct matches so exact title equality is first, then popularity
            direct_matches.sort(key=lambda x: (
                titleNormalize(x.get('title', '')) == norm_raw,
                x.get('vote_count', 0),
                x.get('popularity', 0)
            ), reverse=True)

            # If a top direct match was found, fetch its recommendations & similar films
            if direct_matches:
                top_id = direct_matches[0].get('id')
                try:
                    r_resp = http_session.get(f"{TMDB_BASE_URL}/movie/{top_id}/recommendations", params={'api_key': TMDB_KEY}, timeout=5).json()
                    for rm in r_resp.get('results', [])[:10]:
                        if rm.get('id') and rm.get('id') not in seen_ids:
                            seen_ids.add(rm.get('id'))
                            results.append(rm)
                except Exception: pass
        except Exception: pass

    # 2. Suggested Titles from Gemini (concurrent lookup)
    suggested_titles = ai_analysis.get('suggested_titles', [])
    if suggested_titles:
        def fetch_title(t):
            try:
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': t}, timeout=5).json()
                m = resp.get('results', []) if isinstance(resp, dict) else []
                return m[0] if m else None
            except Exception: return None
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            for res in executor.map(fetch_title, suggested_titles):
                if res and res.get('id') and res.get('id') not in seen_ids:
                    seen_ids.add(res.get('id'))
                    results.append(res)

    # 3. Search Query / Theme from Gemini
    search_query = ai_analysis.get('search_query', '')
    if search_query and search_query != clean_raw:
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': search_query}, timeout=6).json()
            matches = resp.get('results', []) if isinstance(resp, dict) else []
            for m in matches[:10]:
                if m.get('id') and m.get('id') not in seen_ids:
                    seen_ids.add(m.get('id'))
                    results.append(m)
        except Exception: pass

    # 4. Discover by Genres
    desiredGenres = ai_analysis.get('genres', [])
    if desiredGenres:
        targetGenreIds = [str(genreDict[name]) for name in desiredGenres if name in genreDict]
        if targetGenreIds:
            genreIdString = "|".join(targetGenreIds)
            discoverUrl = f"{TMDB_BASE_URL}/discover/movie"
            discoverParams = {
                'api_key': TMDB_KEY, 'with_genres': genreIdString,
                'vote_average.gte': 6.0, 'vote_count.gte': 60, 
                'sort_by': 'popularity.desc', 'language': 'en-US', 'page': 1
            }
            try:
                resp = http_session.get(discoverUrl, params=discoverParams, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        for m in data.get('results', []):
                            if m.get('id') and m.get('id') not in seen_ids:
                                seen_ids.add(m.get('id'))
                                results.append(m)
            except Exception: pass

    # Filter recommendations by watched (keep direct matches even if watched so user can inspect/log them)
    unwatched_results = []
    for movie in results:
        m_id = movie.get('id')
        title_norm = titleNormalize(movie.get('title', ''))
        if (title_norm not in watchedSet_titles) and (m_id not in watchedSet_ids):
            unwatched_results.append(movie)

    all_candidates = direct_matches + unwatched_results

    # If specific streaming platform filter is set, query concurrently
    if streaming_filter != "All Platforms" and all_candidates:
        def check_stream(m):
            provs = get_watch_providers(m.get('id'))
            m['providers'] = provs
            return m if any(streaming_filter.lower() in p.lower() for p in provs) else None
            
        with ThreadPoolExecutor(max_workers=8) as executor:
            all_candidates = [m for m in executor.map(check_stream, all_candidates) if m]

    finalPicks = []
    for movie in all_candidates:
        genres = [idToGenre[g] for g in movie.get('genre_ids', []) if g in idToGenre]
        overview = movie.get('overview', '')
        title_norm = titleNormalize(movie.get('title', ''))
        
        if ai_model:
            score = predict_movie_score(
                ai_model, ai_columns, ai_vectorizer, ai_encoders,
                genres=genres, context=user_context, overview=overview
            )
            for hated in hated_movies:
                if (hated in title_norm) or (title_norm in hated):
                    score = max(0.5, score - 2.5)
                    break
            movie['ai_score'] = score
        else:
            movie['ai_score'] = 3.5
            
        finalPicks.append(movie)

    # Sort unwatched recommendation results by AI score while keeping direct search matches prominent
    directs = [m for m in finalPicks if m.get('is_direct_match')]
    others = [m for m in finalPicks if not m.get('is_direct_match')]
    others.sort(key=lambda x: x.get('ai_score', 0), reverse=True)

    return directs + others
