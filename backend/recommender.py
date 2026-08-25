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

# Words that describe a *mood* rather than name a film. A query built only from these
# must never promote same-named films to the top: searching "thriller" should return a
# thriller mood board, not the 1983 short film titled "Thriller".
MOOD_TERMS = {
    'action', 'adventure', 'animation', 'animated', 'comedy', 'comedies', 'crime',
    'documentary', 'documentaries', 'drama', 'dramas', 'family', 'fantasy', 'history',
    'historical', 'horror', 'music', 'musical', 'mystery', 'romance', 'romantic',
    'scifi', 'sci-fi', 'science', 'fiction', 'thriller', 'thrillers', 'war', 'western',
    'westerns', 'noir', 'neonoir', 'neo-noir', 'indie', 'arthouse', 'blockbuster',
    'happy', 'sad', 'sadder', 'melancholic', 'melancholy', 'tense', 'calm', 'calming',
    'cozy', 'comfort', 'comforting', 'nostalgic', 'nostalgia', 'excited', 'exciting',
    'thoughtful', 'scary', 'spooky', 'creepy', 'intense', 'mysterious', 'gritty',
    'dark', 'light', 'lighthearted', 'feelgood', 'feel-good', 'uplifting', 'depressing',
    'funny', 'hilarious', 'emotional', 'heartwarming', 'heartbreaking', 'weird',
    'surreal', 'mindbending', 'mind-bending', 'bending', 'slow', 'fast',
    'paced', 'pacing', 'violent', 'bloody', 'wholesome', 'chill', 'relaxing',
    'atmospheric', 'moody', 'bleak', 'hopeful', 'epic', 'quiet', 'loud', 'stylish',
    'aesthetic', 'aesthetics', 'vibe', 'vibes', 'mood', 'feeling', 'feels',
    'rainy', 'night', 'nighttime', 'summer', 'winter', 'autumn', 'rain', 'neon',
    'retro', 'vintage', 'classic', 'modern', 'futuristic', 'dystopian', 'cyber',
}

# Words carrying no signal either way; ignored when classifying.
_FILLER_TERMS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'about', 'like', 'some', 'something', 'anything', 'movie', 'movies', 'film',
    'films', 'cinema', 'watch', 'watching', 'me', 'my', 'i', 'want', 'need', 'give',
    'show', 'recommend', 'recommendations', 'good', 'great', 'best',
    'is', 'it', 'that', 'this', 'very', 'really', 'kinda', 'kind', 'sort',
}


def looks_like_mood_query(raw):
    """
    True when the query describes a vibe rather than naming a specific film.

    Used to decide whether TMDB title hits deserve to be pinned to the top of the
    results as direct matches. Both paths - title search and mood/genre discovery -
    always run; this only controls ranking.
    """
    text = str(raw or '').strip().lower()
    if not text:
        return False
    tokens = [t for t in re.split(r'[^a-z0-9\-]+', text) if t]
    if not tokens:
        return False

    meaningful = [t for t in tokens if t not in _FILLER_TERMS]
    if not meaningful:
        return False

    mood_hits = sum(1 for t in meaningful if t in MOOD_TERMS)

    # Every meaningful word is a mood/genre word -> unambiguously a mood query.
    if mood_hits == len(meaningful):
        return True
    # Several mood words together describe a vibe, not a title.
    if mood_hits >= 2 and len(meaningful) >= 3:
        return True
    # A long phrase with any mood word in it is a vibe. Kept at 5+ words so real
    # titles like "A Rainy Day in New York" stay searchable as titles.
    if len(meaningful) >= 5 and mood_hits >= 1:
        return True
    return False


def _is_strong_title_match(norm_query, norm_title):
    """
    Strict title matching for pinning a result as a direct match.

    Deliberately NOT a two-way substring test: that is what made every film whose
    title merely contains the query outrank the actual mood recommendations.
    """
    if not norm_query or not norm_title:
        return False
    if norm_query == norm_title:
        return True
    # Tolerate a singular/plural difference ("swing girl" -> "Swing Girls").
    if norm_query + 's' == norm_title or norm_query == norm_title + 's':
        return True
    # Tolerate a leading-article difference.
    for article in ('the', 'a', 'an'):
        if norm_title.startswith(article) and norm_title[len(article):] == norm_query:
            return True
        if norm_query.startswith(article) and norm_query[len(article):] == norm_title:
            return True
    return False


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

    # 1. Title search on TMDB.
    #
    # Both this and the mood/genre discovery below always run, so "Blade Runner" and
    # "melancholic rainy-night sci-fi" both work. The difference is ranking: only a
    # genuine title match gets pinned to the top as a direct match.
    is_mood = looks_like_mood_query(clean_raw)

    # For a pure vibe query, a title search only contributes films that happen to share
    # the mood word as a name (searching "thriller" surfacing the film *Thriller*).
    # Genre discovery below is the right source for those, so skip the title pass.
    if clean_raw and not is_mood:
        search_queries = [clean_raw]
        # Singular/plural retry helps real titles, but on a mood word it only drags in
        # more same-named films, so only do it when the query looks like a title.
        if not is_mood:
            if clean_raw.endswith('s'):
                search_queries.append(clean_raw[:-1])
            else:
                search_queries.append(clean_raw + 's')

        for q in search_queries:
            try:
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': q}, timeout=6).json()
                matches = resp.get('results', []) if isinstance(resp, dict) else []
                if not matches:
                    resp2 = http_session.get(f"{TMDB_BASE_URL}/search/multi", params={'api_key': TMDB_KEY, 'query': q}, timeout=6).json()
                    matches = [m for m in resp2.get('results', []) if m.get('media_type') != 'person'] if isinstance(resp2, dict) else []

                for m in matches[:10]:
                    m_id = m.get('id')
                    if not m_id or m_id in seen_ids:
                        continue
                    seen_ids.add(m_id)
                    m_title = m.get('title') or m.get('name') or m.get('original_title') or ''
                    m_norm = titleNormalize(m_title)

                    if (not is_mood) and _is_strong_title_match(norm_raw, m_norm):
                        m['is_direct_match'] = True
                        if (m_norm in watchedSet_titles) or (m_id in watchedSet_ids):
                            m['is_watched'] = True
                        direct_matches.append(m)
                    else:
                        results.append(m)
            except Exception: pass

        # Exact title equality first, then popularity.
        direct_matches.sort(key=lambda x: (
            titleNormalize(x.get('title', '')) == norm_raw,
            x.get('vote_count', 0),
            x.get('popularity', 0)
        ), reverse=True)

        # Pull in films similar to the confirmed title match.
        if direct_matches:
            top_id = direct_matches[0].get('id')
            try:
                r_resp = http_session.get(f"{TMDB_BASE_URL}/movie/{top_id}/recommendations", params={'api_key': TMDB_KEY}, timeout=5).json()
                for rm in r_resp.get('results', [])[:10]:
                    if rm.get('id') and rm.get('id') not in seen_ids:
                        seen_ids.add(rm.get('id'))
                        results.append(rm)
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
                'vote_average.gte': 6.2, 'vote_count.gte': 300, 
                'sort_by': 'popularity.desc', 'language': 'en-US', 'page': 1
            }
            # Pull a few pages so the personal model has a real pool to rank: one page of
            # popularity.desc is dominated by whatever released this month. Fetched in
            # parallel to keep the endpoint responsive on a small Render instance.
            def fetch_discover_page(page):
                try:
                    params = dict(discoverParams, page=page)
                    resp = http_session.get(discoverUrl, params=params, timeout=6)
                    if resp.status_code != 200:
                        return []
                    data = resp.json()
                    return data.get('results', []) if isinstance(data, dict) else []
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=3) as executor:
                for page_results in executor.map(fetch_discover_page, (1, 2, 3)):
                    for m in page_results:
                        if m.get('id') and m.get('id') not in seen_ids:
                            seen_ids.add(m.get('id'))
                            results.append(m)

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
                genres=genres, context=user_context, overview=overview,
                runtime=movie.get('runtime')
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
