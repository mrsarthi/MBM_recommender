import os
import re
import time
import threading
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from backend.config import TMDB_KEY, TMDB_BASE_URL, PROFILE_PATH, APP_MEMORY_FILE
from backend.predictions import predict_movie_score, predict_movie_scores_batch, get_watch_providers

class SimpleCachedSession:
    """Thread-safe in-memory cache on top of requests.Session with connection pooling and retries."""
    def __init__(self, ttl_seconds=604800):
        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=25, pool_maxsize=25)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._cache = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, url, params=None, timeout=6.0, **kwargs):
        param_tuple = tuple(sorted(params.items())) if params else ()
        cache_key = (url, param_tuple)
        now = time.time()

        with self._lock:
            if cache_key in self._cache:
                resp_obj, exp = self._cache[cache_key]
                if now < exp:
                    return resp_obj

        resp = self._session.get(url, params=params, timeout=timeout, **kwargs)
        if resp.status_code == 200:
            with self._lock:
                self._cache[cache_key] = (resp, now + self._ttl)
                if len(self._cache) > 2000:
                    oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])[:500]
                    for k in oldest_keys:
                        del self._cache[k]
        return resp

# Thread-safe cached HTTP session with connection pooling
http_session = SimpleCachedSession(ttl_seconds=604800)

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

# Words that describe a *mood*, concept, or trope rather than name a specific film.
# A query built from these must never promote same-named obscure films to the top.
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
    'time', 'travel', 'timetravel', 'cyberpunk', 'heist', 'robbery', 'zombie', 'zombies',
    'vampire', 'vampires', 'werewolf', 'alien', 'aliens', 'apocalypse', 'post-apocalyptic',
    'dystopia', 'space', 'robot', 'robots', 'ai', 'artificial', 'intelligence',
    'superhero', 'superheroes', 'detective', 'murder', 'killer', 'serial',
    'investigation', 'whodunnit', 'slasher', 'haunted', 'ghost', 'paranormal',
    'possession', 'exorcism', 'demon', 'survival', 'revenge', 'martial', 'arts',
    'sports', 'racing', 'prison', 'spy', 'espionage', 'conspiracy', 'multiverse',
    'dimension', 'parallel', 'loop', 'temporal', 'body', 'found', 'footage', 'psychological'
}

# Cache for TMDB keyword lookups to minimize API overhead (<0.5ms hit)
_tmdb_keyword_cache = {}

def _get_tmdb_keywords(query_str, api_key):
    query_clean = str(query_str or '').strip().lower()
    if not query_clean or not api_key:
        return []
    if query_clean in _tmdb_keyword_cache:
        return _tmdb_keyword_cache[query_clean]
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/search/keyword", params={'api_key': api_key, 'query': query_clean}, timeout=4).json()
        results = resp.get('results', []) if isinstance(resp, dict) else []
        kw_ids = [str(k['id']) for k in results[:4] if 'id' in k]
        _tmdb_keyword_cache[query_clean] = kw_ids
        return kw_ids
    except Exception:
        return []

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


def analyze(watchedSet_titles, watchedSet_ids, hated_movies, ai_analysis, ai_model, ai_columns, ai_vectorizer, ai_encoders, user_context="Alone", streaming_filter="All Platforms", raw_prompt="", source="all", username=None, tmdb_key=None, gemini_key=None):
    """
    Finds candidates matching direct movie name, query/mood, or similar films,
    scores candidates using personal AI model, and returns curated recommendations.
    """
    active_tmdb = (tmdb_key or '').strip()
    if not active_tmdb and username:
        try:
            from backend.db import get_user
            u_obj = get_user(username)
            if u_obj and u_obj.get('tmdb_key'):
                active_tmdb = str(u_obj['tmdb_key']).strip()
        except Exception:
            pass
    if not active_tmdb:
        active_tmdb = TMDB_KEY or ''

    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }
    idToGenre = {v: k for k, v in genreDict.items()}
    
    if source == "watchlist" and username:
        from backend.db import get_user_watchlist, get_user_taste_anchors
        from backend.gemini_client import filter_and_rank_watchlist_with_ai
        
        wl_movies = get_user_watchlist(username)
        if not wl_movies:
            return []

        taste_anchors = None
        try:
            taste_anchors = get_user_taste_anchors(username)
        except Exception:
            pass

        prompt_lower = (raw_prompt or '').lower()
        genre_keywords = {
            'horror': 'Horror', 'scary': 'Horror', 'slasher': 'Horror',
            'comedy': 'Comedy', 'funny': 'Comedy', 'hilarious': 'Comedy',
            'thriller': 'Thriller', 'tense': 'Thriller', 'suspense': 'Thriller',
            'sci-fi': 'Science Fiction', 'scifi': 'Science Fiction',
            'action': 'Action', 'drama': 'Drama', 'romance': 'Romance',
            'mystery': 'Mystery', 'crime': 'Crime', 'animation': 'Animation',
            'fantasy': 'Fantasy', 'western': 'Western', 'documentary': 'Documentary'
        }
        explicit_target_genres = {g for kw, g in genre_keywords.items() if kw in prompt_lower}
        # Merge genres from ai_analysis and raw prompt
        if ai_analysis and isinstance(ai_analysis, dict):
            for g in ai_analysis.get('genres', []):
                if g: explicit_target_genres.add(g)

        ai_matches = filter_and_rank_watchlist_with_ai(
            raw_prompt, wl_movies, custom_api_key=gemini_key, taste_context=taste_anchors
        )

        hated_set = {titleNormalize(h) for h in hated_movies if h}
        raw_scores = predict_movie_scores_batch(
            ai_model, ai_columns, ai_vectorizer, ai_encoders,
            wl_movies, context=user_context
        ) if ai_model else [3.8] * len(wl_movies)

        search_query_text = (ai_analysis.get('search_query', '') if isinstance(ai_analysis, dict) else '') or ''
        combined_query_text = f"{prompt_lower} {search_query_text.lower()}"

        synonym_groups = {
            'weird': ['weird', 'surreal', 'bizarre', 'strange', 'mutant', 'unconventional', 'psychedelic', 'cult', 'absurd', 'grotesque', 'insane'],
            'scary': ['scary', 'spooky', 'terrifying', 'slasher', 'haunting', 'paranormal', 'creepy'],
            'funny': ['funny', 'comedy', 'hilarious', 'humor', 'satire', 'spoof', 'wit'],
            'niche': ['niche', 'indie', 'arthouse', 'obscure', 'gem', 'underground', 'cult'],
            'epic': ['epic', 'universe', 'multiverse', 'adventure', 'monumental', 'grand']
        }

        candidates = []
        for idx, m in enumerate(wl_movies):
            m_id = int(m.get('movie_id') or m.get('id') or 0)
            m_genres = [g.strip() for g in str(m.get('genres', '')).split(',') if g.strip()]
            m_overview = str(m.get('overview', '')).lower()
            m_title = str(m.get('title', ''))
            m_norm_title = titleNormalize(m_title)

            # Retrieve AI thematic match data
            match_data = ai_matches.get(m_id)
            if match_data:
                thematic_rel = match_data.get('relevance', 0.6)
                vibe_pitch = match_data.get('vibe_pitch', '')
            else:
                thematic_rel = 0.15 if ai_matches else 0.50
                vibe_pitch = ''

            # Concept intersection matching
            concept_hits = 0
            for c_key, c_terms in synonym_groups.items():
                if any(t in combined_query_text for t in c_terms):
                    if any(t in m_overview or t in m_title.lower() or any(t in g.lower() for g in m_genres) for t in c_terms):
                        concept_hits += 1

            # Genre intersection
            genre_hits = 0
            if explicit_target_genres:
                m_genres_lower = [g.lower() for g in m_genres]
                genre_hits = sum(1 for tg in explicit_target_genres if tg.lower() in m_genres_lower)
                if genre_hits > 0:
                    thematic_rel = min(1.0, thematic_rel + 0.20 * genre_hits)
                else:
                    thematic_rel = max(0.05, thematic_rel - 0.40)

            base_score = raw_scores[idx]
            # Exact title equality check (fixes hated-movie substring penalty bug on 'Up' vs 'Upgrade')
            if m_norm_title in hated_set:
                base_score = max(0.5, base_score - 2.5)

            multiplier = 0.30 + (0.75 * thematic_rel)
            if thematic_rel >= 0.85:
                multiplier += 0.10
            final_ai_score = round(min(5.0, max(0.5, base_score * multiplier)), 2)

            # Multi-concept specificity boost
            concept_bonus = (concept_hits * 0.15) + (genre_hits * 0.10)

            # Unified rank score
            rank_score = (thematic_rel * 0.50) + ((final_ai_score / 5.0) * 0.30) + concept_bonus

            m_copy = dict(m)
            m_copy['id'] = m_id
            m_copy['movie_id'] = m_id
            m_copy['ai_score'] = final_ai_score
            m_copy['rank_score'] = round(rank_score, 4)
            m_copy['thematic_relevance'] = thematic_rel
            m_copy['vibe_pitch'] = vibe_pitch
            m_copy['is_direct_match'] = thematic_rel >= 0.80
            m_copy['is_watched'] = False

            if not ai_matches or thematic_rel >= 0.30:
                candidates.append(m_copy)

        if not candidates:
            for m in wl_movies:
                m_copy = dict(m)
                m_copy['id'] = m.get('movie_id')
                m_copy['ai_score'] = 3.5
                m_copy['rank_score'] = 0.5
                candidates.append(m_copy)

        if streaming_filter != "All Platforms":
            def check_stream(m):
                provs = get_watch_providers(m.get('id'), tmdb_key=active_tmdb)
                if any(streaming_filter.lower() in p.lower() for p in provs):
                    m['providers'] = provs
                    return m
                return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                candidates = [m for m in executor.map(check_stream, candidates) if m]

        candidates.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        return candidates

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
    if clean_raw and not is_mood and active_tmdb:
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
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': active_tmdb, 'query': q}, timeout=6).json()
                matches = resp.get('results', []) if isinstance(resp, dict) else []
                if not matches:
                    resp2 = http_session.get(f"{TMDB_BASE_URL}/search/multi", params={'api_key': active_tmdb, 'query': q}, timeout=6).json()
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
                r_resp = http_session.get(f"{TMDB_BASE_URL}/movie/{top_id}/recommendations", params={'api_key': active_tmdb}, timeout=5).json()
                for rm in r_resp.get('results', [])[:10]:
                    if rm.get('id') and rm.get('id') not in seen_ids:
                        seen_ids.add(rm.get('id'))
                        results.append(rm)
            except Exception: pass

    # 2. Suggested Titles from Gemini (concurrent lookup with high thematic priority)
    suggested_titles = ai_analysis.get('suggested_titles', [])
    if suggested_titles and active_tmdb:
        def fetch_title(item):
            if isinstance(item, dict):
                t = item.get('title', '')
                year_hint = item.get('year', '')
                pitch = item.get('vibe_pitch', '')
            else:
                t = str(item)
                year_hint = ''
                pitch = ''
            if not t:
                return None

            try:
                params = {'api_key': active_tmdb, 'query': t}
                if year_hint and str(year_hint).isdigit() and len(str(year_hint)) == 4:
                    params['year'] = year_hint
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5).json()
                m_list = resp.get('results', []) if isinstance(resp, dict) else []
                if not m_list and 'year' in params:
                    # Retry without year restriction in case year slightly differs
                    resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': active_tmdb, 'query': t}, timeout=5).json()
                    m_list = resp.get('results', []) if isinstance(resp, dict) else []

                if m_list:
                    m = m_list[0]
                    m['thematic_match'] = True
                    m['thematic_weight'] = 1.15
                    if pitch:
                        m['vibe_pitch'] = pitch
                    return m
                return None
            except Exception:
                return None
            
        with ThreadPoolExecutor(max_workers=8) as executor:
            for res in executor.map(fetch_title, suggested_titles):
                if res and res.get('id') and res.get('id') not in seen_ids:
                    seen_ids.add(res.get('id'))
                    results.append(res)

    # 3. TMDB Keyword-Constrained Thematic Discovery
    search_query = (ai_analysis.get('search_query') or clean_raw or '').strip()
    kw_ids = []
    if search_query and active_tmdb:
        kw_ids = _get_tmdb_keywords(search_query, active_tmdb)
        if not kw_ids and clean_raw and clean_raw != search_query:
            kw_ids = _get_tmdb_keywords(clean_raw, active_tmdb)

    if kw_ids and active_tmdb:
        def fetch_kw_discover_page(page):
            try:
                params = {
                    'api_key': active_tmdb,
                    'with_keywords': "|".join(kw_ids),
                    'vote_count.gte': 40,
                    'vote_average.gte': 5.8,
                    'sort_by': 'popularity.desc',
                    'language': 'en-US',
                    'page': page
                }
                resp = http_session.get(f"{TMDB_BASE_URL}/discover/movie", params=params, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get('results', []) if isinstance(data, dict) else []
                return []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=3) as executor:
            for page_results in executor.map(fetch_kw_discover_page, (1, 2, 3)):
                for m in page_results:
                    if m.get('id') and m.get('id') not in seen_ids:
                        seen_ids.add(m.get('id'))
                        m['thematic_match'] = True
                        m['thematic_weight'] = 1.10
                        results.append(m)

    # 4. Search Query / Theme Fallback
    if search_query and search_query != clean_raw and active_tmdb and len(results) < 20:
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': active_tmdb, 'query': search_query}, timeout=6).json()
            matches = resp.get('results', []) if isinstance(resp, dict) else []
            for m in matches[:10]:
                if m.get('id') and m.get('id') not in seen_ids:
                    seen_ids.add(m.get('id'))
                    results.append(m)
        except Exception: pass

    # 5. Discover by Genres (Fallback when thematic keyword pool is small)
    desiredGenres = ai_analysis.get('genres', [])
    if desiredGenres and active_tmdb and len(results) < 25:
        targetGenreIds = [str(genreDict[name]) for name in desiredGenres if name in genreDict]
        if targetGenreIds:
            genreIdString = "|".join(targetGenreIds)
            discoverUrl = f"{TMDB_BASE_URL}/discover/movie"
            discoverParams = {
                'api_key': active_tmdb, 'with_genres': genreIdString,
                'vote_average.gte': 6.2, 'vote_count.gte': 300, 
                'sort_by': 'popularity.desc', 'language': 'en-US', 'page': 1
            }
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
                for page_results in executor.map(fetch_discover_page, (1, 2)):
                    for m in page_results:
                        if m.get('id') and m.get('id') not in seen_ids:
                            seen_ids.add(m.get('id'))
                            results.append(m)

    # Filter recommendations: never recommend films the user has already watched / logged
    unwatched_directs = []
    for movie in direct_matches:
        m_id = movie.get('id')
        title_norm = titleNormalize(movie.get('title', ''))
        if (title_norm not in watchedSet_titles) and (m_id not in watchedSet_ids):
            unwatched_directs.append(movie)

    unwatched_results = []
    for movie in results:
        m_id = movie.get('id')
        title_norm = titleNormalize(movie.get('title', ''))
        if (title_norm not in watchedSet_titles) and (m_id not in watchedSet_ids):
            unwatched_results.append(movie)

    all_candidates = unwatched_directs + unwatched_results

    # If specific streaming platform filter is set, query concurrently
    if streaming_filter != "All Platforms" and all_candidates:
        def check_stream(m):
            provs = get_watch_providers(m.get('id'), tmdb_key=active_tmdb)
            m['providers'] = provs
            return m if any(streaming_filter.lower() in p.lower() for p in provs) else None
            
        with ThreadPoolExecutor(max_workers=8) as executor:
            all_candidates = [m for m in executor.map(check_stream, all_candidates) if m]

    for movie in all_candidates:
        movie['genres'] = [idToGenre[g] for g in movie.get('genre_ids', []) if g in idToGenre]

    # Vectorized batch prediction for all candidates (< 5ms)
    raw_scores = predict_movie_scores_batch(
        ai_model, ai_columns, ai_vectorizer, ai_encoders,
        all_candidates, context=user_context
    ) if ai_model else [3.5] * len(all_candidates)

    hated_set = {titleNormalize(h) for h in hated_movies if h}
    finalPicks = []
    for idx, movie in enumerate(all_candidates):
        title_norm = titleNormalize(movie.get('title', ''))
        thematic_weight = movie.get('thematic_weight', 1.0)
        raw_score = raw_scores[idx]
        
        # Exact title equality check (fixes hated-movie substring penalty bug)
        if title_norm in hated_set:
            raw_score = max(0.5, raw_score - 2.5)

        score = round(min(5.0, raw_score * thematic_weight), 2)
        movie['ai_score'] = score
        finalPicks.append(movie)

    # Sort unwatched recommendation results by AI score while keeping direct search matches prominent
    directs = [m for m in finalPicks if m.get('is_direct_match')]
    others = [m for m in finalPicks if not m.get('is_direct_match')]
    others.sort(key=lambda x: x.get('ai_score', 0), reverse=True)

    return directs + others
