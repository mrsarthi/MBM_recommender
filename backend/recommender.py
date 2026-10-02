import math
import re
import time
import threading
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from backend.config import TMDB_KEY, TMDB_BASE_URL
from backend.predictions import predict_movie_scores_batch, get_watch_providers
from backend.collaborative import collaborative_engine
from backend.query_parser import _has_positive_content

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

# ── Person Filmography Cache (LRU) ──
_person_filmography_cache = {}
_person_cache_lock = threading.Lock()
_PERSON_CACHE_TTL = 604800  # 7 days


def _get_person_filmography(name, tmdb_key):
    """
    Queries TMDB /search/person then /person/{id}/movie_credits with memory LRU cache.
    Returns {'person_id': id, 'name': name, 'cast_ids': set(...), 'crew_ids': set(...), 'all_ids': set(...)}.
    """
    if not name or not tmdb_key:
        return None

    cache_key = (name.lower().strip(),)
    now = time.time()
    with _person_cache_lock:
        if cache_key in _person_filmography_cache:
            result, exp = _person_filmography_cache[cache_key]
            if now < exp:
                return result

    try:
        search_resp = http_session.get(
            f"{TMDB_BASE_URL}/search/person",
            params={'api_key': tmdb_key, 'query': name},
            timeout=6
        ).json()

        if not isinstance(search_resp, dict):
            return None

        results = search_resp.get('results', [])
        if not results:
            return None

        best = results[0]
        person_id = best.get('id')
        person_name = best.get('name', name)
        if not person_id:
            return None

        credits_resp = http_session.get(
            f"{TMDB_BASE_URL}/person/{person_id}/movie_credits",
            params={'api_key': tmdb_key},
            timeout=8
        ).json()

        if not isinstance(credits_resp, dict):
            return None

        cast_members = credits_resp.get('cast', [])
        crew_members = credits_resp.get('crew', [])

        cast_ids = set()
        crew_ids = set()

        for c in cast_members:
            cid = c.get('id')
            if cid:
                cast_ids.add(cid)

        directed_ids = set()
        for c in crew_members:
            cid = c.get('id')
            if cid:
                crew_ids.add(cid)
                if c.get('job') == 'Director':
                    directed_ids.add(cid)

        # A director's "movies" are the ones they directed, not everything they produced
        known_for = best.get('known_for_department') or ''
        if known_for == 'Directing' and directed_ids:
            signature_ids = directed_ids
        else:
            signature_ids = cast_ids | directed_ids

        result = {
            'person_id': person_id,
            'name': person_name,
            'known_for': known_for,
            'cast_ids': cast_ids,
            'crew_ids': crew_ids,
            'directed_ids': directed_ids,
            'signature_ids': signature_ids,
            'all_ids': cast_ids | crew_ids,
        }

        with _person_cache_lock:
            _person_filmography_cache[cache_key] = (result, now + _PERSON_CACHE_TTL)
            if len(_person_filmography_cache) > 200:
                oldest = sorted(_person_filmography_cache.keys(),
                                key=lambda k: _person_filmography_cache[k][1])[:50]
                for k in oldest:
                    del _person_filmography_cache[k]

        return result
    except Exception:
        return None

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
    'dimension', 'parallel', 'loop', 'temporal', 'body', 'found', 'footage', 'psychological',
    'gore', 'gory', 'splatter', 'nudity', 'nude', 'erotic', 'erotica', 'sex', 'sexual',
    'steamy', 'sensual', 'odyssey', 'quest', 'voyage', 'mythology', 'mythic',
    'anticipated', 'upcoming', 'coming'
}

# ── Exclusion by TMDB tag ──
# Plot overviews rarely say "nudity" or "gore", so exclusions are enforced against
# TMDB's own keyword tags: server-side via without_keywords on /discover, and by
# checking /movie/{id}/keywords for candidates from sources that can't filter.
_exclusion_tag_cache = {}
_movie_tag_cache = {}
_MOVIE_TAG_CACHE_MAX = 5000


def _get_exclusion_keyword_ids(term, api_key):
    """TMDB keyword ids whose name contains the term as a whole word ('nudity' -> 'female nudity', ...)."""
    term_clean = str(term or '').strip().lower()
    if not term_clean or not api_key:
        return []
    if term_clean in _exclusion_tag_cache:
        return _exclusion_tag_cache[term_clean]
    pattern = re.compile(r'\b' + re.escape(term_clean) + r'\b')
    ids = []
    try:
        for page in (1, 2):
            resp = http_session.get(f"{TMDB_BASE_URL}/search/keyword",
                                    params={'api_key': api_key, 'query': term_clean, 'page': page}, timeout=4).json()
            if not isinstance(resp, dict):
                break
            for k in resp.get('results', []):
                if k.get('id') and pattern.search(str(k.get('name', '')).lower()):
                    ids.append(str(k['id']))
            if page >= int(resp.get('total_pages') or 1):
                break
    except Exception:
        return ids
    _exclusion_tag_cache[term_clean] = ids
    return ids


def _get_movie_meta(movie_id, api_key):
    """
    TMDB tags and original language for a movie in one request, cached:
    {'tag_ids': set, 'tag_names': set, 'language': str}, or None if the lookup failed.
    """
    if not movie_id or not api_key:
        return None
    if movie_id in _movie_tag_cache:
        return _movie_tag_cache[movie_id]
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}",
                                params={'api_key': api_key, 'append_to_response': 'keywords'}, timeout=5).json()
        if not isinstance(resp, dict) or 'id' not in resp:
            return None
        keywords = (resp.get('keywords') or {}).get('keywords', []) or []
        meta = {
            'tag_ids': {str(k['id']) for k in keywords if k.get('id')},
            'tag_names': {_norm_tag(k.get('name')) for k in keywords if k.get('name')},
            'language': str(resp.get('original_language') or ''),
        }
    except Exception:
        return None
    if len(_movie_tag_cache) >= _MOVIE_TAG_CACHE_MAX:
        _movie_tag_cache.clear()
    _movie_tag_cache[movie_id] = meta
    return meta


def _backfill_movie_metadata_async(known_rows, pending_ids, api_key, limit=1000):
    """
    Saves looked-up language/tags in a background thread, and fetches the same for
    pending_ids (films not needed for this search) so later searches find them stored.
    """
    if not known_rows and not pending_ids:
        return

    def run():
        try:
            from backend.db import fill_movie_metadata
            if known_rows:
                fill_movie_metadata(known_rows)
            pending = list(pending_ids[:limit])
            with ThreadPoolExecutor(max_workers=6) as executor:
                for start in range(0, len(pending), 60):
                    chunk = pending[start:start + 60]
                    metas = executor.map(lambda mid: _get_movie_meta(mid, api_key), chunk)
                    fill_movie_metadata([(mid, meta['language'], ', '.join(sorted(meta['tag_names'])))
                                         for mid, meta in zip(chunk, metas) if meta])
        except Exception as e:
            print(f"[WARN] metadata backfill failed: {e}")

    threading.Thread(target=run, daemon=True).start()


def _get_movie_keyword_ids(movie_id, api_key):
    """Set of TMDB keyword ids tagged on a movie, or None if the lookup failed."""
    meta = _get_movie_meta(movie_id, api_key)
    return meta['tag_ids'] if meta else None


def _filter_by_excluded_tags(movies, excluded_tag_ids, api_key, want=None, batch=24):
    """
    Drops movies tagged with any excluded keyword id, preserving order.
    Movies marked 'tags_checked' (already filtered by /discover) are kept as-is.
    With want set, stops once that many survivors are found (for long ranked lists).
    A failed lookup keeps the movie rather than emptying the results.
    """
    if not excluded_tag_ids or not movies:
        return movies
    excluded = set(excluded_tag_ids)
    kept = []

    def check(m):
        if m.get('tags_checked'):
            return True
        tags = _get_movie_keyword_ids(m.get('id') or m.get('movie_id'), api_key)
        return tags is None or not (tags & excluded)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for start in range(0, len(movies), batch):
            chunk = movies[start:start + batch]
            for m, ok in zip(chunk, executor.map(check, chunk)):
                if ok:
                    kept.append(m)
            if want and len(kept) >= want:
                break
    return kept


def _user_top_genres(username, limit=3):
    """Genres the user rates highest, for prompts that only say what to avoid."""
    if not username:
        return []
    try:
        from backend.db import get_user_diary
        rows, _, _ = get_user_diary(username)
    except Exception:
        return []
    totals = {}
    for r in rows:
        try:
            rating = float(r.get('rating') or r.get('Rating') or 0)
        except (TypeError, ValueError):
            continue
        if rating < 3.5:
            continue
        for g in str(r.get('genres') or '').split(','):
            g = g.strip()
            if g:
                totals[g] = totals.get(g, 0) + rating
    return [g for g, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


def _person_ids_for_role(person_data, role):
    """Films that count as the person's: directed for 'director', acted in for 'actor'."""
    if not person_data:
        return set()
    if role == 'director' and person_data.get('directed_ids'):
        return person_data['directed_ids']
    if role == 'actor' and person_data.get('cast_ids'):
        return person_data['cast_ids']
    return person_data.get('signature_ids') or person_data.get('all_ids') or set()


def _person_name_matches(query_name, found_name):
    """True when TMDB's person is the one typed: same name, or the typed words are part of it."""
    import difflib
    q = re.sub(r'[^a-z ]', '', str(query_name or '').lower()).split()
    f = re.sub(r'[^a-z ]', '', str(found_name or '').lower()).split()
    if not q or not f:
        return False
    if set(q) <= set(f):
        return True
    return difflib.SequenceMatcher(None, ' '.join(q), ' '.join(f)).ratio() >= 0.85


# ── Search intent & relevance ──
_exact_tag_cache = {}
_FRAMING_WORDS = {'movies', 'movie', 'films', 'film', 'flicks', 'flick', 'something', 'anything',
                  'recommend', 'show', 'give', 'some', 'any', 'stuff'}


def _norm_tag(name):
    return re.sub(r'[\s\-_]+', ' ', re.sub(r'[()]', '', str(name or '').lower())).strip()


def _get_exact_keyword_ids(name, api_key):
    """TMDB keyword ids whose name equals `name` (ignoring case, hyphens, plural 's')."""
    target = _norm_tag(name)
    if not target or not api_key:
        return []
    if target in _exact_tag_cache:
        return _exact_tag_cache[target]
    accepted = {target, target + 's', target[:-1] if target.endswith('s') else target}
    ids = []
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/search/keyword",
                                params={'api_key': api_key, 'query': name}, timeout=4).json()
        for k in (resp.get('results', []) if isinstance(resp, dict) else []):
            name = _norm_tag(k.get('name'))
            # Location tags carry their country: "tokyo, japan", "paris, france"
            if k.get('id') and (name in accepted or name.split(',')[0].strip() == target):
                ids.append(str(k['id']))
    except Exception:
        return ids
    _exact_tag_cache[target] = ids
    return ids


def _build_intent(analysis, query_text, api_key, excluded_tag_ids=()):
    """
    Structured intent for a prompt: required/preferred/avoided genres, exact TMDB tag
    ids, overview terms, and whether any words look like part of a film title.
    """
    from backend.concepts import analyze_concepts, FILLER
    from backend.query_parser import LANGUAGE_MAP

    intent = analyze_concepts(query_text, excluded_genres=analysis.get('excluded_genres'))
    # Parser-only sub-genres ("giallo", "space opera") still contribute tags; they define
    # the theme when no concept did.
    tag_names = list(intent['tags'])
    core_names = list(intent['core_tags'])
    for kw in analysis.get('thematic_keywords') or []:
        if kw not in tag_names:
            tag_names.append(kw)
        if not intent['core_tags'] and kw not in core_names:
            core_names.append(kw)

    tokens = re.findall(r"[a-z0-9][a-z0-9'\-]*", (query_text or '').lower())
    person = f"{analysis.get('person') or ''} {analysis.get('person_matched') or ''}".lower() if analysis.get('person') else ''
    person_tokens = set(re.findall(r"[a-z0-9][a-z0-9'\-]*", person))
    person_tokens |= {t + "'s" for t in person_tokens}
    uncovered = [t for t in tokens
                 if t not in FILLER and t not in intent['covered'] and t not in LANGUAGE_MAP
                 and t not in person_tokens and not re.fullmatch(r"'?\d{2,4}'?s?", t)]
    intent['uncovered'] = uncovered
    intent['framed'] = any(t in _FRAMING_WORDS for t in tokens)

    # Descriptive prompt with an unknown theme word ("submarine movies"): try it as a tag
    if intent['framed']:
        tag_names.extend(uncovered[:3])
        core_names.extend(uncovered[:3])

    def resolve(names):
        ids = []
        if api_key and names:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for found in executor.map(lambda n: _get_exact_keyword_ids(n, api_key), names[:20]):
                    ids.extend(i for i in found if i not in ids and i not in excluded_tag_ids)
        return ids

    # Names are kept for matching locally stored tags (watchlist) without API calls
    intent['core_tag_names'] = list(dict.fromkeys(_norm_tag(n) for n in core_names))
    intent['tag_names'] = list(dict.fromkeys(_norm_tag(n) for n in tag_names))
    intent['core_tag_ids'] = resolve(core_names)
    intent['tag_ids'] = intent['core_tag_ids'] + [i for i in resolve(tag_names) if i not in intent['core_tag_ids']]

    # Fall back to the parser's genre guess only when no concept was recognized
    if not intent['genre_groups'] and not intent['soft_genres']:
        for g in analysis.get('genres') or []:
            intent['soft_genres'][g] = intent['soft_genres'].get(g, 0) + 1.0

    # Documentaries and TV movies only belong in results when asked for
    required = {g for group in intent['genre_groups'] for g in group}
    for g in ('Documentary', 'TV Movie'):
        if g not in required and g not in intent['soft_genres']:
            intent['avoid_genres'][g] = max(intent['avoid_genres'].get(g, 0), 1.5)
    return intent


def _relevance(genres, text, intent, tag_hit=False, person_hit=None, ref_hit=None):
    """
    How well a film fits the prompt, in [0, 1], and whether it meets the required genres.
    genres: set of genre names; text: lowercased title + overview.
    """
    parts = []
    groups = intent['genre_groups']
    hard_ok = all(genres & set(group) for group in groups)
    if groups:
        parts.append((1.0 if hard_ok else 0.0, 3.0))
    soft = intent['soft_genres']
    if soft:
        total = sum(soft.values())
        parts.append((sum(w for g, w in soft.items() if g in genres) / total, 2.0))
    if intent['tag_ids'] or intent['text_terms']:
        # tag_hit 2 = carries a theme-defining tag, 1 = only a supporting tag
        text_hit = any(term in text for term in intent['text_terms'])
        if tag_hit == 2 or (tag_hit and not intent['core_tag_ids']):
            theme = 1.0
        elif tag_hit:
            theme = 0.6
        else:
            theme = 0.5 if text_hit else 0.0
        parts.append((theme, 3.0 if intent['tag_ids'] else 1.5))
    if person_hit is not None:
        parts.append((1.0 if person_hit else 0.0, 4.0))
    if ref_hit is not None:
        parts.append((1.0 if ref_hit else 0.2, 2.0))

    rel = sum(s * w for s, w in parts) / sum(w for _, w in parts) if parts else 0.6
    avoid = max((w for g, w in intent['avoid_genres'].items() if g in genres), default=0.0)
    rel -= 0.15 * avoid
    return max(0.0, min(1.0, rel)), hard_ok


def _quality(m, prefer_obscure=False, upcoming=False):
    """Rating prior in [0, 1]: Bayesian-shrunk TMDB rating, so 9.0 from 12 votes isn't 'great'."""
    if upcoming:
        return min(1.0, math.log10(float(m.get('popularity') or 0) + 1) / 3)
    votes = float(m.get('vote_count') or 0)
    rating = float(m.get('vote_average') or 0)
    shrunk = (votes * rating + 150 * 6.4) / (votes + 150)
    q = max(0.0, min(1.0, (shrunk - 5.5) / 3.0))
    if prefer_obscure and 0 < votes < 2000:
        q = min(1.0, q + 0.15)
    return q


# Words carrying no signal either way; ignored when classifying and matching.
_FILLER_TERMS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'about', 'like', 'some', 'something', 'anything', 'movie', 'movies', 'film',
    'films', 'cinema', 'watch', 'watching', 'me', 'my', 'i', 'want', 'need', 'give',
    'show', 'recommend', 'recommendations', 'good', 'great', 'best', 'top',
    'is', 'it', 'that', 'this', 'very', 'really', 'kinda', 'kind', 'sort',
    'not', 'no', 'without', 'never', 'non', 'old', 'new', 'classic', 'classics',
    'vintage', 'recent', 'modern', 'latest', 'rated', 'acclaimed', 'masterpiece',
    'most', 'before', 'after', 'prior', 'than', 'but', 'however', 'between', 'during', 'until', 'since'
}

def _get_franchise_key(title):
    """
    Extracts a canonical root stem for franchise/sequel clustering.
    E.g. 'Saw II', 'Saw: The Final Chapter', 'Jigsaw' -> 'saw'
    'The Godfather Part II' -> 'the godfather'
    """
    if not title:
        return ""
    t = str(title).lower().strip()
    # Handle known franchise subtitles/spinoffs
    if any(s in t for s in ['saw', 'jigsaw', 'spiral']):
        if 'saw' in t or 'jigsaw' in t:
            return 'saw'
    # Split by colon, hyphen, 'part'
    t = re.split(r'[:\-]|\bpart\b|\bchapter\b|\bvol(?:ume|\.)?\b', t)[0].strip()
    # Remove trailing roman numerals or digits
    t = re.sub(r'\s+(?:[ivxlcdm]+|\d+)$', '', t).strip()
    return t



def looks_like_mood_query(raw):
    """
    True when the query describes a vibe or meta-intent rather than naming a specific film.
    """
    text = str(raw or '').strip().lower()
    if not text:
        return False

    # Conversational comparison / request / discovery patterns are never literal film titles
    if re.search(r'\b(?:something\s+like|movies?\s+like|films?\s+like|similar\s+to|in\s+the\s+vein\s+of|recommend\s+me|find\s+me|show\s+me|give\s+me|looking\s+for)\b', text):
        return True
    if re.search(r'\b(?:most\s+anticipated|upcoming|coming\s+soon|unreleased|before\s+\d{4}|after\s+\d{4})\b', text):
        return True
    if re.search(r'\b(?:gore|nudity|erotic|slasher|splatter)\s*(?:movies?|films?)?\b', text):
        return True

    tokens = [t for t in re.split(r'[^a-z0-9\-]+', text) if t]
    if not tokens:
        return False

    meaningful = [t for t in tokens if t not in _FILLER_TERMS]
    if not meaningful:
        return False

    mood_hits = sum(1 for t in meaningful if t in MOOD_TERMS)

    if mood_hits == len(meaningful):
        return True
    if mood_hits >= 2 and len(meaningful) >= 3:
        return True
    if len(meaningful) >= 4 and mood_hits >= 1:
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


def analyze(watchedSet_titles, watchedSet_ids, hated_movies, ai_analysis, ai_model, ai_columns, ai_vectorizer, ai_encoders, user_context="Alone", streaming_filter="All Platforms", raw_prompt="", source="all", username=None, tmdb_key=None):
    """
    Finds candidates matching direct movie name, query/mood, or similar films,
    scores candidates using personal AI model, and returns curated recommendations.
    """
    active_tmdb = (tmdb_key or '').strip()
    user_id = None
    if username:
        try:
            from backend.db import get_user
            u_obj = get_user(username)
            if u_obj:
                user_id = u_obj.get('id')
                if not active_tmdb and u_obj.get('tmdb_key'):
                    active_tmdb = str(u_obj['tmdb_key']).strip()
        except Exception:
            pass
    if not active_tmdb:
        active_tmdb = TMDB_KEY or ''

    # Constraints shared by the watchlist and discovery branches.
    person = ai_analysis.get('person') if isinstance(ai_analysis, dict) else None
    runtime_max = ai_analysis.get('runtime_max') if isinstance(ai_analysis, dict) else None
    vote_average_min = ai_analysis.get('vote_average_min') if isinstance(ai_analysis, dict) else None
    analysis = ai_analysis if isinstance(ai_analysis, dict) else {}
    # The parser guesses names from capitalised word pairs ("Blade Runner"); keep the
    # person only if TMDB knows someone by that name.
    if person and active_tmdb:
        person_data = _get_person_filmography(person, active_tmdb)
        if not person_data or not _person_name_matches(person, person_data.get('name')):
            person = None
            analysis = dict(analysis, person=None)
    exclusion_only = bool(analysis.get('exclusion_only'))
    # Non-negated part of the prompt; older callers without it fall back to the raw prompt.
    query_text = (analysis['positive_query'] if 'positive_query' in analysis else raw_prompt) or ''
    if exclusion_only or not _has_positive_content(query_text):
        query_text = ''

    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }
    idToGenre = {v: k for k, v in genreDict.items()}

    excluded_tag_ids = set()
    if active_tmdb:
        for term in analysis.get('negated_terms', []) or []:
            excluded_tag_ids.update(_get_exclusion_keyword_ids(term, active_tmdb))
    excluded_genre_ids = [str(genreDict[g]) for g in (analysis.get('excluded_genres') or []) if g in genreDict]

    def with_exclusions(params):
        """Adds server-side exclusions to a /discover/movie params dict."""
        if excluded_tag_ids:
            params['without_keywords'] = '|'.join(sorted(excluded_tag_ids))
        if excluded_genre_ids:
            params['without_genres'] = '|'.join(excluded_genre_ids)
        return params

    def mark_checked(movies):
        for m in movies:
            m['tags_checked'] = True
        return movies
    
    if source == "watchlist" and username:
        from backend.db import get_user_watchlist
        from backend.query_parser import rank_watchlist_relevance

        wl_movies = get_user_watchlist(username)
        if not wl_movies:
            return []

        # Same intent as discovery, but tags are matched by name against the TMDB tags
        # stored with each film, so no keyword-id lookups are needed.
        intent = _build_intent(analysis, query_text, None, excluded_tag_ids)
        intent['tag_ids'] = intent['tag_names']
        intent['core_tag_ids'] = intent['core_tag_names']
        core_names, tag_names = set(intent['core_tag_names']), set(intent['tag_names'])
        has_intent = bool(intent['genre_groups'] or intent['soft_genres'] or tag_names or intent['text_terms'])
        year_min, year_max = analysis.get('year_min'), analysis.get('year_max')
        languages = analysis.get('languages', []) or []

        def tag_hit_for(names):
            names = {n.split(',')[0].strip() for n in names} | set(names)
            return 2 if names & core_names else (1 if names & tag_names else 0)

        # Free-text matching for words no concept explains ("lighthouse", a director's surname)
        ai_matches = rank_watchlist_relevance(raw_prompt, wl_movies)
        raw_scores = predict_movie_scores_batch(
            ai_model, ai_columns, ai_vectorizer, ai_encoders,
            wl_movies, context=user_context
        ) if ai_model else [3.8] * len(wl_movies)
        hated_set = {titleNormalize(h) for h in hated_movies if h}

        candidates = []
        for idx, m in enumerate(wl_movies):
            m_id = int(m.get('movie_id') or m.get('id') or 0)
            year = str(m.get('year') or '').replace('.0', '')[:4]
            if year.isdigit() and ((year_min and int(year) < year_min) or (year_max and int(year) > year_max)):
                continue
            try:
                m_runtime = int(m.get('runtime') or 0)
            except (ValueError, TypeError):
                m_runtime = 0
            if runtime_max and m_runtime and m_runtime > runtime_max:
                continue
            m_vote = float(m.get('vote_average') or 0)
            if vote_average_min and m_vote and m_vote < vote_average_min:
                continue

            genres = {g.strip() for g in str(m.get('genres', '')).split(',') if g.strip()}
            text = f"{str(m.get('title', '')).lower()} {str(m.get('overview', '')).lower()}"
            stored_tags = {_norm_tag(t) for t in str(m.get('keywords') or '').split(',') if t.strip()}

            match_data = ai_matches.get(m_id)
            token_rel = match_data.get('relevance', 0.6) if match_data else (0.15 if ai_matches else 0.5)

            if has_intent:
                rel, hard_ok = _relevance(genres, text, intent, tag_hit=tag_hit_for(stored_tags))
                rel = 0.75 * rel + 0.25 * token_rel
            else:
                rel, hard_ok = token_rel, True

            predicted = raw_scores[idx]
            if titleNormalize(m.get('title', '')) in hated_set:
                predicted = max(0.5, predicted - 2.5)

            m_copy = dict(m)
            m_copy.pop('keywords', None)
            m_copy.update({
                'id': m_id, 'movie_id': m_id, 'is_watched': False,
                'vibe_pitch': match_data.get('vibe_pitch', '') if match_data else '',
                'relevance': rel, 'hard_ok': hard_ok, 'predicted_rating': round(predicted, 2),
                'has_stored_tags': bool(stored_tags), '_genres': genres, '_text': text,
            })
            candidates.append(m_copy)

        def rank_of(m):
            q = max(0.0, min(1.0, (float(m.get('vote_average') or 6.4) - 5.5) / 3.0))
            pred_n = max(0.0, min(1.0, (m['predicted_rating'] - 1.0) / 4.0))
            rank = 0.55 * m['relevance'] + 0.30 * pred_n + 0.15 * q
            return rank * (1.0 if m['hard_ok'] else 0.4)

        candidates.sort(key=rank_of, reverse=True)

        # Language and tags are stored with each film. Films synced before they were
        # stored are looked up on TMDB (leading candidates now, the rest in the
        # background) and written back, so each film is fetched only once.
        def missing_meta(m):
            return (languages and not m.get('original_language')) or (tag_names and not m['has_stored_tags'])

        need_meta = [m for m in candidates[:150] if missing_meta(m)]
        rest = [m['id'] for m in candidates[150:] if missing_meta(m)]
        if (need_meta or rest) and active_tmdb:
            with ThreadPoolExecutor(max_workers=8) as executor:
                metas = list(executor.map(lambda m: _get_movie_meta(m['id'], active_tmdb), need_meta))
            backfill = []
            for m, meta in zip(need_meta, metas):
                if not meta:
                    continue
                m['original_language'] = m.get('original_language') or meta['language']
                if tag_names and not m['has_stored_tags']:
                    rel, _ = _relevance(m['_genres'], m['_text'], intent, tag_hit=tag_hit_for(meta['tag_names']))
                    m['relevance'] = max(m['relevance'], 0.75 * rel + 0.25 * m['relevance'])
                backfill.append((m['id'], meta['language'], ', '.join(sorted(meta['tag_names']))))
            _backfill_movie_metadata_async(backfill, rest, active_tmdb)
            candidates.sort(key=rank_of, reverse=True)

        if languages:
            # Films whose language is still unknown can't be confirmed, so they're left out
            candidates = [m for m in candidates if m.get('original_language') in languages]

        for m in candidates:
            rank = rank_of(m)
            m['rank_score'] = round(rank, 4)
            m['ai_score'] = round(max(0.5, min(5.0, 1.0 + 4.0 * rank)), 2)
            m['is_direct_match'] = m['relevance'] >= 0.85 and m['hard_ok']
            for key in ('_genres', '_text', 'hard_ok', 'has_stored_tags'):
                m.pop(key, None)

        # Person entity filtering (director/actor): their own films, falling back to the
        # director field when TMDB credits are unavailable
        if person and active_tmdb:
            person_data = _get_person_filmography(person, active_tmdb)
            person_ids = _person_ids_for_role(person_data, analysis.get('person_role'))
            person_surname = person.lower().split()[-1]
            person_candidates = [m for m in candidates
                                 if m['id'] in person_ids or person_surname in str(m.get('director', '')).lower()]
            if person_candidates:
                candidates = person_candidates

        if streaming_filter != "All Platforms":
            def check_stream(m):
                provs = get_watch_providers(m.get('id'), tmdb_key=active_tmdb)
                if any(streaming_filter.lower() in p.lower() for p in provs):
                    m['providers'] = provs
                    return m
                return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                candidates = [m for m in executor.map(check_stream, candidates) if m]

        return _filter_by_excluded_tags(candidates, excluded_tag_ids, active_tmdb, want=40)

    direct_matches = []
    results = []
    seen_ids = set()

    clean_raw = query_text.strip()
    norm_raw = titleNormalize(clean_raw)

    year_min = analysis.get('year_min')
    year_max = analysis.get('year_max')
    languages = analysis.get('languages', []) or []
    lang_param = "|".join(languages) if languages else None
    is_upcoming = bool(analysis.get('is_upcoming'))
    ref_entity = analysis.get('reference_entity')

    intent = _build_intent(analysis, clean_raw, active_tmdb, excluded_tag_ids)
    # Every meaningful word is explained by a genre/theme/year/language/person, so the
    # prompt describes films rather than naming one ("christmas movies", "musicals").
    concept_query = bool(clean_raw) and not intent['uncovered']

    def add(m, **flags):
        m_id = m.get('id')
        if not m_id or m_id in seen_ids:
            return False
        seen_ids.add(m_id)
        m.update(flags)
        results.append(m)
        return True

    def apply_common(params):
        if lang_param:
            params['with_original_language'] = lang_param
        if year_min:
            params['primary_release_date.gte'] = f"{year_min}-01-01"
        if year_max:
            params['primary_release_date.lte'] = f"{year_max}-12-31"
        if vote_average_min:
            params['vote_average.gte'] = max(float(params.get('vote_average.gte', 0)), vote_average_min)
            params['vote_count.gte'] = max(int(params.get('vote_count.gte', 0)), 300)
        if runtime_max:
            params['with_runtime.lte'] = runtime_max
        return with_exclusions(params)

    def discover(params):
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/discover/movie", params=apply_common(dict(params)), timeout=6)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return mark_checked(data.get('results', []) if isinstance(data, dict) else [])
        except Exception:
            return []

    # 0. A title starting with a negation word ("No Time to Die", "Without a Paddle")
    # is parsed as an exclusion; an exact title match on the full prompt wins over that.
    raw_title = (raw_prompt or '').strip()
    if active_tmdb and analysis.get('negated_terms') and raw_title and len(raw_title.split()) <= 6:
        norm_full = titleNormalize(raw_title)
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': active_tmdb, 'query': raw_title}, timeout=6).json()
            for m in (resp.get('results', []) if isinstance(resp, dict) else [])[:10]:
                m_id = m.get('id')
                m_norm = titleNormalize(m.get('title') or m.get('original_title') or '')
                if m_id and m_id not in seen_ids and _is_strong_title_match(norm_full, m_norm):
                    seen_ids.add(m_id)
                    m['is_direct_match'] = True
                    m['exact_title_request'] = True
                    m['tags_checked'] = True
                    if (m_norm in watchedSet_titles) or (m_id in watchedSet_ids):
                        m['is_watched'] = True
                    direct_matches.append(m)
        except Exception:
            pass

    # 1. Title search on TMDB.
    #
    # A prompt made only of genre/theme words ("christmas movies") describes films, so a
    # title search would only surface specials literally named "Christmas Movie". A bare
    # concept word with no framing ("Alien", "Heat") may still be a title: search it,
    # but only accept a well-known exact match.
    is_mood = looks_like_mood_query(clean_raw)
    bare_concept = concept_query and not intent['framed']
    title_search = bool(clean_raw) and active_tmdb and (bare_concept or not is_mood) and not (concept_query and intent['framed'])
    if title_search:
        search_queries = [clean_raw, clean_raw[:-1] if clean_raw.endswith('s') else clean_raw + 's']
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
                    m_norm = titleNormalize(m.get('title') or m.get('name') or m.get('original_title') or '')
                    strong = _is_strong_title_match(norm_raw, m_norm)
                    # "Alien" is a famous film; "War" or "Horror" titles are obscure noise
                    if strong and concept_query and int(m.get('vote_count') or 0) < 1000:
                        strong = False
                    if strong:
                        seen_ids.add(m_id)
                        m['is_direct_match'] = True
                        if (m_norm in watchedSet_titles) or (m_id in watchedSet_ids):
                            m['is_watched'] = True
                        direct_matches.append(m)
                    elif not concept_query:
                        add(m)
            except Exception:
                pass

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
                    add(rm, ref_ripple=True)
            except Exception:
                pass

    # 2. Reference entity ("something like X"): the film plus TMDB's recommendations for it
    if ref_entity and active_tmdb:
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': active_tmdb, 'query': ref_entity}, timeout=5).json()
            ref_matches = resp.get('results', []) if isinstance(resp, dict) else []
            if ref_matches:
                top_ref = ref_matches[0]
                top_ref_id = top_ref.get('id')
                add(top_ref, ref_ripple=True)
                for page in (1, 2):
                    r_resp = http_session.get(f"{TMDB_BASE_URL}/movie/{top_ref_id}/recommendations",
                                              params={'api_key': active_tmdb, 'page': page}, timeout=5).json()
                    for rm in r_resp.get('results', []):
                        add(rm, ref_ripple=True)
        except Exception:
            pass

    # 3. Person discovery (director / actor)
    person_data = _get_person_filmography(person, active_tmdb) if (person and active_tmdb) else None
    person_ids = _person_ids_for_role(person_data, analysis.get('person_role'))
    if person_data and person_data.get('person_id'):
        base = {'api_key': active_tmdb, 'with_people': person_data['person_id']}
        with ThreadPoolExecutor(max_workers=2) as executor:
            pages = list(executor.map(discover, [dict(base, sort_by='popularity.desc', page=1),
                                                 dict(base, sort_by='vote_count.desc', page=1)]))
        for page_results in pages:
            for m in page_results:
                add(m, person_match=person)

    # 4. Upcoming releases
    if is_upcoming and active_tmdb:
        today_str = datetime.now().strftime('%Y-%m-%d')
        try:
            up_resp = http_session.get(f"{TMDB_BASE_URL}/movie/upcoming", params={'api_key': active_tmdb, 'page': 1}, timeout=5).json()
            for m in (up_resp.get('results', []) if isinstance(up_resp, dict) else []):
                add(m)
        except Exception:
            pass
        for m in discover({'api_key': active_tmdb, 'primary_release_date.gte': today_str,
                           'sort_by': 'popularity.desc', 'page': 1})[:25]:
            add(m)

    # 5. Theme and genre discovery.
    # Required genres are ANDed (comma); a group with alternatives contributes its first
    # (most typical) genre. Tags are exact TMDB keyword ids, ORed.
    groups = intent['genre_groups']
    hard_genre_ids = [str(genreDict[g[0]]) for g in groups if g and g[0] in genreDict]
    with_genres = ",".join(dict.fromkeys(hard_genre_ids))
    if not with_genres and intent['soft_genres']:
        top_soft = sorted(intent['soft_genres'].items(), key=lambda kv: -kv[1])[:2]
        with_genres = "|".join(str(genreDict[g]) for g, _ in top_soft if g in genreDict)
    # A prompt that only says what to avoid ("movies without nudity") has nothing to
    # search for, so fall back to the genres this user rates highest.
    if exclusion_only and not with_genres:
        with_genres = "|".join(str(genreDict[g]) for g in _user_top_genres(username) if g in genreDict)

    # Combined theme requests ("gore and nudity"): every group must be tagged (comma = AND)
    compound_groups = analysis.get('compound_keyword_groups', []) or []
    compound_kw_param = None
    if compound_groups and active_tmdb:
        group_strings = []
        for grp in compound_groups:
            ids = []
            for term in grp:
                ids.extend(i for i in _get_exact_keyword_ids(term, active_tmdb) if i not in ids)
            if ids:
                group_strings.append("|".join(ids[:6]))
        if len(group_strings) >= 2:
            compound_kw_param = ",".join(group_strings)

    jobs = []
    tag_queries = []
    if compound_kw_param:
        tag_queries.append((compound_kw_param, {'tag_hit': 2, 'compound_match': True}))
    elif intent['core_tag_ids']:
        tag_queries.append(("|".join(intent['core_tag_ids'][:20]), {'tag_hit': 2}))
    supporting = [i for i in intent['tag_ids'] if i not in intent['core_tag_ids']]
    if supporting and not compound_kw_param:
        tag_queries.append(("|".join(supporting[:20]), {'tag_hit': 1}))
    for kw_param, flags in tag_queries if active_tmdb else []:
        kw_base = {'api_key': active_tmdb, 'with_keywords': kw_param, 'vote_count.gte': 10}
        if hard_genre_ids:
            kw_base['with_genres'] = ",".join(dict.fromkeys(hard_genre_ids))
        sorts = (('popularity.desc', 1), ('popularity.desc', 2), ('vote_count.desc', 1)) if flags['tag_hit'] == 2             else (('popularity.desc', 1), ('vote_count.desc', 1))
        for sort, page in sorts:
            jobs.append((dict(kw_base, sort_by=sort, page=page), flags))
    if active_tmdb and (with_genres or exclusion_only):
        g_base = {'api_key': active_tmdb, 'vote_count.gte': 50, 'vote_average.gte': 5.8}
        if with_genres:
            g_base['with_genres'] = with_genres
        else:
            g_base['vote_count.gte'] = 300
        for sort, page in (('popularity.desc', 1), ('vote_count.desc', 1), ('vote_average.desc', 1)):
            params = dict(g_base, sort_by=sort, page=page)
            if sort == 'vote_average.desc':
                params['vote_count.gte'] = max(params['vote_count.gte'], 500)
            jobs.append((params, {}))
    if jobs:
        with ThreadPoolExecutor(max_workers=6) as executor:
            pages = list(executor.map(lambda job: discover(job[0]), jobs))
        for (params, flags), page_results in zip(jobs, pages):
            for m in page_results:
                add(m, **flags)

    # 6. Free-text fallback for prompts that aren't fully understood ("films set in a lighthouse")
    search_query = (analysis.get('search_query') or '').strip()
    if search_query and search_query != clean_raw and not concept_query and active_tmdb and len(results) < 20:
        try:
            params = {'api_key': active_tmdb, 'query': search_query}
            if year_min and not year_max:
                params['primary_release_year'] = year_min
            elif year_max and not year_min:
                params['primary_release_year'] = year_max
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=6).json()
            for m in (resp.get('results', []) if isinstance(resp, dict) else [])[:10]:
                add(m)
        except Exception:
            pass

    # Filter recommendations: enforce unwatched, language, year bounds, and negations
    excluded_genres = analysis.get('excluded_genres', []) or []
    excluded_keywords = analysis.get('excluded_keywords', []) or []

    def _passes_filters(m):
        # The whole prompt is this film's title; constraints parsed from it don't apply
        if m.get('exact_title_request'):
            return True
        rel_date = str(m.get('release_date') or m.get('year') or '')
        if rel_date and len(rel_date) >= 4 and rel_date[:4].isdigit():
            myear = int(rel_date[:4])
            if year_max and myear > year_max:
                return False
            if year_min and myear < year_min:
                return False
        if languages:
            m_lang = (m.get('original_language') or '').lower()
            if m_lang and m_lang not in languages:
                return False

        if runtime_max:
            m_runtime = m.get('runtime') or 0
            try:
                m_runtime = int(m_runtime)
            except (ValueError, TypeError):
                m_runtime = 0
            if m_runtime and m_runtime > 0 and m_runtime > runtime_max:
                return False

        if vote_average_min:
            m_vote = float(m.get('vote_average') or 0)
            if m_vote and m_vote < vote_average_min:
                return False

        # Check excluded genres
        m_g_ids = m.get('genre_ids', [])
        m_genres = [idToGenre[g].lower() for g in m_g_ids if g in idToGenre]
        if isinstance(m.get('genres'), list):
            m_genres.extend([str(g).lower() for g in m.get('genres', [])])
        elif isinstance(m.get('genres'), str):
            m_genres.extend([g.strip().lower() for g in m.get('genres', '').split(',') if g.strip()])

        if excluded_genres:
            for eg in excluded_genres:
                if eg.lower() in m_genres:
                    return False

        if excluded_keywords:
            m_title = str(m.get('title') or '').lower()
            m_overview = str(m.get('overview') or '').lower()
            for ek in excluded_keywords:
                if ek in m_title or ek in m_overview or any(ek in g for g in m_genres):
                    return False

        return True

    unwatched_directs = []
    for movie in direct_matches:
        m_id = movie.get('id')
        title_norm = titleNormalize(movie.get('title', ''))
        if (title_norm not in watchedSet_titles) and (m_id not in watchedSet_ids) and _passes_filters(movie):
            unwatched_directs.append(movie)

    unwatched_results = []
    for movie in results:
        m_id = movie.get('id')
        title_norm = titleNormalize(movie.get('title', ''))
        if (title_norm not in watchedSet_titles) and (m_id not in watchedSet_ids) and _passes_filters(movie):
            unwatched_results.append(movie)

    all_candidates = unwatched_directs + unwatched_results

    # Title search, recommendations and reference ripples can't filter by tag server-side
    all_candidates = _filter_by_excluded_tags(all_candidates, excluded_tag_ids, active_tmdb)

    # If specific streaming platform filter is set, query concurrently
    if streaming_filter != "All Platforms" and all_candidates:
        def check_stream(m):
            provs = get_watch_providers(m.get('id'), tmdb_key=active_tmdb)
            m['providers'] = provs
            return m if any(streaming_filter.lower() in p.lower() for p in provs) else None

        with ThreadPoolExecutor(max_workers=8) as executor:
            all_candidates = [m for m in executor.map(check_stream, all_candidates) if m]

    for movie in all_candidates:
        if movie.get('genre_ids'):
            movie['genres'] = [idToGenre[g] for g in movie.get('genre_ids', []) if g in idToGenre]
        elif not isinstance(movie.get('genres'), list):
            movie['genres'] = []

    # Vectorized batch prediction for all candidates (< 5ms)
    raw_scores = predict_movie_scores_batch(
        ai_model, ai_columns, ai_vectorizer, ai_encoders,
        all_candidates, context=user_context
    ) if ai_model else [3.5] * len(all_candidates)

    # Fetch collaborative filtering predictions if user_id is available
    cf_predictions = {}
    if user_id:
        try:
            cand_ids = [m.get('id') for m in all_candidates if m.get('id')]
            cf_predictions = collaborative_engine.get_collaborative_predictions(user_id, cand_ids)
        except Exception:
            cf_predictions = {}

    # Ranking: relevance to the prompt decides *which* films; the personal model and a
    # rating-quality prior decide the order among relevant films.
    hated_set = {titleNormalize(h) for h in hated_movies if h}
    finalPicks = []
    for idx, movie in enumerate(all_candidates):
        title_norm = titleNormalize(movie.get('title', ''))
        predicted = raw_scores[idx]
        if title_norm in hated_set:
            predicted = max(0.5, predicted - 2.5)
        m_id = movie.get('id')
        if m_id in cf_predictions:
            movie['collaborative_score'] = cf_predictions[m_id]
            predicted = (predicted * 0.75) + (cf_predictions[m_id] * 0.25)

        genres = set(movie.get('genres') or [])
        text = f"{str(movie.get('title') or '').lower()} {str(movie.get('overview') or '').lower()}"
        person_hit = (m_id in person_ids) if person_data else None
        ref_hit = bool(movie.get('ref_ripple')) if ref_entity else None
        rel, hard_ok = _relevance(genres, text, intent, tag_hit=int(movie.get('tag_hit') or 0),
                                  person_hit=person_hit, ref_hit=ref_hit)
        if movie.get('is_direct_match'):
            rel, hard_ok = 1.0, True

        quality = _quality(movie, prefer_obscure=intent['prefer_obscure'], upcoming=is_upcoming)
        predicted_norm = max(0.0, min(1.0, (predicted - 1.0) / 4.0))
        rank = 0.55 * rel + 0.25 * predicted_norm + 0.20 * quality
        if not hard_ok:
            rank *= 0.45
        if person_hit is False:
            rank *= 0.35

        movie['relevance'] = round(rel, 3)
        movie['rank_score'] = round(rank, 4)
        movie['predicted_rating'] = round(max(0.5, min(5.0, predicted)), 2)
        # The "% match" shown in the app follows the ranking, so order and number agree
        movie['ai_score'] = round(max(0.5, min(5.0, 1.0 + 4.0 * rank)), 2)
        finalPicks.append(movie)

    # Direct title matches stay on top; everything else by rank
    directs = [m for m in finalPicks if m.get('is_direct_match')]
    others = [m for m in finalPicks if not m.get('is_direct_match')]
    others.sort(key=lambda x: x.get('rank_score', 0), reverse=True)

    # Franchise & Sequel Diversity Gating:
    # Cap single franchises to max 1 entry in the top 12 (and max 2 overall) to prevent franchise flooding
    diverse_others = []
    overflow_sequels = []
    franchise_counts = {}

    for m in others:
        fkey = _get_franchise_key(m.get('title', ''))
        count = franchise_counts.get(fkey, 0) if fkey else 0
        if len(diverse_others) < 12:
            if fkey and count >= 1:
                overflow_sequels.append(m)
                continue
        elif fkey and count >= 2:
            overflow_sequels.append(m)
            continue

        if fkey:
            franchise_counts[fkey] = count + 1
        diverse_others.append(m)

    diverse_others.extend(overflow_sequels)
    return directs + diverse_others


