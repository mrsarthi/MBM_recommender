import threading
import time
import uuid
import re
import html
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from psycopg2.extras import RealDictCursor
from backend.config import TMDB_KEY, TMDB_BASE_URL
from backend.db import (
    get_or_create_user, get_user, upsert_movies_batch, upsert_user_diary,
    upsert_user_watchlist, get_movie_ids_by_slugs, stable_movie_id,
    get_user_diary_map, cleanup_database_duplicates, invalidate_user_cache,
    get_connection, release_connection
)
from backend.in_memory_model import train_user_model_in_memory

# In-memory job tracking
_jobs = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 3600

LB_BASE = "https://letterboxd.com"

# Letterboxd 403s a bare User-Agent on some list pages; it wants a full browser header set.
BROWSER_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://letterboxd.com/',
}

genreDict = {
    28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
    80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
    14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
    9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
    10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
}


def title_normalize(t: str) -> str:
    if not t or str(t).lower() == 'nan':
        return ""
    clean = re.sub(r'^Poster for\s+', '', str(t), flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d{4}\)$', '', clean)
    return re.sub(r'[^a-z0-9]', '', clean.lower())


def _split_title_year(item_name):
    """'Office Romance (2026)' -> ('Office Romance', '2026')"""
    name = html.unescape(str(item_name or '').strip())
    m = re.search(r'^(.*?)\s*\((\d{4})\)\s*$', name)
    if m:
        return m.group(1).strip(), m.group(2)
    return name, ''


def _year_from_slug(slug):
    m = re.search(r'-(\d{4})$', str(slug or ''))
    return m.group(1) if m else ''


# ── Job bookkeeping ──

def _prune_jobs():
    now = time.time()
    stale = [j for j, d in _jobs.items()
             if now - d.get('start_time', now) > _JOB_TTL_SECONDS
             and d.get('status') in ('completed', 'failed')]
    for j in stale:
        _jobs.pop(j, None)


def start_onboarding_job(username, pin=None, tmdb_key=None, gemini_key=None, skip_scrape=False, favorites=None):
    clean_user = (username or '').strip().lstrip('@').lower()
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _prune_jobs()
        _jobs[job_id] = {
            'job_id': job_id,
            'username': clean_user,
            'status': 'running',
            'progress': 3,
            'stage': 'init',
            'message': 'Setting up profile...' if skip_scrape else 'Connecting to Letterboxd...',
            'error': None,
            'diary_count': 0,
            'watchlist_count': 0,
            'start_time': time.time()
        }

    t = threading.Thread(
        target=_run_onboarding_pipeline,
        args=(job_id, clean_user, pin, tmdb_key, gemini_key, skip_scrape, favorites),
        daemon=True
    )
    t.start()
    return job_id


def start_watchlist_sync_job(username, tmdb_key=None):
    clean_user = (username or '').strip().lstrip('@').lower()
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _prune_jobs()
        _jobs[job_id] = {
            'job_id': job_id,
            'username': clean_user,
            'status': 'running',
            'progress': 5,
            'stage': 'init',
            'message': 'Connecting to Letterboxd watchlist...',
            'error': None,
            'watchlist_count': 0,
            'start_time': time.time()
        }

    t = threading.Thread(
        target=_run_watchlist_sync_pipeline,
        args=(job_id, clean_user, tmdb_key),
        daemon=True
    )
    t.start()
    return job_id


def start_diary_sync_job(username, tmdb_key=None):
    clean_user = (username or '').strip().lstrip('@').lower()
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _prune_jobs()
        _jobs[job_id] = {
            'job_id': job_id,
            'username': clean_user,
            'status': 'running',
            'progress': 5,
            'stage': 'init',
            'message': 'Checking latest Letterboxd diary entries...',
            'error': None,
            'diary_count': 0,
            'start_time': time.time()
        }

    t = threading.Thread(
        target=_run_diary_sync_pipeline,
        args=(job_id, clean_user, tmdb_key),
        daemon=True
    )
    t.start()
    return job_id


def start_csv_import_job(username, csv_content, is_watchlist=False, tmdb_key=None):
    """
    Runs a Letterboxd CSV export import in the background.

    A full 700-film export needs hundreds of TMDB lookups, far past any sensible HTTP
    request timeout, so it reports progress through the same job channel as onboarding.
    """
    clean_user = (username or '').strip().lstrip('@').lower()
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _prune_jobs()
        _jobs[job_id] = {
            'job_id': job_id,
            'username': clean_user,
            'status': 'running',
            'progress': 5,
            'stage': 'parsing',
            'message': 'Reading your CSV export...',
            'error': None,
            'start_time': time.time()
        }

    def worker():
        from backend.csv_importer import import_letterboxd_csv
        try:
            def progress_cb(pct, msg):
                _update_job(job_id, pct, 'importing', msg)

            ok, msg = import_letterboxd_csv(
                clean_user, csv_content, is_watchlist=is_watchlist,
                tmdb_key=tmdb_key or TMDB_KEY, job_id=job_id, progress_cb=progress_cb
            )
            _update_job(job_id, 100, 'completed' if ok else 'failed', msg,
                        status='completed' if ok else 'failed',
                        error=None if ok else msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _update_job(job_id, 100, 'failed', 'Import failed: {0}'.format(e),
                        status='failed', error=str(e))

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def get_job_status(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {
            'job_id': job_id,
            'status': 'not_found',
            'progress': 0,
            'message': 'Job not found',
            'error': 'Invalid Job ID'
        }))


def _update_job(job_id, progress, stage, message, status='running', error=None, **extra):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update({
                'progress': progress, 'stage': stage,
                'message': message, 'status': status
            })
            if error:
                _jobs[job_id]['error'] = error
            if extra:
                _jobs[job_id].update(extra)


# ── Letterboxd scraping ──

def get_scrape_session():
    try:
        from curl_cffi import requests as curl_requests
        s = curl_requests.Session()
        s.is_curl_cffi = True
    except ImportError:
        s = requests.Session()
        s.headers.update(BROWSER_HEADERS)
        s.is_curl_cffi = False
    return s

def scrape_letterboxd_diary(username, session=None, max_pages=60, on_page=None):
    """
    Scrapes the FULL diary from the paginated /films/diary/ pages.

    The RSS feed this used to rely on only ever returns the 50 most recent entries,
    which is why a 700-film account only imported 51 rows. Each diary row carries
    data-item-slug, data-item-name ("Title (Year)"), a `rated-N` class (N is out of
    10, so stars are N/2) and a /for/YYYY/MM/DD/ watched date.
    """
    s = session or get_scrape_session()
    is_curl = getattr(s, 'is_curl_cffi', False) or ('curl_cffi' in s.__class__.__module__)

    entries = []
    seen_slugs = set()

    for page in range(1, max_pages + 1):
        url = "{0}/{1}/films/diary/page/{2}/".format(LB_BASE, username, page)
        try:
            if is_curl:
                resp = s.get(url, impersonate="chrome", timeout=15)
            else:
                resp = s.get(url, timeout=15)
        except Exception:
            break
        if resp.status_code != 200:
            break

        rows = re.findall(r'<tr class="diary-entry-row.*?</tr>', resp.text, re.DOTALL)
        if not rows:
            break

        for row in rows:
            slug_m = re.search(r'data-item-slug="([^"]+)"', row)
            if not slug_m:
                continue
            slug = slug_m.group(1).strip()

            # A film rewatched on several dates appears once per viewing, but
            # user_diary is unique on (user_id, movie_id) - keep the newest only.
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            name_m = re.search(r'data-item-name="([^"]*)"', row)
            title, year = _split_title_year(name_m.group(1) if name_m else '')
            if not title:
                title = slug.replace('-', ' ').title()
            if not year:
                year = _year_from_slug(slug)

            rating = None
            rate_m = re.search(r'rated-(\d+)', row)
            if rate_m:
                try:
                    rating = round(int(rate_m.group(1)) / 2.0, 1)
                except ValueError:
                    rating = None

            watched_date = ''
            date_m = re.search(r'/for/(\d{4})/(\d{2})/(\d{2})/', row)
            if date_m:
                watched_date = "{0}-{1}-{2}".format(date_m.group(1), date_m.group(2), date_m.group(3))

            entries.append({
                'slug': slug, 'title': title, 'year_hint': year,
                'rating': rating, 'watched_date': watched_date
            })

        if on_page:
            on_page(page, len(entries))

        # A short page means we reached the end of the diary.
        if len(rows) < 50:
            break

    # Fallback to Letterboxd RSS feed if HTML scrape was blocked or returned no entries
    if not entries:
        try:
            import xml.etree.ElementTree as ET
            rss_url = "{0}/{1}/rss/".format(LB_BASE, username)
            r_resp = s.get(rss_url, timeout=12)
            if r_resp.status_code == 200:
                root = ET.fromstring(r_resp.content)
                ns = {'letterboxd': 'https://letterboxd.com', 'tmdb': 'https://www.themoviedb.org'}
                for item in root.findall('./channel/item'):
                    title_elem = item.find('letterboxd:filmTitle', ns)
                    year_elem = item.find('letterboxd:filmYear', ns)
                    rating_elem = item.find('letterboxd:memberRating', ns)
                    date_elem = item.find('letterboxd:watchedDate', ns)
                    link_elem = item.find('link')

                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''
                    if not title:
                        raw_t = item.find('title')
                        if raw_t is not None and raw_t.text:
                            m = re.match(r'^(.*?),\s*(\d{4})?\s*-\s*([★½]+)?', raw_t.text)
                            if m: title = m.group(1).strip()
                    if not title:
                        continue

                    year = year_elem.text.strip() if year_elem is not None and year_elem.text else ''
                    rating = None
                    if rating_elem is not None and rating_elem.text:
                        try: rating = float(rating_elem.text.strip())
                        except: rating = None
                    date = date_elem.text.strip() if date_elem is not None and date_elem.text else ''
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ''
                    slug = link.rstrip('/').split('/')[-1] if link else title_normalize(title)

                    if slug not in seen_slugs:
                        seen_slugs.add(slug)
                        entries.append({
                            'slug': slug, 'title': title, 'year_hint': year,
                            'rating': rating, 'watched_date': date
                        })
        except Exception:
            pass

    return entries


def scrape_letterboxd_watchlist(username, session=None, max_pages=40):
    """
    Scrapes the full watchlist.

    Reads only data-item-slug / data-item-name, which appear exclusively on film
    posters. The previous version fell back to matching every <img alt="...">, which
    swept up the profile avatar - whose alt text is the member's display name - and
    imported it as a film (e.g. a phantom entry titled "Parth Sarthi Mishra").
    """
    s = session or get_scrape_session()
    is_curl = getattr(s, 'is_curl_cffi', False) or ('curl_cffi' in s.__class__.__module__)

    entries = []
    seen = set()

    for page in range(1, max_pages + 1):
        url = "{0}/{1}/watchlist/page/{2}/".format(LB_BASE, username, page)
        try:
            if is_curl:
                resp = s.get(url, impersonate="chrome", timeout=15)
            else:
                resp = s.get(url, timeout=15)
        except Exception:
            break
        if resp.status_code != 200:
            break

        slugs = re.findall(r'data-item-slug="([^"]+)"', resp.text)
        names = re.findall(r'data-item-name="([^"]*)"', resp.text)
        if not slugs:
            break

        for i, slug in enumerate(slugs):
            slug = slug.strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            raw_name = names[i] if i < len(names) else ''
            title, year = _split_title_year(raw_name)
            if not title:
                title = slug.replace('-', ' ').title()
            if not year:
                year = _year_from_slug(slug)
            entries.append({'slug': slug, 'title': title, 'year_hint': year})

        if "/watchlist/page/{0}/".format(page + 1) not in resp.text:
            break

    return entries


# ── TMDB resolution ──

def _tmdb_details(movie_id, tmdb_k):
    """Full TMDB record including credits and keywords, in a single request."""
    try:
        r = requests.get(
            "{0}/movie/{1}".format(TMDB_BASE_URL, movie_id),
            params={'api_key': tmdb_k, 'append_to_response': 'credits,keywords'},
            timeout=10
        ).json()
    except Exception:
        return None
    if not isinstance(r, dict) or not r.get('id'):
        return None

    credits = r.get('credits') or {}
    director = ''
    for c in credits.get('crew', []) or []:
        if c.get('job') == 'Director' and c.get('name'):
            director = c['name']
            break
    cast = ", ".join([c.get('name', '') for c in (credits.get('cast') or [])[:8] if c.get('name')])
    kw_block = (r.get('keywords') or {}).get('keywords', []) or []
    kw = ", ".join([k.get('name', '') for k in kw_block[:15] if k.get('name')])

    return {
        'movie_id': int(r['id']),
        'title': r.get('title') or r.get('original_title') or 'Untitled',
        'year': (r.get('release_date') or '')[:4],
        'genres': ", ".join([g['name'] for g in r.get('genres', [])]) or 'General',
        'overview': r.get('overview') or '',
        'director': director,
        'cast': cast,
        'keywords': kw,
        'runtime': r.get('runtime') or 0,
        'vote_average': round(float(r.get('vote_average') or 7.0), 1),
        'poster_path': r.get('poster_path') or '',
        'backdrop_path': r.get('backdrop_path') or '',
    }


def _tmdb_search_id(entry, tmdb_k):
    """Resolves a scraped Letterboxd entry to a TMDB movie id via title/year search."""
    title = (entry.get('title') or '').strip()
    year = str(entry.get('year_hint') or '').strip()
    slug = (entry.get('slug') or '').strip()
    norm = title_normalize(title)

    queries = [title]
    slug_name = re.sub(r'-\d{4}$', '', slug).replace('-', ' ').strip()
    if slug_name and slug_name.lower() != title.lower():
        queries.append(slug_name)
    stripped = re.sub(r'[:\-,].*$', '', title).strip()
    if stripped and stripped != title:
        queries.append(stripped)

    candidates = []
    for q in queries[:3]:
        param_sets = []
        if year and len(year) == 4:
            param_sets.append({'api_key': tmdb_k, 'query': q, 'year': year})
        param_sets.append({'api_key': tmdb_k, 'query': q})

        for params in param_sets:
            try:
                resp = requests.get("{0}/search/movie".format(TMDB_BASE_URL), params=params,
                                    headers={'User-Agent': 'MBMR/5.0'}, timeout=10).json()
            except Exception:
                continue
            results = resp.get('results', []) if isinstance(resp, dict) else []
            known = set(c['id'] for c in candidates)
            for m in results:
                if m.get('id') and m['id'] not in known:
                    candidates.append(m)
                    known.add(m['id'])
            if candidates:
                break
        if candidates:
            break

    if not candidates:
        return None

    # Prefer exact title in the right year, then exact title, then the most-voted result.
    for m in candidates:
        mt = title_normalize(m.get('title') or m.get('original_title') or '')
        rel = (m.get('release_date') or '')[:4]
        if mt == norm and year and rel == year:
            return int(m['id'])
    for m in candidates:
        mt = title_normalize(m.get('title') or m.get('original_title') or '')
        if mt == norm:
            return int(m['id'])
    candidates.sort(key=lambda x: (x.get('vote_count', 0), x.get('popularity', 0)), reverse=True)
    return int(candidates[0]['id'])


def _placeholder_movie(entry):
    """Deterministic local record for a film TMDB could not resolve."""
    slug = entry.get('slug') or entry.get('title') or 'unknown'
    return {
        'movie_id': stable_movie_id(slug),
        'title': entry.get('title') or str(slug).replace('-', ' ').title(),
        'year': entry.get('year_hint') or '',
        'genres': 'General',
        'overview': '',
        'director': '',
        'cast': '',
        'keywords': '',
        'runtime': 0,
        'vote_average': 7.0,
        'poster_path': '',
        'backdrop_path': '',
        'letterboxd_slug': slug,
    }


def resolve_entries(entries, tmdb_k, job_id=None, base_progress=0, span=0, label='films'):
    """
    Resolves scraped Letterboxd entries to full movie records.

    Three tiers, cheapest first:
      1. slug already in the shared `movies` table -> no network call at all
      2. TMDB search by title/year, then one details call for credits & keywords
      3. deterministic placeholder, so an unmatched film still appears in the diary
    """
    if not entries:
        return [], {}

    slug_to_id = dict(get_movie_ids_by_slugs([e['slug'] for e in entries]))
    unresolved = [e for e in entries if e['slug'] not in slug_to_id]

    movie_records = []
    if not unresolved:
        return movie_records, slug_to_id

    def work(e):
        try:
            return e, _tmdb_search_id(e, tmdb_k)
        except Exception:
            return e, None

    resolved_pairs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, pair in enumerate(ex.map(work, unresolved)):
            resolved_pairs.append(pair)
            if job_id and span and i % 25 == 0:
                pct = base_progress + int(span * (i / max(len(unresolved), 1)))
                _update_job(job_id, pct, 'resolving',
                            'Matching {0} on TMDB ({1}/{2})...'.format(label, i, len(unresolved)))

    need_details = [(e, mid) for e, mid in resolved_pairs if mid]
    details = []
    if need_details:
        with ThreadPoolExecutor(max_workers=8) as ex:
            details = list(ex.map(lambda pair: _tmdb_details(pair[1], tmdb_k), need_details))

    for (e, _mid), det in zip(need_details, details):
        if det:
            det['letterboxd_slug'] = e['slug']
            movie_records.append(det)
            slug_to_id[e['slug']] = det['movie_id']
        else:
            ph = _placeholder_movie(e)
            movie_records.append(ph)
            slug_to_id[e['slug']] = ph['movie_id']

    for e, mid in resolved_pairs:
        if not mid:
            ph = _placeholder_movie(e)
            movie_records.append(ph)
            slug_to_id[e['slug']] = ph['movie_id']

    return movie_records, slug_to_id

def repair_user_unhydrated_movies(username: str, tmdb_key: str = None) -> int:
    """
    Finds any diary or watchlist entries for a user that are missing posters or are placeholders,
    resolves them on TMDB, and updates the database records.
    """
    user = get_user(username)
    if not user:
        return 0

    active_tmdb = tmdb_key or (user.get('tmdb_key') if user else '') or TMDB_KEY
    if not active_tmdb:
        return 0

    conn = get_connection()
    missing_entries = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT m.movie_id, m.title, m.year, m.letterboxd_slug, d.watched_date, d.rating
                FROM user_diary d
                JOIN movies m ON d.movie_id = m.movie_id
                WHERE d.user_id = %s
                  AND (m.poster_path IS NULL OR m.poster_path = '' OR m.movie_id >= 900000000 OR m.genres = 'General')
            """, (user['id'],))
            diary_missing = cur.fetchall()
            for r in diary_missing:
                missing_entries.append({
                    'slug': r.get('letterboxd_slug') or title_normalize(r.get('title')),
                    'title': r.get('title'),
                    'year_hint': r.get('year'),
                    'rating': r.get('rating'),
                    'watched_date': r.get('watched_date') or ''
                })

            cur.execute("""
                SELECT DISTINCT m.movie_id, m.title, m.year, m.letterboxd_slug, w.added_date
                FROM user_watchlist w
                JOIN movies m ON w.movie_id = m.movie_id
                WHERE w.user_id = %s
                  AND (m.poster_path IS NULL OR m.poster_path = '' OR m.movie_id >= 900000000 OR m.genres = 'General')
            """, (user['id'],))
            wl_missing = cur.fetchall()
            for r in wl_missing:
                missing_entries.append({
                    'slug': r.get('letterboxd_slug') or title_normalize(r.get('title')),
                    'title': r.get('title'),
                    'year_hint': r.get('year')
                })
    finally:
        release_connection(conn)

    if not missing_entries:
        return 0

    # Deduplicate entries by slug
    deduped_entries = {}
    for e in missing_entries:
        slug = e.get('slug') or title_normalize(e.get('title'))
        if slug and slug not in deduped_entries:
            deduped_entries[slug] = e

    entries_to_resolve = list(deduped_entries.values())
    movie_records, slug_to_id = resolve_entries(entries_to_resolve, active_tmdb, label='unhydrated films')

    if movie_records:
        upsert_movies_batch(movie_records)

    diary_updates = []
    for e in entries_to_resolve:
        if 'watched_date' in e or 'rating' in e:
            mid = slug_to_id.get(e['slug'])
            if mid:
                diary_updates.append({
                    'movie_id': mid,
                    'rating': e.get('rating'),
                    'watched_date': e.get('watched_date') or ''
                })

    if diary_updates:
        upsert_user_diary(user['id'], diary_updates)

    cleanup_database_duplicates(user['id'])
    invalidate_user_cache(user['id'])
    return len(movie_records)


# ── Pipelines ──

def _run_watchlist_sync_pipeline(job_id, username, tmdb_key):
    try:
        user = get_or_create_user(username, tmdb_key=tmdb_key)
        if not user:
            _update_job(job_id, 100, 'error', 'Failed to find or create user profile',
                        status='failed', error='User creation failed')
            return

        active_tmdb = tmdb_key or (user.get('tmdb_key') if user else '') or TMDB_KEY

        _update_job(job_id, 15, 'watchlist_scrape', 'Reading Letterboxd watchlist for @{0}...'.format(username))
        session = get_scrape_session()
        wl_entries = scrape_letterboxd_watchlist(username, session=session)

        _update_job(job_id, 45, 'watchlist_scrape',
                    'Found {0} watchlist films.'.format(len(wl_entries)),
                    watchlist_count=len(wl_entries))

        wl_links = []
        if wl_entries:
            wl_records, wl_slug_to_id = resolve_entries(
                wl_entries, active_tmdb, job_id=job_id,
                base_progress=45, span=40, label='watchlist films'
            )
            if wl_records:
                upsert_movies_batch(wl_records)
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            for e in wl_entries:
                mid = wl_slug_to_id.get(e['slug'])
                if mid:
                    wl_links.append({'movie_id': mid, 'added_date': today})
            if wl_links:
                upsert_user_watchlist(user['id'], wl_links)

        cleanup_database_duplicates(user['id'])
        invalidate_user_cache(user['id'])

        _update_job(
            job_id, 100, 'completed',
            'Done - Synced {0} watchlist films.'.format(len(wl_links)),
            status='completed', watchlist_count=len(wl_links)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job(job_id, 100, 'failed', 'Watchlist sync failed: {0}'.format(e),
                    status='failed', error=str(e))


def _run_diary_sync_pipeline(job_id, username, tmdb_key):
    try:
        user = get_or_create_user(username, tmdb_key=tmdb_key)
        if not user:
            _update_job(job_id, 100, 'error', 'Failed to find or create user profile',
                        status='failed', error='User creation failed')
            return

        active_tmdb = tmdb_key or (user.get('tmdb_key') if user else '') or TMDB_KEY

        _update_job(job_id, 10, 'diary_check', 'Checking for new diary entries for @{0}...'.format(username))
        existing_map = get_user_diary_map(user['id'])

        session = get_scrape_session()
        is_curl = getattr(session, 'is_curl_cffi', False) or ('curl_cffi' in session.__class__.__module__)

        new_or_updated = []
        seen_slugs = set()
        max_sync_pages = 5

        for page in range(1, max_sync_pages + 1):
            _update_job(job_id, min(10 + page * 5, 35), 'diary_scrape',
                        'Checking diary page {0}...'.format(page))
            url = "{0}/{1}/films/diary/page/{2}/".format(LB_BASE, username, page)
            try:
                if is_curl:
                    resp = session.get(url, impersonate="chrome", timeout=15)
                else:
                    resp = session.get(url, timeout=15)
            except Exception:
                break
            if resp.status_code != 200:
                break

            rows = re.findall(r'<tr class="diary-entry-row.*?</tr>', resp.text, re.DOTALL)
            if not rows:
                break

            page_new_count = 0
            for row in rows:
                slug_m = re.search(r'data-item-slug="([^"]+)"', row)
                if not slug_m:
                    continue
                slug = slug_m.group(1).strip()
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)

                name_m = re.search(r'data-item-name="([^"]*)"', row)
                title, year = _split_title_year(name_m.group(1) if name_m else '')
                if not title:
                    title = slug.replace('-', ' ').title()
                if not year:
                    year = _year_from_slug(slug)

                rating = None
                rate_m = re.search(r'rated-(\d+)', row)
                if rate_m:
                    try:
                        rating = round(int(rate_m.group(1)) / 2.0, 1)
                    except ValueError:
                        rating = None

                watched_date = ''
                date_m = re.search(r'/for/(\d{4})/(\d{2})/(\d{2})/', row)
                if date_m:
                    watched_date = "{0}-{1}-{2}".format(date_m.group(1), date_m.group(2), date_m.group(3))

                existing = existing_map.get(f"slug_{slug.lower()}") or existing_map.get(f"title_{title.lower()}")
                if existing:
                    ex_rating = existing.get('rating')
                    ex_date = existing.get('watched_date') or ''
                    ex_mid = int(existing.get('movie_id') or 0)
                    ex_poster = existing.get('poster_path') or ''
                    if ex_mid < 900000000 and ex_poster:
                        if (ex_rating == rating or (ex_rating is None and rating is None)) and (ex_date == watched_date or not watched_date):
                            continue

                page_new_count += 1
                new_or_updated.append({
                    'slug': slug, 'title': title, 'year_hint': year,
                    'rating': rating, 'watched_date': watched_date
                })

            if page_new_count == 0 and len(rows) > 0:
                break
            if len(rows) < 50:
                break

        # Fallback to RSS if HTML scraping returned no entries and nothing was visited
        if not new_or_updated and not seen_slugs:
            try:
                import xml.etree.ElementTree as ET
                rss_url = "{0}/{1}/rss/".format(LB_BASE, username)
                r_resp = session.get(rss_url, timeout=12)
                if r_resp.status_code == 200:
                    root = ET.fromstring(r_resp.content)
                    ns = {'letterboxd': 'https://letterboxd.com', 'tmdb': 'https://www.themoviedb.org'}
                    for item in root.findall('./channel/item'):
                        title_elem = item.find('letterboxd:filmTitle', ns)
                        year_elem = item.find('letterboxd:filmYear', ns)
                        rating_elem = item.find('letterboxd:memberRating', ns)
                        date_elem = item.find('letterboxd:watchedDate', ns)
                        link_elem = item.find('link')

                        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''
                        if not title:
                            raw_t = item.find('title')
                            if raw_t is not None and raw_t.text:
                                m = re.match(r'^(.*?),\s*(\d{4})?\s*-\s*([★½]+)?', raw_t.text)
                                if m: title = m.group(1).strip()
                        if not title: continue

                        year = year_elem.text.strip() if year_elem is not None and year_elem.text else ''
                        rating = None
                        if rating_elem is not None and rating_elem.text:
                            try: rating = float(rating_elem.text.strip())
                            except: rating = None
                        date = date_elem.text.strip() if date_elem is not None and date_elem.text else ''
                        link = link_elem.text.strip() if link_elem is not None and link_elem.text else ''
                        slug = link.rstrip('/').split('/')[-1] if link else title_normalize(title)

                        existing = existing_map.get(f"slug_{slug.lower()}") or existing_map.get(f"title_{title.lower()}")
                        if existing:
                            ex_rating = existing.get('rating')
                            ex_date = existing.get('watched_date') or ''
                            ex_mid = int(existing.get('movie_id') or 0)
                            ex_poster = existing.get('poster_path') or ''
                            if ex_mid < 900000000 and ex_poster:
                                if (ex_rating == rating or (ex_rating is None and rating is None)) and (ex_date == date or not date):
                                    continue

                        if slug not in seen_slugs:
                            seen_slugs.add(slug)
                            new_or_updated.append({
                                'slug': slug, 'title': title, 'year_hint': year,
                                'rating': rating, 'watched_date': date
                            })
            except Exception:
                pass

        if not new_or_updated:
            _update_job(job_id, 100, 'completed', 'Your diary is already up to date (no new entries found).',
                        status='completed', diary_count=0)
            return

        _update_job(job_id, 45, 'resolving', 'Resolving {0} new diary films on TMDB...'.format(len(new_or_updated)))
        movie_records, slug_to_id = resolve_entries(
            new_or_updated, active_tmdb, job_id=job_id,
            base_progress=45, span=40, label='new diary films'
        )

        diary_links = []
        for e in new_or_updated:
            mid = slug_to_id.get(e['slug'])
            if not mid:
                continue
            diary_links.append({
                'movie_id': mid,
                'rating': e.get('rating'),
                'watched_date': e.get('watched_date') or ''
            })

        _update_job(job_id, 90, 'saving', 'Saving {0} new diary entries...'.format(len(diary_links)))
        if movie_records:
            upsert_movies_batch(movie_records)
        if diary_links:
            upsert_user_diary(user['id'], diary_links)

        cleanup_database_duplicates(user['id'])
        invalidate_user_cache(user['id'])

        _update_job(
            job_id, 100, 'completed',
            'Done - Synced {0} new diary entries.'.format(len(diary_links)),
            status='completed', diary_count=len(diary_links)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job(job_id, 100, 'failed', 'Diary sync failed: {0}'.format(e),
                    status='failed', error=str(e))


def _run_onboarding_pipeline(job_id, username, pin, tmdb_key, gemini_key, skip_scrape=False, favorites=None):
    try:
        user = get_or_create_user(username, pin=pin, tmdb_key=tmdb_key, gemini_key=gemini_key)
        if not user:
            _update_job(job_id, 100, 'error', 'Failed to create user record',
                        status='failed', error='User creation failed')
            return

        active_tmdb = tmdb_key or (user.get('tmdb_key') if user else '') or TMDB_KEY

        _update_job(job_id, 5, 'auth', 'Setting up profile for @{0}...'.format(username))
        
        diary_links = []
        wl_links = []

        if not skip_scrape:
            session = get_scrape_session()

            # 1. Diary - every page, not just the 50-entry RSS window
            _update_job(job_id, 10, 'diary_scrape',
                        'Reading full Letterboxd diary for @{0}...'.format(username))

            def page_progress(page, count):
                _update_job(job_id, min(10 + page, 28), 'diary_scrape',
                            'Reading diary page {0} ({1} films so far)...'.format(page, count))

            diary_entries = scrape_letterboxd_diary(username, session=session, on_page=page_progress)
            _update_job(job_id, 30, 'diary_scrape',
                        'Found {0} diary entries.'.format(len(diary_entries)),
                        diary_count=len(diary_entries))

            # 2. Resolve diary films against TMDB
            movie_records, slug_to_id = resolve_entries(
                diary_entries, active_tmdb, job_id=job_id,
                base_progress=30, span=30, label='diary films'
            )

            for e in diary_entries:
                mid = slug_to_id.get(e['slug'])
                if not mid:
                    continue
                diary_links.append({
                    'movie_id': mid,
                    'rating': e.get('rating'),
                    'watched_date': e.get('watched_date') or ''
                })

            _update_job(job_id, 62, 'saving',
                        'Saving {0} diary films to the database...'.format(len(diary_links)))
            if movie_records:
                upsert_movies_batch(movie_records)
            if diary_links:
                upsert_user_diary(user['id'], diary_links)

            # 3. Watchlist
            _update_job(job_id, 68, 'watchlist', 'Reading watchlist for @{0}...'.format(username))
            wl_entries = scrape_letterboxd_watchlist(username, session=session)
            _update_job(job_id, 72, 'watchlist',
                        'Found {0} watchlist films.'.format(len(wl_entries)),
                        watchlist_count=len(wl_entries))

            if wl_entries:
                wl_records, wl_slug_to_id = resolve_entries(
                    wl_entries, active_tmdb, job_id=job_id,
                    base_progress=72, span=16, label='watchlist films'
                )
                if wl_records:
                    upsert_movies_batch(wl_records)
                today = pd.Timestamp.now().strftime('%Y-%m-%d')
                for e in wl_entries:
                    mid = wl_slug_to_id.get(e['slug'])
                    if mid:
                        wl_links.append({'movie_id': mid, 'added_date': today})
                if wl_links:
                    upsert_user_watchlist(user['id'], wl_links)
        
        # 4. Handle favorites directly if provided
        if favorites:
            _update_job(job_id, 80, 'favorites', 'Saving selected favorite films...')
            fav_movies = []
            fav_diary = []
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            for f in favorites:
                movie_id = int(f.get('movie_id') or f.get('id') or 0)
                if not movie_id:
                    continue
                fav_movies.append({
                    'movie_id': movie_id,
                    'title': f.get('title', 'Untitled'),
                    'year': str(f.get('year', '')).split('-')[0],
                    'genres': f.get('genres', ''),
                    'overview': f.get('overview', ''),
                    'director': f.get('director', ''),
                    'cast': f.get('cast', ''),
                    'runtime': int(f.get('runtime', 0)),
                    'vote_average': float(f.get('vote_average', 7.0)),
                    'poster_path': f.get('poster_path', ''),
                    'backdrop_path': f.get('backdrop_path', '')
                })
                fav_diary.append({
                    'movie_id': movie_id,
                    'rating': float(f.get('rating') if f.get('rating') is not None else 5.0),
                    'watched_date': today
                })
            if fav_movies:
                upsert_movies_batch(fav_movies)
            if fav_diary:
                upsert_user_diary(user['id'], fav_diary)
                diary_links.extend(fav_diary)

        # Clean up any potential duplicates & invalidate cache
        cleanup_database_duplicates(user['id'])
        invalidate_user_cache(user['id'])

        # 5. Train the in-memory taste model
        _update_job(job_id, 92, 'ai_training', 'Calibrating your personal AI taste model...')
        model, _cols, _vec, _enc = train_user_model_in_memory(username)
        model_note = 'AI model calibrated' if model is not None else 'Not enough rated films to train yet'

        _update_job(
            job_id, 100, 'completed',
            'Done - {0} diary films, {1} watchlist films. {2}.'.format(
                len(diary_links), len(wl_links), model_note),
            status='completed', diary_count=len(diary_links), watchlist_count=len(wl_links)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job(job_id, 100, 'failed', 'Sync failed: {0}'.format(e),
                    status='failed', error=str(e))
