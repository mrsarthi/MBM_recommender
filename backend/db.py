import os
import hashlib
import json
import time
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
from backend.config import DATABASE_URL

# Threaded connection pool
_pool = None
_pool_lock = __import__('threading').Lock()

# Fast in-memory query cache for instantaneous UI responses (< 0.1ms)
_query_cache = {}
_CACHE_TTL = 86400.0

def _get_cache(key):
    entry = _query_cache.get(key)
    if entry and (time.time() - entry['time'] < _CACHE_TTL):
        return entry['data']
    return None

def _set_cache(key, data):
    _query_cache[key] = {'data': data, 'time': time.time()}

def invalidate_user_cache(user_id_or_name=""):
    global _query_cache
    if not user_id_or_name:
        _query_cache.clear()
        return

    # Resolve to user_id if it's a username string
    user_id = None
    if isinstance(user_id_or_name, int):
        user_id = user_id_or_name
    elif str(user_id_or_name).isdigit():
        user_id = int(user_id_or_name)
    else:
        try:
            user = get_user(str(user_id_or_name))
            if user:
                user_id = user['id']
        except Exception:
            pass

    if user_id is not None:
        prefix_wl = f"wl_{user_id}"
        prefix_diary = f"diary_{user_id}"
        to_delete = [k for k in _query_cache if k == prefix_wl or k.startswith(f"{prefix_diary}_")]
        for k in to_delete:
            _query_cache.pop(k, None)
    else:
        # Fallback to string matching
        u_str = str(user_id_or_name).lower()
        to_delete = [k for k in _query_cache if u_str in k.lower()]
        for k in to_delete:
            _query_cache.pop(k, None)

def get_db_pool():
    global _pool
    with _pool_lock:
        if _pool is None or _pool.closed:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL is not set in environment or .env")
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=10, dsn=DATABASE_URL,
                connect_timeout=10,
                keepalives=1, keepalives_idle=30,
                keepalives_interval=10, keepalives_count=3,
            )
        return _pool

def _reset_pool():
    """Drops the whole pool. Used when Neon has suspended and every socket is dead."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try: _pool.closeall()
            except Exception: pass
        _pool = None

def get_connection():
    """
    Checks out a *live* connection.

    Neon's free tier auto-suspends after ~5 minutes idle and silently kills every
    open socket, so a pooled connection is very often dead on arrival. We probe it
    with SELECT 1 and transparently replace it if the probe fails.
    """
    last_err = None
    for attempt in range(3):
        conn = None
        try:
            conn = get_db_pool().getconn()
            if conn.closed:
                raise psycopg2.InterfaceError("connection already closed")
            # Clear any aborted transaction left behind by a previous failure.
            if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                conn.rollback()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
            last_err = e
            if conn is not None:
                try: get_db_pool().putconn(conn, close=True)
                except Exception: pass
            if attempt == 1:
                _reset_pool()
    raise last_err if last_err else RuntimeError("Could not obtain a database connection")

def release_connection(conn):
    if conn is None:
        return
    try:
        p = get_db_pool()
    except Exception:
        return
    try:
        # Never return a connection sitting in a failed transaction to the pool.
        broken = conn.closed or conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR
        p.putconn(conn, close=bool(broken))
    except Exception:
        try: p.putconn(conn, close=True)
        except Exception: pass

def stable_movie_id(seed: str) -> int:
    """
    Deterministic pseudo-TMDB id for films we could not resolve on TMDB.

    Must NOT use hash(): Python randomises string hashing per process, so the same
    film would get a different id on every restart and pile up duplicate rows.
    Offset far above TMDB's real id range so it can never collide with a real film.
    """
    digest = hashlib.md5(str(seed or '').strip().lower().encode('utf-8')).hexdigest()
    return 900000000 + (int(digest[:12], 16) % 99000000)

def hash_pin(pin: str) -> str:
    if not pin:
        return ""
    return hashlib.sha256(f"mbmr_salt_{pin.strip()}".encode('utf-8')).hexdigest()

def init_db():
    """Initializes the Neon PostgreSQL database schema."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    pin_hash VARCHAR(255),
                    tmdb_key VARCHAR(255),
                    gemini_key VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    year VARCHAR(20),
                    genres TEXT,
                    overview TEXT,
                    director VARCHAR(255),
                    cast_members TEXT,
                    keywords TEXT,
                    runtime INTEGER DEFAULT 0,
                    vote_average NUMERIC(4, 1) DEFAULT 7.0,
                    poster_path VARCHAR(255),
                    backdrop_path VARCHAR(255),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS user_diary (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    movie_id INTEGER REFERENCES movies(movie_id) ON DELETE CASCADE,
                    rating NUMERIC(3, 1),
                    watched_date VARCHAR(30),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(user_id, movie_id)
                );

                CREATE TABLE IF NOT EXISTS user_watchlist (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    movie_id INTEGER REFERENCES movies(movie_id) ON DELETE CASCADE,
                    added_date VARCHAR(30),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(user_id, movie_id)
                );

                ALTER TABLE movies ADD COLUMN IF NOT EXISTS letterboxd_slug VARCHAR(255);

                CREATE INDEX IF NOT EXISTS idx_movies_slug ON movies(letterboxd_slug);
                CREATE INDEX IF NOT EXISTS idx_user_diary_user_id ON user_diary(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_watchlist_user_id ON user_watchlist(user_id);
                CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
            """)
            conn.commit()
            print("[OK] Neon PostgreSQL schema verified.")
    finally:
        release_connection(conn)

# ── User Operations ──

def get_user(username: str):
    """
    Read-only user lookup. Returns None if the user does not exist.

    Read paths must use this rather than get_or_create_user(), which would insert a
    row for every username that ever appears in a query string.
    """
    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user or clean_user == 'guest':
        return None
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (clean_user,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        release_connection(conn)


def get_or_create_user(username: str, pin: str = None, tmdb_key: str = None, gemini_key: str = None):
    clean_user = username.strip().lstrip('@').lower()
    if not clean_user:
        return None
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (clean_user,))
            user = cur.fetchone()
            if user:
                # Update keys or PIN if provided
                updates = []
                params = []
                if pin:
                    updates.append("pin_hash = %s")
                    params.append(hash_pin(pin))
                if tmdb_key:
                    updates.append("tmdb_key = %s")
                    params.append(tmdb_key.strip())
                if gemini_key:
                    updates.append("gemini_key = %s")
                    params.append(gemini_key.strip())
                
                if updates:
                    updates.append("updated_at = NOW()")
                    params.append(clean_user)
                    cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE username = %s RETURNING *", params)
                    updated = cur.fetchone()
                    conn.commit()
                    if updated:
                        user = updated
                return dict(user)
            else:
                pin_h = hash_pin(pin) if pin else ""
                cur.execute("""
                    INSERT INTO users (username, pin_hash, tmdb_key, gemini_key)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *
                """, (clean_user, pin_h, (tmdb_key or '').strip(), (gemini_key or '').strip()))
                conn.commit()
                return dict(cur.fetchone())
    finally:
        release_connection(conn)

def verify_user_pin(username: str, pin: str):
    clean_user = username.strip().lstrip('@').lower()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (clean_user,))
            user = cur.fetchone()
            if not user:
                return False, "User not found", None
            if not user.get('pin_hash'):
                # User has no PIN set yet
                return True, "No PIN set", dict(user)
            if user.get('pin_hash') == hash_pin(pin):
                return True, "Authenticated", dict(user)
            return False, "Invalid PIN", None
    finally:
        release_connection(conn)

# ── Shared Movies Operations ──

def upsert_movies_batch(movies_list):
    """
    Batch upserts metadata for movies into the shared movies table.
    movies_list: list of dicts with keys (movie_id, title, year, genres, overview, director, cast, keywords, runtime, vote_average, poster_path, backdrop_path)
    """
    if not movies_list:
        return
    
    deduped_records = {}
    for m in movies_list:
        m_id = m.get('movie_id') or m.get('id')
        if not m_id:
            continue
        try:
            m_id = int(float(m_id))
        except (ValueError, TypeError):
            continue

        title = str(m.get('title') or m.get('Name') or 'Untitled').strip()
        if title.lower() == 'nan':
            title = 'Untitled'

        year = str(m.get('year') or m.get('Year') or m.get('release_date') or '').split('-')[0].replace('.0', '').strip()
        if year.lower() == 'nan':
            year = ''

        genres = m.get('genres', '')
        if isinstance(genres, list):
            genres_str = ", ".join([g['name'] if isinstance(g, dict) else str(g) for g in genres])
        else:
            genres_str = str(genres) if str(genres).lower() != 'nan' else 'General'

        overview = str(m.get('overview') or '') if str(m.get('overview') or '').lower() != 'nan' else ''
        director = str(m.get('director') or '') if str(m.get('director') or '').lower() != 'nan' else ''
        cast = str(m.get('cast') or m.get('cast_members') or '') if str(m.get('cast') or '').lower() != 'nan' else ''
        keywords = str(m.get('keywords') or '') if str(m.get('keywords') or '').lower() != 'nan' else ''
        
        try: runtime = int(float(m.get('runtime', 0)))
        except: runtime = 0
        
        try: vote_avg = float(m.get('vote_average', 7.0))
        except: vote_avg = 7.0
        
        poster = str(m.get('poster_path') or '') if str(m.get('poster_path') or '').lower() != 'nan' else ''
        backdrop = str(m.get('backdrop_path') or '') if str(m.get('backdrop_path') or '').lower() != 'nan' else ''

        slug = str(m.get('letterboxd_slug') or m.get('slug') or '').strip()
        if slug.lower() == 'nan':
            slug = ''

        deduped_records[m_id] = (
            m_id, title, year, genres_str, overview, director, cast, keywords,
            runtime, vote_avg, poster, backdrop, slug
        )

    records = list(deduped_records.values())
    if not records:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO movies (
                    movie_id, title, year, genres, overview, director, cast_members, keywords,
                    runtime, vote_average, poster_path, backdrop_path, letterboxd_slug
                ) VALUES %s
                ON CONFLICT (movie_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    year = COALESCE(NULLIF(EXCLUDED.year, ''), movies.year),
                    genres = COALESCE(NULLIF(EXCLUDED.genres, ''), movies.genres),
                    overview = COALESCE(NULLIF(EXCLUDED.overview, ''), movies.overview),
                    director = COALESCE(NULLIF(EXCLUDED.director, ''), movies.director),
                    cast_members = COALESCE(NULLIF(EXCLUDED.cast_members, ''), movies.cast_members),
                    keywords = COALESCE(NULLIF(EXCLUDED.keywords, ''), movies.keywords),
                    runtime = CASE WHEN EXCLUDED.runtime > 0 THEN EXCLUDED.runtime ELSE movies.runtime END,
                    vote_average = EXCLUDED.vote_average,
                    poster_path = COALESCE(NULLIF(EXCLUDED.poster_path, ''), movies.poster_path),
                    backdrop_path = COALESCE(NULLIF(EXCLUDED.backdrop_path, ''), movies.backdrop_path),
                    letterboxd_slug = COALESCE(NULLIF(EXCLUDED.letterboxd_slug, ''), movies.letterboxd_slug),
                    updated_at = NOW()
            """, records)
            conn.commit()
    finally:
        release_connection(conn)

def get_movie_ids_by_slugs(slugs):
    """
    Maps Letterboxd slugs to already-known TMDB ids.

    This is what makes a re-sync cheap: any film already resolved by this user - or by
    any other user, since `movies` is shared - never hits the TMDB search API again.
    """
    clean = [str(s).strip() for s in (slugs or []) if str(s or '').strip()]
    if not clean:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT letterboxd_slug, movie_id FROM movies "
                "WHERE letterboxd_slug = ANY(%s) AND letterboxd_slug IS NOT NULL",
                (clean,)
            )
            return {r[0]: r[1] for r in cur.fetchall()}
    finally:
        release_connection(conn)

def get_existing_movie_ids(movie_ids):
    """Returns set of movie_ids that already exist in the shared movies table."""
    if not movie_ids:
        return set()
    clean_ids = [int(x) for x in movie_ids if x and str(x).isdigit()]
    if not clean_ids:
        return set()
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT movie_id FROM movies WHERE movie_id = ANY(%s)", (clean_ids,))
            return {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)

# ── User Diary Operations ──

def upsert_user_diary(user_id: int, diary_entries):
    """
    diary_entries: list of dicts with keys (movie_id, rating, watched_date)
    """
    if not diary_entries or not user_id:
        return 0
    
    deduped = {}
    for d in diary_entries:
        m_id = d.get('movie_id') or d.get('id')
        if not m_id:
            continue
        try:
            m_id = int(float(m_id))
        except:
            continue
        
        # An unrated film must stay NULL. Defaulting it to 3.5 invents a rating the
        # user never gave and then trains the taste model on it.
        raw_rating = d.get('Rating') if d.get('Rating') is not None else d.get('rating')
        if raw_rating is None or str(raw_rating).strip().lower() in ('', 'nan', 'none'):
            rating = None
        else:
            try:
                rating = float(raw_rating)
            except (TypeError, ValueError):
                rating = None

        date = str(d.get('Date') or d.get('watched_date') or '')
        if date.lower() == 'nan':
            date = ''
        # watched_date is VARCHAR(30); a malformed cell must not abort the whole batch.
        date = date[:30]

        deduped[m_id] = (user_id, m_id, rating, date)

    records = list(deduped.values())
    if not records:
        return 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO user_diary (user_id, movie_id, rating, watched_date)
                VALUES %s
                ON CONFLICT (user_id, movie_id) DO UPDATE SET
                    rating = COALESCE(EXCLUDED.rating, user_diary.rating),
                    watched_date = COALESCE(NULLIF(EXCLUDED.watched_date, ''), user_diary.watched_date)
            """, records)
            conn.commit()
            invalidate_user_cache(user_id)
            return len(records)
    finally:
        release_connection(conn)

def get_user_diary(username: str, search: str = '', rating_filter: str = 'All', year_filter: str = 'All', sort_mode: str = 'Newest Log First'):
    user = get_user(username)
    if not user:
        return [], 0, 0.0

    cache_key = f"diary_{user['id']}_{search}_{rating_filter}_{year_filter}_{sort_mode}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT 
                    m.movie_id, m.movie_id AS id, m.title, m.year, m.genres, m.overview,
                    m.director, m.cast_members AS cast, m.runtime, m.vote_average,
                    m.poster_path, m.backdrop_path,
                    d.rating AS "Rating", d.rating, d.watched_date AS "Date", d.watched_date
                FROM user_diary d
                JOIN movies m ON d.movie_id = m.movie_id
                WHERE d.user_id = %s
            """
            params = [user['id']]

            if rating_filter != 'All':
                try:
                    r_val = float(rating_filter.replace('★', '').strip())
                    query += " AND d.rating = %s"
                    params.append(r_val)
                except:
                    pass

            if year_filter != 'All':
                query += " AND m.year = %s"
                params.append(str(year_filter))

            if search:
                query += " AND (m.title ILIKE %s OR m.director ILIKE %s OR m.genres ILIKE %s)"
                s_param = f"%{search}%"
                params.extend([s_param, s_param, s_param])

            if sort_mode == 'Newest Log First':
                query += " ORDER BY d.watched_date DESC NULLS LAST, d.id DESC"
            elif sort_mode == 'Highest Rating First':
                query += " ORDER BY d.rating DESC, d.watched_date DESC NULLS LAST"
            elif sort_mode == 'Lowest Rating First':
                query += " ORDER BY d.rating ASC, d.watched_date DESC NULLS LAST"
            elif sort_mode == 'Release Year (Newest)':
                query += " ORDER BY m.year DESC NULLS LAST"
            elif sort_mode == 'Release Year (Oldest)':
                query += " ORDER BY m.year ASC NULLS LAST"
            else:
                query += " ORDER BY d.watched_date DESC NULLS LAST"

            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            total_count = len(rows)
            ratings = [float(r['rating']) for r in rows if r.get('rating') is not None]
            avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
            res = (rows, total_count, avg_rating)
            _set_cache(cache_key, res)
            return res
    finally:
        release_connection(conn)

# ── User Watchlist Operations ──

def upsert_user_watchlist(user_id: int, watchlist_entries):
    if not watchlist_entries or not user_id:
        return 0
    
    deduped = {}
    for w in watchlist_entries:
        m_id = w.get('movie_id') or w.get('id')
        if not m_id:
            continue
        try:
            m_id = int(float(m_id))
        except:
            continue
        date = str(w.get('added_date') or pd.Timestamp.now().strftime('%Y-%m-%d'))[:30]
        deduped[m_id] = (user_id, m_id, date)

    records = list(deduped.values())
    if not records:
        return 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO user_watchlist (user_id, movie_id, added_date)
                VALUES %s
                ON CONFLICT (user_id, movie_id) DO UPDATE SET
                    added_date = EXCLUDED.added_date
            """, records)
            conn.commit()
            invalidate_user_cache(user_id)
            return len(records)
    finally:
        release_connection(conn)

def get_user_watchlist(username: str):
    user = get_user(username)
    if not user:
        return []

    cache_key = f"wl_{user['id']}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    m.movie_id, m.movie_id AS id, m.title, m.year, m.genres, m.overview,
                    m.director, m.cast_members AS cast, m.runtime, m.vote_average,
                    m.poster_path, m.backdrop_path, w.added_date
                FROM user_watchlist w
                JOIN movies m ON w.movie_id = m.movie_id
                WHERE w.user_id = %s
                ORDER BY w.id DESC
            """, (user['id'],))
            rows = [dict(r) for r in cur.fetchall()]
            _set_cache(cache_key, rows)
            return rows
    finally:
        release_connection(conn)

def add_to_user_watchlist(username: str, movie_data: dict):
    user = get_or_create_user(username)
    if not user:
        return False, "User not found"

    upsert_movies_batch([movie_data])
    m_id = int(movie_data.get('movie_id') or movie_data.get('id'))
    upsert_user_watchlist(user['id'], [{'movie_id': m_id, 'added_date': pd.Timestamp.now().strftime('%Y-%m-%d')}])
    invalidate_user_cache(user['id'])
    return True, f"Added to Watchlist!"

def remove_from_user_watchlist(username: str, movie_id: int):
    user = get_user(username)
    if not user or not movie_id:
        return False, "User or movie not found"
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_watchlist WHERE user_id = %s AND movie_id = %s", (user['id'], int(movie_id)))
            conn.commit()
            invalidate_user_cache(user['id'])
            return True, "Removed from Watchlist."
    finally:
        release_connection(conn)

def get_diary_training_df(username: str):
    """
    Extracts user diary merged with full shared movies metadata for in-memory AI training.
    """
    user = get_user(username)
    if not user:
        return pd.DataFrame()
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    m.movie_id, m.title, m.title AS "Name", m.year, m.year AS "Year", m.genres, m.overview,
                    m.director, m.cast_members AS cast, m.keywords, m.runtime,
                    d.rating, d.rating AS "Rating", d.watched_date, d.watched_date AS "Date"
                FROM user_diary d
                JOIN movies m ON d.movie_id = m.movie_id
                WHERE d.user_id = %s
                ORDER BY d.watched_date DESC NULLS LAST
            """, (user['id'],))
            rows = [dict(r) for r in cur.fetchall()]
            return pd.DataFrame(rows)
    finally:
        release_connection(conn)
