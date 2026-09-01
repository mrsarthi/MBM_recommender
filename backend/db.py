import os
import re
import hashlib
import json
import time
import base64
import secrets
from cryptography.fernet import Fernet
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
from backend.config import DATABASE_URL, ENCRYPTION_KEY

def get_encryption_fernet():
    key = ENCRYPTION_KEY
    return Fernet(key.encode('utf-8'))

def encrypt_key(val: str) -> str:
    if not val:
        return ""
    try:
        f = get_encryption_fernet()
        return f.encrypt(val.strip().encode('utf-8')).decode('utf-8')
    except Exception as e:
        print(f"[WARN] Key encryption failed: {e}")
        return val

def decrypt_key(val: str) -> str:
    if not val:
        return ""
    if not val.startswith('gAAAAA'):
        return val
    try:
        f = get_encryption_fernet()
        return f.decrypt(val.encode('utf-8')).decode('utf-8')
    except Exception as e:
        print(f"[WARN] Key decryption failed: {e}")
        return ""

# Threaded connection pool
_pool = None
_pool_lock = __import__('threading').Lock()

# Fast in-memory query cache for rapid UI responses (< 0.1ms) with 15s freshness TTL
_query_cache = {}
_CACHE_TTL = 15.0

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
    salt = secrets.token_bytes(16)
    iterations = 100000
    h = hashlib.pbkdf2_hmac('sha256', pin.strip().encode('utf-8'), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${h.hex()}"

def verify_pin(pin: str, stored_hash: str) -> bool:
    if not pin or not stored_hash:
        return False
    if not stored_hash.startswith("pbkdf2_sha256$"):
        legacy_hash = hashlib.sha256(f"mbmr_salt_{pin.strip()}".encode('utf-8')).hexdigest()
        return legacy_hash == stored_hash
    try:
        parts = stored_hash.split('$')
        if len(parts) != 4:
            return False
        _, iterations_str, salt_hex, hash_hex = parts
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        computed_hash = hashlib.pbkdf2_hmac('sha256', pin.strip().encode('utf-8'), salt, iterations)
        return secrets.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False

def cleanup_database_duplicates(user_id: int = None):
    """
    Cleans up duplicate records across movies, user_diary, and user_watchlist:
    1. If placeholder movies (movie_id >= 900000000) have a matching real TMDB movie with the same letterboxd_slug,
       migrate user_diary and user_watchlist foreign keys to the real movie_id and delete the placeholder.
    2. Removes any movies from user_watchlist that already exist in user_diary for the same user.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 1. Find placeholder movies that match a real TMDB movie on letterboxd_slug or (title + year)
            cur.execute("""
                SELECT p.movie_id AS placeholder_id, r.movie_id AS real_id
                FROM movies p
                JOIN movies r ON (
                    (p.letterboxd_slug = r.letterboxd_slug AND p.letterboxd_slug IS NOT NULL AND p.letterboxd_slug != '')
                    OR (LOWER(p.title) = LOWER(r.title) AND (p.year = r.year OR p.year = '' OR r.year = ''))
                )
                WHERE p.movie_id >= 900000000 AND r.movie_id < 900000000
            """)
            replacements = cur.fetchall()
            for ph_id, real_id in replacements:
                # Update user_diary references safely
                cur.execute("""
                    UPDATE user_diary SET movie_id = %s
                    WHERE movie_id = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM user_diary d2 WHERE d2.user_id = user_diary.user_id AND d2.movie_id = %s
                    )
                """, (real_id, ph_id, real_id))
                cur.execute("DELETE FROM user_diary WHERE movie_id = %s", (ph_id,))

                # Update user_watchlist references safely
                cur.execute("""
                    UPDATE user_watchlist SET movie_id = %s
                    WHERE movie_id = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM user_watchlist w2 WHERE w2.user_id = user_watchlist.user_id AND w2.movie_id = %s
                    )
                """, (real_id, ph_id, real_id))
                cur.execute("DELETE FROM user_watchlist WHERE movie_id = %s", (ph_id,))

                # Delete the redundant placeholder movie row
                cur.execute("DELETE FROM movies WHERE movie_id = %s", (ph_id,))

            # 2. Remove watchlist items that are already logged in diary for the same user
            if user_id:
                cur.execute("""
                    DELETE FROM user_watchlist w
                    WHERE w.user_id = %s
                    AND EXISTS (
                        SELECT 1 FROM user_diary d WHERE d.user_id = w.user_id AND d.movie_id = w.movie_id
                    )
                """, (user_id,))
            else:
                cur.execute("""
                    DELETE FROM user_watchlist w
                    WHERE EXISTS (
                        SELECT 1 FROM user_diary d WHERE d.user_id = w.user_id AND d.movie_id = w.movie_id
                    )
                """)

            conn.commit()
            if user_id:
                invalidate_user_cache(user_id)
            else:
                invalidate_user_cache()
    except Exception as e:
        print(f"[WARN] cleanup_database_duplicates: {e}")
        try: conn.rollback()
        except Exception: pass
    finally:
        release_connection(conn)

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
                ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_attempts INT DEFAULT 0;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP WITH TIME ZONE NULL;

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
            cur.execute("SELECT id, username, pin_hash, tmdb_key, gemini_key FROM users WHERE username = %s", (clean_user,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                d['tmdb_key'] = decrypt_key(d.get('tmdb_key', ''))
                d['gemini_key'] = decrypt_key(d.get('gemini_key', ''))
                return d
            return None
    finally:
        release_connection(conn)


def get_or_create_user(username: str, pin: str = None, tmdb_key: str = None, gemini_key: str = None):
    clean_user = username.strip().lstrip('@').lower()
    if not clean_user:
        return None
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, username, pin_hash, tmdb_key, gemini_key FROM users WHERE username = %s", (clean_user,))
            user = cur.fetchone()
            if user:
                # Security: If account already has a PIN hash set, DO NOT allow overwriting it blindly
                existing_pin_hash = user.get('pin_hash')
                updates = []
                params = []
                if pin and not existing_pin_hash:
                    updates.append("pin_hash = %s")
                    params.append(hash_pin(pin))
                if tmdb_key:
                    updates.append("tmdb_key = %s")
                    params.append(encrypt_key(tmdb_key))
                if gemini_key:
                    updates.append("gemini_key = %s")
                    params.append(encrypt_key(gemini_key))
                
                if updates:
                    updates.append("updated_at = NOW()")
                    params.append(clean_user)
                    cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE username = %s RETURNING id, username, pin_hash, tmdb_key, gemini_key", params)
                    updated = cur.fetchone()
                    conn.commit()
                    if updated:
                        user = updated
                d = dict(user)
                d['tmdb_key'] = decrypt_key(d.get('tmdb_key', ''))
                d['gemini_key'] = decrypt_key(d.get('gemini_key', ''))
                return d
            else:
                pin_h = hash_pin(pin) if pin else ""
                tmdb_enc = encrypt_key(tmdb_key) if tmdb_key else ""
                gemini_enc = encrypt_key(gemini_key) if gemini_key else ""
                cur.execute("""
                    INSERT INTO users (username, pin_hash, tmdb_key, gemini_key)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, username, pin_hash, tmdb_key, gemini_key
                """, (clean_user, pin_h, tmdb_enc, gemini_enc))
                conn.commit()
                d = dict(cur.fetchone())
                d['tmdb_key'] = decrypt_key(d.get('tmdb_key', ''))
                d['gemini_key'] = decrypt_key(d.get('gemini_key', ''))
                return d
    finally:
        release_connection(conn)

def verify_user_pin(username: str, pin: str):
    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user:
        return False, "Invalid credentials", None
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, username, pin_hash, tmdb_key, gemini_key, failed_attempts, locked_until 
                FROM users 
                WHERE username = %s
            """, (clean_user,))
            user = cur.fetchone()
            if not user:
                return False, "Invalid credentials", None
            
            # Check lockout
            locked_until = user.get('locked_until')
            if locked_until:
                cur.execute("SELECT NOW() < %s AS is_locked, EXTRACT(EPOCH FROM (%s - NOW()))::INT AS wait_sec", (locked_until, locked_until))
                lock_res = cur.fetchone()
                if lock_res and lock_res.get('is_locked'):
                    wait_sec = max(1, lock_res.get('wait_sec', 900))
                    return False, f"Account temporarily locked. Try again in {wait_sec}s.", None
            
            stored_hash = user.get('pin_hash')
            # Strict: reject empty pin_hash or incorrect PIN
            if not stored_hash or not pin or not verify_pin(pin, stored_hash):
                # Increment failed attempts
                cur.execute("""
                    UPDATE users 
                    SET failed_attempts = COALESCE(failed_attempts, 0) + 1,
                        locked_until = CASE 
                            WHEN COALESCE(failed_attempts, 0) + 1 >= 5 THEN NOW() + INTERVAL '15 minutes'
                            ELSE locked_until 
                        END,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING failed_attempts, locked_until
                """, (user['id'],))
                updated_lock = cur.fetchone()
                conn.commit()
                if updated_lock and updated_lock.get('failed_attempts', 0) >= 5:
                    return False, "Account locked for 15 minutes due to too many failed attempts.", None
                return False, "Invalid credentials", None
            
            # PIN is valid -> Reset failed attempts and lockout
            if not stored_hash.startswith("pbkdf2_sha256$"):
                new_hash = hash_pin(pin)
                cur.execute("""
                    UPDATE users 
                    SET pin_hash = %s, failed_attempts = 0, locked_until = NULL, updated_at = NOW() 
                    WHERE id = %s
                """, (new_hash, user['id']))
            else:
                cur.execute("""
                    UPDATE users 
                    SET failed_attempts = 0, locked_until = NULL, updated_at = NOW() 
                    WHERE id = %s
                """, (user['id'],))
            conn.commit()

            d = dict(user)
            d['tmdb_key'] = decrypt_key(d.get('tmdb_key', ''))
            d['gemini_key'] = decrypt_key(d.get('gemini_key', ''))
            return True, "Login successful", d
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

    Only returns movies that have a real TMDB id (< 900000000) and non-empty poster_path,
    ensuring that unhydrated placeholders are properly re-resolved against TMDB.
    """
    clean = [str(s).strip() for s in (slugs or []) if str(s or '').strip()]
    if not clean:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT letterboxd_slug, movie_id FROM movies "
                "WHERE letterboxd_slug = ANY(%s) AND letterboxd_slug IS NOT NULL "
                "AND movie_id < 900000000 AND poster_path IS NOT NULL AND poster_path != ''",
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

def get_user_diary_map(user_id: int):
    """
    Returns a dict mapping known slugs, movie_ids, and titles to their diary record:
    {'mid_123': {...}, 'slug_inception': {...}, 'title_inception': {...}}
    for fast O(1) incremental sync diffing.
    """
    if not user_id:
        return {}
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.movie_id, d.rating, d.watched_date, m.letterboxd_slug, m.title, m.year, m.poster_path
                FROM user_diary d
                JOIN movies m ON d.movie_id = m.movie_id
                WHERE d.user_id = %s
            """, (user_id,))
            rows = cur.fetchall()
            diary_map = {}
            for r in rows:
                item = dict(r)
                mid = item['movie_id']
                slug = str(item.get('letterboxd_slug') or '').strip().lower()
                title = str(item.get('title') or '').strip().lower()
                diary_map[f"mid_{mid}"] = item
                if slug:
                    diary_map[f"slug_{slug}"] = item
                if title:
                    diary_map[f"title_{title}"] = item
            return diary_map
    finally:
        release_connection(conn)

# ── User Watchlist Operations ──

def upsert_user_watchlist(user_id: int, watchlist_entries):
    if not watchlist_entries or not user_id:
        return 0
    
    deduped = {}
    for w in watchlist_entries:
        if isinstance(w, (int, str)) and str(w).isdigit():
            m_id = int(w)
            date = pd.Timestamp.now().strftime('%Y-%m-%d')
        elif isinstance(w, dict):
            m_id = w.get('movie_id') or w.get('id')
            if not m_id:
                continue
            try:
                m_id = int(float(m_id))
            except:
                continue
            date = str(w.get('added_date') or pd.Timestamp.now().strftime('%Y-%m-%d'))[:30]
        else:
            continue
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

def get_user_taste_anchors(username: str):
    """
    Extracts a quick, high-signal taste profile (top directors, 5★ favorites, top genres)
    from the user's logged diary in Neon DB for grounding AI prompts.
    """
    user = get_user(username)
    if not user:
        return {'top_directors': [], 'favorite_movies': [], 'top_genres': [], 'preferred_decades': []}

    cache_key = f"anchors_{user['id']}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    records, total, _ = get_user_diary(username, sort_mode='Highest Rating First')
    if not records:
        empty_res = {'top_directors': [], 'favorite_movies': [], 'top_genres': [], 'preferred_decades': []}
        _set_cache(cache_key, empty_res)
        return empty_res

    # 1. Top 5-star / High-rated Favorites
    fav_movies = []
    seen_titles = set()
    for r in records:
        r_score = r.get('Rating') or r.get('rating')
        t = str(r.get('title') or '').strip()
        if r_score and float(r_score) >= 4.0 and t and t.lower() not in seen_titles:
            seen_titles.add(t.lower())
            fav_movies.append(t)
            if len(fav_movies) >= 5:
                break

    # 2. Top Directors
    director_scores = {}
    director_counts = {}
    for r in records:
        r_score = r.get('Rating') or r.get('rating')
        director = str(r.get('director') or '').strip()
        if not director or director.lower() in ('nan', 'none', 'unknown'):
            continue
        # Split in case multiple directors listed
        for d in re.split(r'[,/]', director):
            d_clean = d.strip()
            if not d_clean or len(d_clean) < 3:
                continue
            director_counts[d_clean] = director_counts.get(d_clean, 0) + 1
            if r_score:
                director_scores.setdefault(d_clean, []).append(float(r_score))

    # Prefer directors with multiple logged films and high avg rating
    scored_directors = []
    for d, counts in director_counts.items():
        scores = director_scores.get(d, [3.5])
        avg_score = sum(scores) / len(scores)
        # Weight by count and avg score
        if avg_score >= 3.5:
            scored_directors.append((d, avg_score, counts))

    scored_directors.sort(key=lambda x: (x[1] >= 4.0, x[2], x[1]), reverse=True)
    top_directors = [d[0] for d in scored_directors[:4]]

    # 3. Top Genres
    genre_counts = {}
    for r in records:
        g_str = str(r.get('genres') or '')
        for g in g_str.split(','):
            g_clean = g.strip()
            if g_clean and g_clean.lower() not in ('nan', 'none', 'general'):
                genre_counts[g_clean] = genre_counts.get(g_clean, 0) + 1
    top_genres = [g[0] for g in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:4]]

    # 4. Preferred Decades
    decade_counts = {}
    for r in records:
        y_str = str(r.get('year') or '')[:4]
        if y_str.isdigit() and len(y_str) == 4:
            dec = f"{y_str[:3]}0s"
            decade_counts[dec] = decade_counts.get(dec, 0) + 1
    top_decades = [d[0] for d in sorted(decade_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

    anchors = {
        'top_directors': top_directors,
        'favorite_movies': fav_movies[:4],
        'top_genres': top_genres,
        'preferred_decades': top_decades
    }
    _set_cache(cache_key, anchors)
    return anchors

