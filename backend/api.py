import os
import sys
import json
import threading
import time
import urllib.parse
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import pandas as pd

# IP + Username login rate limiter state (max 5 failures per 15 mins)
_login_failures = defaultdict(list)
_failures_lock = threading.Lock()

def check_login_rate_limit(ip, username):
    now = time.time()
    key = (ip, username.lower())
    with _failures_lock:
        _login_failures[key] = [t for t in _login_failures[key] if now - t < 900]
        if len(_login_failures[key]) >= 5:
            return False, int(900 - (now - _login_failures[key][0]))
    return True, 0

def record_login_failure(ip, username):
    now = time.time()
    key = (ip, username.lower())
    with _failures_lock:
        _login_failures[key].append(now)

def reset_login_failures(ip, username):
    key = (ip, username.lower())
    with _failures_lock:
        _login_failures.pop(key, None)

from backend.config import BASE_DIR, TMDB_KEY, GEMINI_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE, LETTERBOXD_USERNAME
from backend.db import (
    init_db, get_user, get_or_create_user, verify_user_pin, get_user_diary,
    get_user_watchlist, add_to_user_watchlist, remove_from_user_watchlist,
    upsert_movies_batch, upsert_user_diary, get_user_taste_anchors
)
from backend.in_memory_model import get_or_train_user_model, invalidate_user_model, train_user_model_in_memory
from backend.jobs import (
    start_onboarding_job, start_diary_sync_job, start_watchlist_sync_job,
    start_csv_import_job, get_job_status, repair_user_unhydrated_movies
)
from backend.predictions import predict_movie_score, predict_movie_scores_batch, get_post_watch_recommendations, get_watch_providers
from backend.gemini_client import interpret_query_with_ai, generate_matchmaker_pitch
from backend.recommender import analyze, titleNormalize, http_session
from backend.watchlist import get_mood_cluster, pick_movie_for_tonight

# Initialize Neon DB schema on startup
try:
    init_db()
except Exception as e:
    print(f"[WARN] DB initialization warning: {e}")

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MBMRRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        frontend_dir = os.path.join(BASE_DIR, 'frontend')
        super().__init__(*args, directory=frontend_dir, **kwargs)

    def _get_request_user(self, data=None):
        data = data or {}
        if isinstance(data, dict) and 'username' in data:
            val = data.get('username')
            if isinstance(val, list): val = val[0] if val else ''
            val_str = str(val).strip().lstrip('@').lower() if val else ''
            if val_str and val_str != 'guest':
                return val_str
        header_u = self.headers.get('X-Letterboxd-User')
        if header_u is not None:
            clean_h = header_u.strip().lstrip('@').lower()
            if clean_h and clean_h != 'guest':
                return clean_h
            return ''
        return ''

    def _get_request_keys(self, user=None, body=None, query=None, allow_env_fallback=True):
        """
        Retrieves TMDB and Gemini keys in strict priority:
        1. Explicit HTTP Headers (X-TMDB-Key, X-Gemini-Key)
        2. Body / Query parameter (tmdb_key, gemini_key)
        3. Database user profile (users.tmdb_key, users.gemini_key)
        4. Environment variables (config.TMDB_KEY, config.GEMINI_API_KEY) - only if allow_env_fallback is True
        """
        tmdb = (
            self.headers.get('X-TMDB-Key') or 
            (body.get('tmdb_key') if isinstance(body, dict) else '') or
            (query.get('tmdb_key', [''])[0] if isinstance(query, dict) else '') or
            ''
        ).strip()
        
        gemini = (
            self.headers.get('X-Gemini-Key') or 
            (body.get('gemini_key') if isinstance(body, dict) else '') or
            (query.get('gemini_key', [''])[0] if isinstance(query, dict) else '') or
            ''
        ).strip()

        if user and (not tmdb or not gemini):
            try:
                user_obj = get_user(user)
                if user_obj:
                    if not tmdb and user_obj.get('tmdb_key'):
                        tmdb = str(user_obj['tmdb_key']).strip()
                    if not gemini and user_obj.get('gemini_key'):
                        gemini = str(user_obj['gemini_key']).strip()
            except Exception:
                pass

        if allow_env_fallback:
            if not tmdb and TMDB_KEY and TMDB_KEY != 'YOUR_TMDB_API_KEY_HERE':
                tmdb = TMDB_KEY
            if not gemini and GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
                gemini = GEMINI_API_KEY

        return tmdb, gemini

    def _get_allowed_origin(self):
        origin = self.headers.get('Origin')
        if not origin:
            return None
        origin = origin.strip().rstrip('/')
        allowed = [
            'https://mbm-recommender.vercel.app',
            'https://mbmr.onrender.com',
            'http://localhost:8899',
            'http://127.0.0.1:8899',
            'http://localhost:3000',
            'http://127.0.0.1:3000'
        ]
        if origin in allowed:
            return origin
        return None

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        self.send_header('Content-Security-Policy', 
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https://image.tmdb.org https://a.ltrbxd.com; "
            "connect-src 'self' https://api.themoviedb.org https://generativelanguage.googleapis.com;"
        )
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        allowed_origin = self._get_allowed_origin()
        if allowed_origin:
            self.send_header('Access-Control-Allow-Origin', allowed_origin)
            self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-TMDB-Key, X-Gemini-Key, X-Letterboxd-User, X-User-Pin')
        self.end_headers()

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        allowed_origin = self._get_allowed_origin()
        if allowed_origin:
            self.send_header('Access-Control-Allow-Origin', allowed_origin)
            self.send_header('Access-Control-Allow-Credentials', 'true')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/api/status':
            self._handle_status(query)
        elif path == '/api/diary':
            self._handle_diary(query)
        elif path == '/api/watchlist':
            self._handle_get_watchlist(query)
        elif path == '/api/taste_radar':
            self._handle_taste_radar(query)
        elif path == '/api/ripple':
            self._handle_ripple(query)
        elif path == '/api/search_tmdb':
            self._handle_search_tmdb(query)
        elif path == '/api/onboarding/status':
            self._handle_onboarding_status(query)
        elif path.startswith('/api/'):
            self._send_json({'error': 'Endpoint not found'}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length', 0))
        
        # Max size check: 10MB
        if content_length > 10 * 1024 * 1024:
            self._send_json({'error': 'Payload too large'}, 413)
            return

        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            body = json.loads(post_data.decode('utf-8'))
        except Exception:
            body = {}

        if path == '/api/auth/login':
            self._handle_login(body)
        elif path == '/api/onboarding/start':
            self._handle_onboarding_start(body)
        elif path == '/api/recommend':
            self._handle_recommend(body)
        elif path == '/api/watchlist/add':
            self._handle_add_watchlist(body)
        elif path == '/api/watchlist/remove':
            self._handle_remove_watchlist(body)
        elif path == '/api/watchlist/pick_tonight':
            self._handle_pick_tonight(body)
        elif path == '/api/watchlist/sync':
            self._handle_sync_watchlist(body)
        elif path == '/api/log_movie':
            self._handle_log_movie(body)
        elif path == '/api/sync_letterboxd':
            self._handle_sync_letterboxd(body)
        elif path == '/api/retrain':
            self._handle_retrain(body)
        elif path == '/api/import_csv':
            self._handle_import_csv(body)
        elif path == '/api/user/keys':
            self._handle_update_keys(body)
        else:
            self._send_json({'error': 'POST endpoint not found'}, 404)

    # ── Auth & Onboarding Handlers ──

    def _handle_update_keys(self, body):
        user = self._get_request_user(body)
        pin = str(body.get('pin') or '').strip()
        tmdb = (body.get('tmdb_key') or self.headers.get('X-TMDB-Key') or '').strip()
        gemini = (body.get('gemini_key') or self.headers.get('X-Gemini-Key') or '').strip()

        if not user:
            self._send_json({'success': False, 'message': 'Username is required'}, 400)
            return

        # Verify PIN before updating keys
        ok, msg, user_obj = verify_user_pin(user, pin)
        if not ok:
            self._send_json({'success': False, 'message': 'Invalid PIN. Keys cannot be updated.'}, 401)
            return

        u = get_or_create_user(user, pin=pin or None, tmdb_key=tmdb or None, gemini_key=gemini or None)
        if u:
            self._send_json({'success': True, 'message': 'API keys updated successfully', 'user': {
                'username': u['username'],
                'has_tmdb': bool(u.get('tmdb_key')),
                'has_gemini': bool(u.get('gemini_key'))
            }})
        else:
            self._send_json({'success': False, 'message': 'Failed to update user keys'}, 500)

    def _handle_import_csv(self, body):
        user = self._get_request_user(body)
        csv_text = body.get('csv_content', '')
        is_wl = bool(body.get('is_watchlist', False))
        tmdb = (body.get('tmdb_key') or self.headers.get('X-TMDB-Key') or '').strip()
        if not user or not csv_text:
            self._send_json({'success': False, 'message': 'Username and csv_content required'}, 400)
            return

        # Runs in the background and reports through /api/onboarding/status: a full
        # export needs hundreds of TMDB lookups and would otherwise time out.
        job_id = start_csv_import_job(user, csv_text, is_watchlist=is_wl, tmdb_key=tmdb)
        self._send_json({'success': True, 'job_id': job_id, 'message': 'Import started'})

    def _handle_login(self, body):
        username = (body.get('username') or '').strip().lstrip('@').lower()
        pin = str(body.get('pin') or '').strip()
        if not username:
            self._send_json({'success': False, 'message': 'Username is required'}, 400)
            return

        # Resolve IP for rate limiting
        ip = self.headers.get('X-Forwarded-For', self.client_address[0]).split(',')[0].strip()
        allowed, wait_sec = check_login_rate_limit(ip, username)
        if not allowed:
            self._send_json({
                'success': False,
                'message': f'Too many failed attempts. Please wait {wait_sec} seconds before trying again.'
            }, 429)
            return

        ok, msg, user = verify_user_pin(username, pin)
        if ok and user:
            reset_login_failures(ip, username)
            self._send_json({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'username': user['username'],
                    'has_tmdb': bool(user.get('tmdb_key')),
                    'has_gemini': bool(user.get('gemini_key'))
                }
            })
        else:
            record_login_failure(ip, username)
            self._send_json({'success': False, 'message': msg or 'Invalid credentials'}, 401)

    def _handle_onboarding_start(self, body):
        username = (body.get('username') or '').strip().lstrip('@').lower()
        pin = str(body.get('pin') or '').strip()
        tmdb = (body.get('tmdb_key') or self.headers.get('X-TMDB-Key') or '').strip()
        gemini = (body.get('gemini_key') or self.headers.get('X-Gemini-Key') or '').strip()
        skip_scrape = bool(body.get('skip_scrape', False))
        favorites = body.get('favorites', [])

        if not username:
            self._send_json({'success': False, 'message': 'Username is required'}, 400)
            return

        if not pin or len(pin) < 4:
            self._send_json({'success': False, 'message': 'A 4-6 digit PIN is required to secure your profile for multi-device login'}, 400)
            return

        job_id = start_onboarding_job(
            username, pin=pin, tmdb_key=tmdb, gemini_key=gemini,
            skip_scrape=skip_scrape, favorites=favorites
        )
        self._send_json({
            'success': True,
            'job_id': job_id,
            'message': 'Onboarding job started'
        })

    def _handle_onboarding_status(self, query):
        job_id = query.get('job_id', [''])[0]
        if not job_id:
            self._send_json({'error': 'job_id is required'}, 400)
            return
        status_info = get_job_status(job_id)
        self._send_json(status_info)

    # ── User Status & Data Handlers ──

    def _handle_status(self, query=None):
        user = self._get_request_user(query)
        total_films = 0
        avg_rating = 0.0
        watchlist_count = 0
        has_tmdb = False
        has_gemini = False

        if user:
            try:
                _, total_films, avg_rating = get_user_diary(user)
                wl = get_user_watchlist(user)
                watchlist_count = len(wl)
                
                # Fetch keys status
                user_obj = get_user(user)
                if user_obj:
                    has_tmdb = bool(user_obj.get('tmdb_key'))
                    has_gemini = bool(user_obj.get('gemini_key'))
            except Exception as e:
                print(f"Status query error: {e}")

        # Check if user has an in-memory trained model
        ai_model, _, _, _ = get_or_train_user_model(user) if user else (None, None, None, None)
        model_status = "Ready" if ai_model else "Model not calibrated"

        self._send_json({
            'username': user or 'guest',
            'total_films': total_films,
            'watchlist_count': watchlist_count,
            'avg_rating': avg_rating,
            'model_status': model_status,
            'has_tmdb': has_tmdb,
            'has_gemini': has_gemini,
            'version': '5.0.0 (Neon DB Edition)'
        })

    def _handle_diary(self, query):
        user = self._get_request_user(query)
        if not user:
            self._send_json({'films': [], 'total': 0})
            return

        search = query.get('search', [''])[0].strip().lower()
        rating_filter = query.get('rating', ['All'])[0]
        year_filter = query.get('year', ['All'])[0]
        sort_mode = query.get('sort', ['Newest Log First'])[0]

        try:
            records, total, _ = get_user_diary(
                user, search=search, rating_filter=rating_filter,
                year_filter=year_filter, sort_mode=sort_mode
            )
            # Sanitize any nan or null representations
            cleaned = []
            for r in records:
                cleaned.append({
                    'movie_id': r.get('movie_id'),
                    'id': r.get('movie_id'),
                    'title': str(r.get('title') or 'Untitled'),
                    'year': str(r.get('year') or '').replace('.0', ''),
                    'genres': str(r.get('genres') or ''),
                    'overview': str(r.get('overview') or ''),
                    'director': str(r.get('director') or ''),
                    'cast': str(r.get('cast') or ''),
                    'runtime': int(r.get('runtime') or 0),
                    'vote_average': float(r.get('vote_average') or 7.0),
                    'poster_path': str(r.get('poster_path') or ''),
                    'backdrop_path': str(r.get('backdrop_path') or ''),
                    'Rating': float(r.get('Rating') or 3.5) if r.get('Rating') is not None else None,
                    'Date': str(r.get('Date') or '')
                })
            # If any diary records lack posters or are unhydrated placeholders, self-heal in background
            has_unhydrated = any(
                (not r.get('poster_path')) or int(r.get('movie_id') or 0) >= 900000000 or r.get('genres') == 'General'
                for r in records
            )
            if has_unhydrated:
                tmdb_key, _ = self._get_request_keys(user=user, query=query)
                if tmdb_key:
                    threading.Thread(target=repair_user_unhydrated_movies, args=(user, tmdb_key), daemon=True).start()

            self._send_json({'films': cleaned, 'total': total})
        except Exception as e:
            print(f"[ERROR] _handle_diary: {e}")
            self._send_json({'error': 'Internal server error', 'films': [], 'total': 0}, 500)

    def _handle_taste_radar(self, query=None):
        user = self._get_request_user(query)
        if not user:
            self._send_json({'radar': {}, 'badges': []})
            return

        try:
            records, total, _ = get_user_diary(user)
            genres_counts = {}
            for r in records:
                g_str = r.get('genres') or ''
                for g in str(g_str).split(','):
                    g = g.strip()
                    if g and g.lower() != 'nan' and g.lower() != 'general':
                        genres_counts[g] = genres_counts.get(g, 0) + 1

            top_genres = sorted(genres_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            max_val = max([v for _, v in top_genres], default=1)
            radar_data = [{ 'genre': k, 'count': v, 'pct': int((v / max_val) * 100) } for k, v in top_genres]

            badges = [
                {'title': '🌌 Sci-Fi & Mind-Bending', 'desc': 'High affinity for futuristic, dystopian & complex plots'},
                {'title': f'🎬 {total} Films Logged', 'desc': 'Calibrated AI taste profile active in RAM'},
                {'title': '⚡ Live Neon Sync', 'desc': 'Multi-device cloud synchronization active'}
            ]
            self._send_json({'radar': radar_data, 'badges': badges})
        except Exception as e:
            print(f"[ERROR] _handle_taste_radar: {e}")
            self._send_json({'radar': [], 'badges': [], 'error': 'Internal server error'})

    def _handle_get_watchlist(self, query):
        user = self._get_request_user(query)
        if not user:
            self._send_json({'watchlist': [], 'total': 0})
            return

        cluster_filter = query.get('cluster', ['All'])[0]
        sort_mode = query.get('sort', ['Highest Predicted ★'])[0]
        platform_filter = query.get('platform', ['All Platforms'])[0]

        ai_model, ai_columns, ai_vectorizer, ai_encoders = get_or_train_user_model(user)

        try:
            raw_items = get_user_watchlist(user)
            if not raw_items:
                self._send_json({'watchlist': [], 'total': 0})
                return

            # Vectorized batch prediction in RAM (< 5ms)
            scores = [3.8] * len(raw_items)
            if ai_model:
                scores = predict_movie_scores_batch(ai_model, ai_columns, ai_vectorizer, ai_encoders, raw_items)

            items = []
            for i, r in enumerate(raw_items):
                genres = str(r.get('genres') or '').split(',')
                genres = [g.strip() for g in genres if g.strip() and g.strip().lower() != 'nan']
                runtime = int(r.get('runtime') or 0)
                overview = str(r.get('overview') or '')
                year = str(r.get('year') or '').replace('.0', '')
                title = str(r.get('title') or 'Untitled')
                if title.lower() == 'nan': title = 'Untitled'

                clusters = get_mood_cluster(", ".join(genres), runtime)

                items.append({
                    'movie_id': r.get('movie_id'),
                    'id': r.get('movie_id'),
                    'title': title,
                    'poster_path': str(r.get('poster_path') or ''),
                    'backdrop_path': str(r.get('backdrop_path') or ''),
                    'genres': genres,
                    'overview': overview,
                    'year': year,
                    'release_date': year,
                    'runtime': runtime,
                    'vote_average': float(r.get('vote_average') or 7.0),
                    'ai_score': scores[i],
                    'clusters': clusters,
                    'added_date': str(r.get('added_date') or '')
                })

            # Filter by Cluster
            if cluster_filter != 'All':
                items = [m for m in items if cluster_filter in m.get('clusters', [])]

            # Filter by Streaming Platform
            if platform_filter != 'All Platforms':
                filtered = []
                for m in items:
                    provs = get_watch_providers(m.get('movie_id'))
                    if any(platform_filter.lower() in p.lower() for p in provs):
                        m['providers'] = provs
                        filtered.append(m)
                items = filtered

            # Sort
            if sort_mode == 'Highest Predicted ★':
                items.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            elif sort_mode == 'Runtime (Shortest First)':
                items.sort(key=lambda x: x.get('runtime', 999) if x.get('runtime', 0) > 0 else 999)
            elif sort_mode == 'Recently Added':
                items.sort(key=lambda x: x.get('added_date', ''), reverse=True)
            elif sort_mode == 'TMDB Rating':
                items.sort(key=lambda x: x.get('vote_average', 0), reverse=True)

            self._send_json({'watchlist': items, 'total': len(items)})
        except Exception as e:
            print(f"[ERROR] _handle_get_watchlist: {e}")
            self._send_json({'error': 'Internal server error', 'watchlist': [], 'total': 0}, 500)

    def _handle_add_watchlist(self, body):
        user = self._get_request_user(body)
        if not user:
            self._send_json({'success': False, 'message': 'User required'}, 400)
            return

        ok, msg = add_to_user_watchlist(user, body)
        self._send_json({'success': ok, 'message': msg})

    def _handle_remove_watchlist(self, body):
        user = self._get_request_user(body)
        movie_id = body.get('movie_id') or body.get('id')
        if not user or not movie_id:
            self._send_json({'success': False, 'message': 'User and movie_id required'}, 400)
            return

        ok, msg = remove_from_user_watchlist(user, movie_id)
        self._send_json({'success': ok, 'message': msg})

    def _handle_pick_tonight(self, body):
        user = self._get_request_user(body)
        tmdb_key, _ = self._get_request_keys(user=user, body=body)
        if not user:
            self._send_json({'success': False, 'movie': None, 'message': 'Username is required'})
            return

        ai_model, ai_columns, ai_vectorizer, ai_encoders = get_or_train_user_model(user)
        raw_items = get_user_watchlist(user)
        if not raw_items:
            self._send_json({'success': False, 'movie': None, 'message': 'Watchlist is empty. Add a few films first!'})
            return

        # Vectorized batch prediction in RAM
        scores = [3.8] * len(raw_items)
        if ai_model:
            scores = predict_movie_scores_batch(ai_model, ai_columns, ai_vectorizer, ai_encoders, raw_items)

        items = []
        for i, r in enumerate(raw_items):
            genres = str(r.get('genres') or '').split(',')
            genres = [g.strip() for g in genres if g.strip() and g.strip().lower() != 'nan']
            runtime = int(r.get('runtime') or 0)
            year = str(r.get('year') or '').replace('.0', '')
            ai_score = scores[i]

            items.append({
                'movie_id': r.get('movie_id'),
                'id': r.get('movie_id'),
                'title': str(r.get('title') or 'Untitled'),
                'poster_path': str(r.get('poster_path') or ''),
                'backdrop_path': str(r.get('backdrop_path') or ''),
                'genres': genres,
                'overview': str(r.get('overview') or ''),
                'year': year,
                'release_date': year,
                'runtime': runtime,
                'vote_average': float(r.get('vote_average') or 7.0),
                'ai_score': ai_score,
                'clusters': get_mood_cluster(", ".join(genres), runtime)
            })

        duration = body.get('duration', 'Any')
        mood = body.get('mood', 'Any')
        platform = body.get('platform', 'All Platforms')

        candidates = items.copy()
        if not candidates:
            self._send_json({'success': False, 'movie': None, 'message': 'Watchlist is empty. Add a few films first!'})
            return

        if duration == '< 100 mins':
            candidates = [m for m in candidates if m['runtime'] > 0 and m['runtime'] <= 100] or candidates
        elif duration == '< 2 hours':
            candidates = [m for m in candidates if m['runtime'] <= 120] or candidates
        elif duration == '2h+ Epic':
            candidates = [m for m in candidates if m['runtime'] >= 120] or candidates

        if mood != 'Any':
            matched = [m for m in candidates if mood in m['clusters']]
            if matched: candidates = matched

        if platform != 'All Platforms':
            matched = []
            for m in candidates:
                provs = get_watch_providers(m['movie_id'], tmdb_key=tmdb_key)
                if any(platform.lower() in p.lower() for p in provs):
                    m['providers'] = provs
                    matched.append(m)
            if matched: candidates = matched

        candidates.sort(key=lambda x: (x.get('ai_score', 0), x.get('vote_average', 0)), reverse=True)
        winner = candidates[0]
        
        # Ground matchmaker pitch with user taste anchors and Gemini
        taste_anchors = get_user_taste_anchors(user) if user else None
        winner['pitch'] = generate_matchmaker_pitch(
            winner,
            user_taste=taste_anchors,
            duration_pref=duration,
            mood_pref=mood,
            custom_api_key=gemini_key
        )
        winner['providers'] = get_watch_providers(winner['movie_id'], tmdb_key=tmdb_key)
        self._send_json({'success': True, 'movie': winner, 'message': 'Match found!'})

    def _handle_recommend(self, body):
        prompt = body.get('prompt', '').strip()
        if not prompt:
            self._send_json({'error': 'Prompt is required'}, 400)
            return

        user = self._get_request_user(body)
        context = body.get('context', 'Alone')
        streaming = body.get('streaming', 'All Platforms')
        source = body.get('source', 'all')
        tmdb_key, gemini_key = self._get_request_keys(user=user, body=body)

        # Load user watched titles & IDs from Neon DB
        watched_titles = []
        watched_ids = []
        taste_context = None
        if user:
            try:
                diary_rows, _, _ = get_user_diary(user)
                watched_titles = [titleNormalize(r['title']) for r in diary_rows if r.get('title')]
                watched_ids = [r['movie_id'] for r in diary_rows if r.get('movie_id')]
                taste_context = get_user_taste_anchors(user)
            except Exception:
                pass

        ai_model, ai_columns, ai_vectorizer, ai_encoders = get_or_train_user_model(user) if user else (None, None, None, None)
        ai_analysis = interpret_query_with_ai(prompt, custom_api_key=gemini_key, taste_context=taste_context)

        picks = analyze(
            watched_titles, watched_ids, [],
            ai_analysis, ai_model, ai_columns, ai_vectorizer, ai_encoders,
            user_context=context, streaming_filter=streaming, raw_prompt=prompt,
            source=source, username=user, tmdb_key=tmdb_key, gemini_key=gemini_key
        )

        # Sanitize picks against nan
        clean_picks = []
        for p in picks[:40]:
            clean_picks.append({
                'id': p.get('id'),
                'movie_id': p.get('id'),
                'title': str(p.get('title') or 'Untitled'),
                'poster_path': str(p.get('poster_path') or ''),
                'backdrop_path': str(p.get('backdrop_path') or ''),
                'year': str(p.get('release_date') or p.get('year') or '').split('-')[0].replace('.0', ''),
                'genres': p.get('genres') if isinstance(p.get('genres'), list) else str(p.get('genres') or '').split(', '),
                'overview': str(p.get('overview') or ''),
                'vote_average': float(p.get('vote_average') or 7.0),
                'ai_score': float(p.get('ai_score') or 3.8),
                'is_direct_match': bool(p.get('is_direct_match', False)),
                'is_watched': bool(p.get('is_watched', False)),
                'vibe_pitch': str(p.get('vibe_pitch') or '')
            })

        self._send_json({
            'prompt': prompt,
            'analysis': ai_analysis,
            'count': len(clean_picks),
            'candidates': clean_picks
        })

    def _handle_search_tmdb(self, query):
        q = query.get('q', [''])[0].strip()
        user = self._get_request_user(query)
        tmdb_key, _ = self._get_request_keys(user=user, query=query, allow_env_fallback=False)
        if not q or not tmdb_key:
            self._send_json({'results': [], 'error': 'TMDB API key is required'})
            return

        try:
            # 1. Direct TMDB URL or numeric ID lookup
            url_m = re.search(r'themoviedb\.org/movie/(\d+)', q)
            if url_m or (q.isdigit() and len(q) >= 2):
                movie_id = url_m.group(1) if url_m else q
                try:
                    m_resp = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={'api_key': tmdb_key}, timeout=6).json()
                    if isinstance(m_resp, dict) and m_resp.get('id'):
                        clean_results = [{
                            'id': m_resp.get('id'),
                            'movie_id': m_resp.get('id'),
                            'title': str(m_resp.get('title') or m_resp.get('name') or 'Untitled'),
                            'release_date': str(m_resp.get('release_date') or ''),
                            'poster_path': str(m_resp.get('poster_path') or ''),
                            'backdrop_path': str(m_resp.get('backdrop_path') or ''),
                            'overview': str(m_resp.get('overview') or ''),
                            'vote_average': float(m_resp.get('vote_average') or 7.0),
                            'genre_ids': [g['id'] for g in m_resp.get('genres', []) if 'id' in g]
                        }]
                        self._send_json({'results': clean_results})
                        return
                except Exception as e:
                    print(f"[WARN] direct TMDB movie ID lookup failed: {e}")

            # 2. Extract year if user typed "Title 1996" or "Title (1996)"
            year = None
            clean_q = q
            year_m = re.search(r'^(.*?)\s*\(?(\b(?:19|20)\d\d\b)\)?$', q)
            if year_m:
                clean_q = year_m.group(1).strip()
                year = year_m.group(2).strip()

            params = {'api_key': tmdb_key, 'query': clean_q}
            if year:
                params['primary_release_year'] = year

            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=6).json()
            raw_results = resp.get('results', []) if isinstance(resp, dict) else []

            # If no results with year filter, retry without primary_release_year
            if not raw_results and year:
                params.pop('primary_release_year', None)
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=6).json()
                raw_results = resp.get('results', []) if isinstance(resp, dict) else []

            if not raw_results:
                resp2 = http_session.get(f"{TMDB_BASE_URL}/search/multi", params={'api_key': tmdb_key, 'query': clean_q}, timeout=6).json()
                raw_results = [m for m in resp2.get('results', []) if m.get('media_type') != 'person'] if isinstance(resp2, dict) else []

            # 3. Exact title matching boost
            def _search_rank(m):
                m_title = (m.get('title') or m.get('name') or '').lower().strip()
                cq = clean_q.lower().strip()
                if m_title == cq:
                    return -1000.0 # Exact title match has highest priority
                if m_title.startswith(cq):
                    return -500.0 # Prefix match
                return -float(m.get('popularity') or 0.0)

            raw_results.sort(key=_search_rank)

            clean_results = []
            for m in raw_results[:20]:
                clean_results.append({
                    'id': m.get('id'),
                    'movie_id': m.get('id'),
                    'title': str(m.get('title') or m.get('name') or 'Untitled'),
                    'release_date': str(m.get('release_date') or m.get('first_air_date') or ''),
                    'poster_path': str(m.get('poster_path') or ''),
                    'backdrop_path': str(m.get('backdrop_path') or ''),
                    'overview': str(m.get('overview') or ''),
                    'vote_average': float(m.get('vote_average') or 7.0),
                    'genre_ids': m.get('genre_ids', [])
                })
            self._send_json({'results': clean_results})
        except Exception as e:
            print(f"[ERROR] _handle_search_tmdb: {e}")
            self._send_json({'error': 'Internal server error', 'results': []}, 500)

    def _handle_ripple(self, query):
        movie_id = query.get('movie_id', [''])[0]
        if not movie_id:
            self._send_json({'error': 'movie_id is required'}, 400)
            return

        try:
            user = self._get_request_user(query)
            tmdb_key, _ = self._get_request_keys(user=user, query=query)
            watched_ids = []
            watched_titles = []
            if user:
                diary_rows, _, _ = get_user_diary(user)
                watched_ids = [r['movie_id'] for r in diary_rows if r.get('movie_id')]
                watched_titles = [titleNormalize(r['title']) for r in diary_rows if r.get('title')]

            ai_model, ai_columns, ai_vectorizer, ai_encoders = get_or_train_user_model(user) if user else (None, None, None, None)
            ripples = get_post_watch_recommendations(
                int(movie_id),
                watched_titles=watched_titles,
                watched_ids=watched_ids,
                ai_model=ai_model,
                ai_columns=ai_columns,
                ai_vectorizer=ai_vectorizer,
                ai_encoders=ai_encoders,
                tmdb_key=tmdb_key
            )
            self._send_json({'movie_id': int(movie_id), 'ripples': ripples})
        except Exception as e:
            print(f"[ERROR] _handle_ripple: {e}")
            self._send_json({'error': 'Internal server error'}, 500)

    def _handle_log_movie(self, body):
        user = self._get_request_user(body)
        if not user:
            self._send_json({'success': False, 'message': 'User required to log'}, 400)
            return

        user_rec = get_or_create_user(user)
        title = str(body.get('title') or 'Untitled').strip()
        movie_id = int(body.get('movie_id') or body.get('id') or 0)
        rating = float(body.get('rating') or 3.5)

        # Upsert movie and user diary
        upsert_movies_batch([{
            'movie_id': movie_id,
            'title': title,
            'poster_path': body.get('poster_path', ''),
            'backdrop_path': body.get('backdrop_path', ''),
            'genres': body.get('genres', ''),
            'overview': body.get('overview', ''),
            'year': str(body.get('year', '')).replace('.0', '')
        }])

        watched_date = str(body.get('watched_date') or body.get('date') or pd.Timestamp.now().strftime('%Y-%m-%d')).strip()[:10]

        upsert_user_diary(user_rec['id'], [{
            'movie_id': movie_id,
            'rating': rating,
            'watched_date': watched_date
        }])

        # Auto-remove from watchlist if it was there
        remove_from_user_watchlist(user, movie_id)

        # Invalidate in-memory model so next recommendation recalibrates
        invalidate_user_model(user)

        self._send_json({
            'success': True,
            'title': title,
            'actual': rating,
            'message': f"Logged '{title}' ({rating}★) and updated Watchlist!"
        })

    def _handle_sync_watchlist(self, body):
        user = self._get_request_user(body)
        if not user:
            self._send_json({'success': False, 'message': 'Username required'})
            return
        tmdb_key, _ = self._get_request_keys(user=user, body=body)
        job_id = start_watchlist_sync_job(user, tmdb_key=tmdb_key)
        self._send_json({'success': True, 'job_id': job_id, 'message': 'Watchlist sync started in background'})

    def _handle_sync_letterboxd(self, body):
        user = self._get_request_user(body)
        if not user:
            self._send_json({'success': False, 'message': 'Username required'})
            return
        tmdb_key, gemini_key = self._get_request_keys(user=user, body=body)
        job_id = start_onboarding_job(user, tmdb_key=tmdb_key, gemini_key=gemini_key)
        self._send_json({'success': True, 'job_id': job_id, 'message': 'Diary sync started in background'})

    def _handle_retrain(self, body=None):
        user = self._get_request_user(body)
        if not user:
            self._send_json({'success': False, 'message': 'Username required'}, 400)
            return
        invalidate_user_model(user)
        train_user_model_in_memory(user)
        self._send_json({'success': True, 'message': f'Personal AI Model recalibrated for @{user} in RAM!'})

CineAIRequestHandler = MBMRRequestHandler

def start_server(host='0.0.0.0', port=8899):
    server = ThreadedHTTPServer((host, port), MBMRRequestHandler)
    print(f"MBMR Neon Backend Service running at http://{host}:{port}")
    server.serve_forever()

if __name__ == '__main__':
    start_server()
