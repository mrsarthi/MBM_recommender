import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import pandas as pd

from backend.config import BASE_DIR, PROFILE_PATH, APP_MEMORY_FILE, MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH, TMDB_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE, LETTERBOXD_USERNAME
from backend.predictions import load_ai, predict_movie_score, get_post_watch_recommendations, get_watch_providers
from backend.gemini_client import interpret_query_with_ai
from backend.recommender import analyze, load_watched_data, titleNormalize, http_session
from backend.sync_letterboxd import sync_rss, merge_records_into_profile
from backend.feature_engineering import feature_engineering
from backend.model_train import train_personal_model
from backend.watchlist import load_watchlist, add_to_watchlist, remove_from_watchlist, pick_movie_for_tonight, sync_letterboxd_watchlist

# Load model cache
ai_model, ai_columns, ai_vectorizer, ai_encoders = load_ai(MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class CineAIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        frontend_dir = os.path.join(BASE_DIR, 'frontend')
        super().__init__(*args, directory=frontend_dir, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/api/status':
            self._handle_status()
        elif path == '/api/diary':
            self._handle_diary(query)
        elif path == '/api/watchlist':
            self._handle_get_watchlist(query)
        elif path == '/api/taste_radar':
            self._handle_taste_radar()
        elif path == '/api/ripple':
            self._handle_ripple(query)
        elif path == '/api/search_tmdb':
            self._handle_search_tmdb(query)
        elif path.startswith('/api/'):
            self._send_json({'error': 'Endpoint not found'}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('Content-Length', 0))
        body = {}
        if length > 0:
            try:
                body = json.loads(self.rfile.read(length).decode('utf-8'))
            except Exception: pass

        if path == '/api/recommend':
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
            self._handle_sync(body)
        elif path == '/api/retrain':
            self._handle_retrain()
        else:
            self._send_json({'error': 'POST endpoint not found'}, 404)

    def _handle_status(self):
        total_films = 0
        avg_rating = 0.0
        if os.path.exists(PROFILE_PATH):
            try:
                df = pd.read_csv(PROFILE_PATH)
                total_films = len(df)
                if 'Rating' in df.columns:
                    valid = df['Rating'].dropna()
                    if not valid.empty: avg_rating = round(float(valid.mean()), 2)
            except Exception: pass

        watchlist_items = load_watchlist()
        global ai_model
        model_status = "Ready" if ai_model else "Model not loaded"
        self._send_json({
            'username': LETTERBOXD_USERNAME,
            'total_films': total_films,
            'watchlist_count': len(watchlist_items),
            'avg_rating': avg_rating,
            'model_status': model_status,
            'version': '4.0.0'
        })

    def _handle_recommend(self, body):
        prompt = body.get('prompt', '').strip()
        if not prompt:
            self._send_json({'error': 'Prompt is required'}, 400)
            return

        context = body.get('context', 'Alone')
        streaming = body.get('streaming', 'All Platforms')
        
        watched_titles, watched_ids, hated_movies = load_watched_data()
        ai_analysis = interpret_query_with_ai(prompt)
        
        picks = analyze(
            watched_titles, watched_ids, hated_movies,
            ai_analysis, ai_model, ai_columns, ai_vectorizer, ai_encoders,
            user_context=context, streaming_filter=streaming, raw_prompt=prompt
        )

        self._send_json({
            'prompt': prompt,
            'analysis': ai_analysis,
            'count': len(picks),
            'candidates': picks[:40]
        })

    def _handle_diary(self, query):
        if not os.path.exists(PROFILE_PATH):
            self._send_json({'films': [], 'total': 0})
            return

        df = pd.read_csv(PROFILE_PATH)
        df.columns = [c.strip() for c in df.columns]
        
        search = query.get('search', [''])[0].strip().lower()
        rating_filter = query.get('rating', ['All'])[0]
        year_filter = query.get('year', ['All'])[0]
        sort_mode = query.get('sort', ['Newest Log First'])[0]

        # Filter Search
        if search:
            col = 'Name' if 'Name' in df.columns else 'Title'
            df = df[df[col].fillna('').astype(str).str.lower().str.contains(search, na=False)]

        # Filter Rating
        if 'Rating' in df.columns:
            if rating_filter == '5★ Only':
                df = df[df['Rating'] == 5.0]
            elif rating_filter == '4★+':
                df = df[df['Rating'] >= 4.0]
            elif rating_filter == '3★+':
                df = df[df['Rating'] >= 3.0]
            elif rating_filter == 'Unrated':
                df = df[df['Rating'].isna()]

        # Filter Year
        if year_filter != 'All' and 'Year' in df.columns:
            df = df[df['Year'].fillna('').astype(str).str.startswith(year_filter)]

        # Sort
        if sort_mode == 'Newest Log First' and 'Date' in df.columns:
            df['dt_sort'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='dt_sort', ascending=False, na_position='last')
        elif sort_mode == 'Oldest Log First' and 'Date' in df.columns:
            df['dt_sort'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='dt_sort', ascending=True, na_position='last')
        elif sort_mode == 'Highest Rating' and 'Rating' in df.columns:
            df = df.sort_values(by='Rating', ascending=False)
        elif sort_mode == 'Lowest Rating' and 'Rating' in df.columns:
            df = df.sort_values(by='Rating', ascending=True)
        elif sort_mode == 'Title A-Z':
            col = 'Name' if 'Name' in df.columns else 'Title'
            df = df.sort_values(by=col, ascending=True)

        if 'dt_sort' in df.columns:
            df = df.drop(columns=['dt_sort'])

        # Convert all to strings or native primitives
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            elif df[col].dtype == object:
                df[col] = df[col].fillna('').astype(str)

        records = df.head(200).fillna('').to_dict(orient='records')
        self._send_json({'films': records, 'total': len(df)})

    def _handle_taste_radar(self):
        if not os.path.exists(PROFILE_PATH):
            self._send_json({'radar': {}, 'badges': []})
            return

        df = pd.read_csv(PROFILE_PATH)
        df.columns = [c.strip() for c in df.columns]
        
        genres_counts = {}
        if 'genres' in df.columns:
            for g_str in df['genres'].dropna():
                for g in str(g_str).split(','):
                    g = g.strip()
                    if g: genres_counts[g] = genres_counts.get(g, 0) + 1

        top_genres = sorted(genres_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        max_val = max([v for _, v in top_genres], default=1)
        radar_data = [{ 'genre': k, 'count': v, 'pct': int((v / max_val) * 100) } for k, v in top_genres]

        badges = [
            {'title': '🌌 Sci-Fi Savant', 'desc': 'High affinity for futuristic, dystopian & hard sci-fi'},
            {'title': '🎬 Cinema Veteran', 'desc': 'Over 700+ lifetime films logged and rated'},
            {'title': '🧠 Mind-Bending Explorer', 'desc': 'Deep exploration of psychological thrillers and complex plots'}
        ]

        self._send_json({'radar': radar_data, 'badges': badges})

    def _handle_ripple(self, query):
        movie_id = query.get('movie_id', [None])[0]
        if not movie_id:
            self._send_json({'error': 'movie_id is required'}, 400)
            return

        watched_titles, watched_ids, _ = load_watched_data()
        ripples = get_post_watch_recommendations(movie_id, watched_titles, watched_ids, ai_model, ai_columns, ai_vectorizer, ai_encoders)
        self._send_json({'ripples': ripples})

    def _handle_search_tmdb(self, query):
        q = query.get('q', [''])[0].strip()
        if not q or not TMDB_KEY:
            self._send_json({'results': []})
            return

        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': q}, timeout=6).json()
            self._send_json({'results': resp.get('results', [])[:15]})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _handle_get_watchlist(self, query):
        cluster_filter = query.get('cluster', ['All'])[0]
        sort_mode = query.get('sort', ['Highest Predicted ★'])[0]
        platform_filter = query.get('platform', ['All Platforms'])[0]

        items = load_watchlist(ai_model=ai_model, ai_cols=ai_columns, ai_vec=ai_vectorizer, ai_enc=ai_encoders)

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

    def _handle_add_watchlist(self, body):
        ok, msg = add_to_watchlist(body)
        self._send_json({'success': ok, 'message': msg})

    def _handle_remove_watchlist(self, body):
        movie_id = body.get('movie_id') or body.get('id')
        ok, msg = remove_from_watchlist(movie_id)
        self._send_json({'success': ok, 'message': msg})

    def _handle_pick_tonight(self, body):
        winner, msg = pick_movie_for_tonight(body, ai_model=ai_model, ai_cols=ai_columns, ai_vec=ai_vectorizer, ai_enc=ai_encoders)
        self._send_json({'success': bool(winner), 'movie': winner, 'message': msg})

    def _handle_sync_watchlist(self, body):
        username = (body.get('username') or '').strip() or LETTERBOXD_USERNAME
        ok, msg = sync_letterboxd_watchlist(username)
        self._send_json({'success': ok, 'message': msg})

    def _handle_log_movie(self, body):
        title = body.get('title', '').strip()
        movie_id = body.get('movie_id')
        rating = float(body.get('rating', 3.5))
        context = body.get('context', 'Alone')
        genres = body.get('genres', [])
        overview = body.get('overview', '')
        
        predicted = 3.5
        if ai_model:
            predicted = round(predict_movie_score(ai_model, ai_columns, ai_vectorizer, ai_encoders, genres=genres, context=context, overview=overview), 1)

        diff = round(rating - predicted, 1)
        
        # Save to memory and profile
        if movie_id:
            try:
                remove_from_watchlist(movie_id)
                if os.path.exists(APP_MEMORY_FILE):
                    with open(APP_MEMORY_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"{movie_id},{title}\n")
            except Exception: pass

        merge_records_into_profile([{
            'Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'Name': title,
            'Rating': rating,
            'movie_id': movie_id,
            'poster_path': body.get('poster_path', ''),
            'backdrop_path': body.get('backdrop_path', '')
        }])

        self._send_json({
            'success': True,
            'title': title,
            'predicted': predicted,
            'actual': rating,
            'diff': diff,
            'message': f"Successfully logged {title} ({rating}★) and updated Watchlist"
        })

    def _handle_sync(self, body):
        username = (body.get('username') or '').strip() or LETTERBOXD_USERNAME
        ok, msg = sync_rss(username)
        
        # Reload model
        global ai_model, ai_columns, ai_vectorizer, ai_encoders
        ai_model, ai_columns, ai_vectorizer, ai_encoders = load_ai(MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH)
        
        self._send_json({'success': ok, 'message': msg})

    def _handle_retrain(self):
        fe_ok = feature_engineering()
        if fe_ok:
            tr_ok = train_personal_model()
            if tr_ok:
                global ai_model, ai_columns, ai_vectorizer, ai_encoders
                ai_model, ai_columns, ai_vectorizer, ai_encoders = load_ai(MODEL_PATH, COLUMNS_PATH, VECTORIZER_PATH, ENCODERS_PATH)
                self._send_json({'success': True, 'message': 'Personal AI Model retrained successfully!'})
                return
        self._send_json({'success': False, 'message': 'Retraining failed. Check data.'}, 500)

def start_server(port=8899):
    server = ThreadedHTTPServer(('127.0.0.1', port), CineAIRequestHandler)
    print(f"CineAI Backend Service running at http://127.0.0.1:{port}")
    server.serve_forever()

if __name__ == '__main__':
    start_server()
