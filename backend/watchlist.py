import os
import re
import html
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from backend.config import WATCHLIST_PATH, TMDB_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE
from backend.predictions import predict_movie_score, get_watch_providers, load_ai
from backend.recommender import titleNormalize, load_watched_data, http_session

def get_mood_cluster(genres_str, runtime_mins=0):
    g_lower = str(genres_str).lower()
    clusters = []
    if any(g in g_lower for g in ['science fiction', 'mystery', 'thriller', 'mind-bending']):
        clusters.append('Mind-Bending')
    if any(g in g_lower for g in ['noir', 'crime', 'drama', 'romance']):
        clusters.append('Late Night')
    if any(g in g_lower for g in ['action', 'adventure', 'war']):
        clusters.append('Popcorn & Adrenaline')
    if any(g in g_lower for g in ['comedy', 'animation', 'family']):
        clusters.append('Comfort')
    if runtime_mins > 0 and runtime_mins <= 105:
        clusters.append('Quick Watch')
    return clusters if clusters else ['General']

def load_watchlist(watchlist_path=WATCHLIST_PATH, ai_model=None, ai_cols=None, ai_vec=None, ai_enc=None):
    if not os.path.exists(watchlist_path) or os.path.getsize(watchlist_path) == 0:
        return []

    try:
        df = pd.read_csv(watchlist_path)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return []

    records = []
    for _, row in df.iterrows():
        movie_id = row.get('movie_id')
        if pd.isna(movie_id): continue
        
        try: movie_id = int(float(movie_id))
        except: continue

        title = str(row.get('title', row.get('Name', 'Untitled'))).strip()
        poster = str(row.get('poster_path', '')) if pd.notna(row.get('poster_path')) else ''
        backdrop = str(row.get('backdrop_path', '')) if pd.notna(row.get('backdrop_path')) else ''
        genres = str(row.get('genres', '')).split(',') if pd.notna(row.get('genres')) else []
        genres = [g.strip() for g in genres if g.strip()]
        overview = str(row.get('overview', '')) if pd.notna(row.get('overview')) else ''
        year = str(row.get('year', row.get('Year', ''))).strip()
        
        try: runtime = int(float(row.get('runtime', 0)))
        except: runtime = 0
        
        try: vote_avg = float(row.get('vote_average', 7.0))
        except: vote_avg = 7.0

        ai_score = 3.8
        if ai_model:
            ai_score = round(predict_movie_score(ai_model, ai_cols, ai_vec, ai_enc, genres=genres, overview=overview), 1)

        clusters = get_mood_cluster(", ".join(genres), runtime)

        records.append({
            'movie_id': movie_id,
            'id': movie_id,
            'title': title,
            'poster_path': poster,
            'backdrop_path': backdrop,
            'genres': genres,
            'overview': overview,
            'year': year,
            'release_date': year,
            'runtime': runtime,
            'vote_average': vote_avg,
            'ai_score': ai_score,
            'clusters': clusters,
            'added_date': str(row.get('added_date', ''))
        })

    return records

def add_to_watchlist(movie_data, watchlist_path=WATCHLIST_PATH):
    movie_id = movie_data.get('id') or movie_data.get('movie_id')
    if not movie_id: return False, "Missing movie ID"
    
    try: movie_id = int(movie_id)
    except: return False, "Invalid movie ID"

    title = movie_data.get('title', 'Untitled')
    poster = movie_data.get('poster_path', '')
    backdrop = movie_data.get('backdrop_path', '')
    genres = movie_data.get('genres', [])
    if isinstance(genres, list):
        genres_str = ", ".join([g['name'] if isinstance(g, dict) else str(g) for g in genres])
    else:
        genres_str = str(genres)
        
    overview = movie_data.get('overview', '')
    year = str(movie_data.get('release_date', movie_data.get('year', ''))).split('-')[0]
    runtime = movie_data.get('runtime', 0)
    vote_avg = movie_data.get('vote_average', 7.0)

    # If runtime is missing, fetch from TMDB
    if not runtime and TMDB_KEY:
        try:
            resp = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={'api_key': TMDB_KEY}, timeout=4).json()
            if isinstance(resp, dict):
                runtime = resp.get('runtime', 0)
                if not poster: poster = resp.get('poster_path', '')
                if not backdrop: backdrop = resp.get('backdrop_path', '')
                if not overview: overview = resp.get('overview', '')
        except Exception: pass

    if os.path.exists(watchlist_path) and os.path.getsize(watchlist_path) > 0:
        df = pd.read_csv(watchlist_path)
    else:
        df = pd.DataFrame(columns=['movie_id', 'title', 'poster_path', 'backdrop_path', 'genres', 'overview', 'year', 'runtime', 'vote_average', 'added_date'])

    # Check duplicate
    if not df.empty and 'movie_id' in df.columns:
        if movie_id in df['movie_id'].dropna().astype(int).values:
            return True, f"'{title}' is already in your watchlist."

    new_row = {
        'movie_id': movie_id,
        'title': title,
        'poster_path': poster,
        'backdrop_path': backdrop,
        'genres': genres_str,
        'overview': overview,
        'year': year,
        'runtime': runtime,
        'vote_average': vote_avg,
        'added_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    os.makedirs(os.path.dirname(watchlist_path), exist_ok=True)
    df.to_csv(watchlist_path, index=False)
    return True, f"Added '{title}' to Watchlist!"

def remove_from_watchlist(movie_id, watchlist_path=WATCHLIST_PATH):
    if not os.path.exists(watchlist_path): return True, "Watchlist empty"
    try:
        df = pd.read_csv(watchlist_path)
        if df.empty or 'movie_id' not in df.columns: return True, "Watchlist empty"
        
        m_id = int(movie_id)
        df = df[df['movie_id'].astype(int) != m_id]
        df.to_csv(watchlist_path, index=False)
        return True, "Removed from Watchlist."
    except Exception as e:
        return False, str(e)

def pick_movie_for_tonight(params, ai_model=None, ai_cols=None, ai_vec=None, ai_enc=None, watchlist_path=WATCHLIST_PATH):
    items = load_watchlist(watchlist_path=watchlist_path, ai_model=ai_model, ai_cols=ai_cols, ai_vec=ai_vec, ai_enc=ai_enc)
    if not items:
        return None, "Your watchlist is currently empty. Add a few films first!"

    duration = params.get('duration', 'Any')
    mood = params.get('mood', 'Any')
    platform = params.get('platform', 'All Platforms')

    candidates = items.copy()

    # 1. Filter duration
    if duration == '< 100 mins':
        candidates = [m for m in candidates if m['runtime'] > 0 and m['runtime'] <= 100] or candidates
    elif duration == '< 2 hours':
        candidates = [m for m in candidates if m['runtime'] <= 120] or candidates
    elif duration == '2h+ Epic':
        candidates = [m for m in candidates if m['runtime'] >= 120] or candidates

    # 2. Filter mood / cluster
    if mood != 'Any':
        matched = [m for m in candidates if mood in m['clusters']]
        if matched: candidates = matched

    # 3. Sort by predicted AI rating
    candidates.sort(key=lambda x: (x.get('ai_score', 0), x.get('vote_average', 0)), reverse=True)

    winner = candidates[0]
    
    # Generate intelligent personalized pitch
    cluster_str = ", ".join(winner['clusters'][:2])
    runtime_str = f"{winner['runtime']} min" if winner['runtime'] else "Feature film"
    pitch = (
        f"Selected as your #1 match tonight with a {int(winner['ai_score'] * 20)}% personal affinity score. "
        f"At {runtime_str}, this {cluster_str} pick aligns with your rating history and unwinds decision fatigue."
    )

    winner['pitch'] = pitch
    winner['providers'] = get_watch_providers(winner['movie_id'])
    return winner, "Match found!"

def sync_letterboxd_watchlist(username="sarthi_watcher", watchlist_path=WATCHLIST_PATH):
    clean_user = username.strip().lstrip('@')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}

    genreDict = {
        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
        9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
        10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
    }

    all_entries = []
    seen_slugs = set()

    page = 1
    while True:
        url = f"https://letterboxd.com/{clean_user}/watchlist/page/{page}/"
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                break
            
            items = re.findall(r'data-target-link="/film/([^/"]+)/"[^>]*>.*?<img[^>]*alt="([^"]+)"', resp.text, re.DOTALL)
            if not items:
                # Fallback to img alt
                alts = re.findall(r'<img[^>]+alt="([^"]+)"', resp.text)
                for a in alts:
                    clean = html.unescape(a.strip())
                    if not any(x in clean.lower() for x in ['parth', 'avatar', 'photo', 'letterboxd']) and clean not in seen_slugs:
                        seen_slugs.add(clean)
                        all_entries.append({'slug': clean.lower().replace(' ', '-'), 'title': clean, 'year_hint': ''})
            else:
                for slug, alt in items:
                    if slug not in seen_slugs:
                        seen_slugs.add(slug)
                        clean_title = html.unescape(alt.strip())
                        year_match = re.search(r'-(\d{4})$', slug)
                        year_hint = year_match.group(1) if year_match else ''
                        all_entries.append({
                            'slug': slug,
                            'title': clean_title,
                            'year_hint': year_hint
                        })

            if 'paginate-next' not in resp.text and 'class="next"' not in resp.text:
                break
            page += 1
        except Exception:
            break

    if not all_entries:
        return False, f"No watchlist films found on @{clean_user}'s profile."

    def resolve_entry(entry):
        title = entry['title']
        year_hint = entry['year_hint']
        slug = entry['slug']
        norm_title = titleNormalize(title)
        
        clean_slug_name = re.sub(r'-\d{4}$', '', slug).replace('-', ' ')
        clean_alpha_title = re.sub(r'[^\w\s]', ' ', title)
        
        queries = [title, clean_alpha_title, clean_slug_name]
        queries = list(dict.fromkeys([q.strip() for q in queries if q.strip()]))

        candidates = []
        for q in queries:
            try:
                r = requests.get(f"{TMDB_BASE_URL}/search/movie", params={'api_key': TMDB_KEY, 'query': q}, timeout=6).json()
                for m in r.get('results', []):
                    if m.get('id') not in [c.get('id') for c in candidates]:
                        candidates.append(m)
            except Exception: pass

        if not candidates:
            for q in queries:
                try:
                    r = requests.get(f"{TMDB_BASE_URL}/search/multi", params={'api_key': TMDB_KEY, 'query': q}, timeout=6).json()
                    for m in r.get('results', []):
                        if m.get('media_type') != 'person' and m.get('id') not in [c.get('id') for c in candidates]:
                            candidates.append(m)
                except Exception: pass

        best = None
        if year_hint:
            for m in candidates:
                m_title = m.get('title') or m.get('name') or m.get('original_title') or ''
                rel = (m.get('release_date') or m.get('first_air_date') or '')[:4]
                if (titleNormalize(m_title) == norm_title or titleNormalize(m_title) == titleNormalize(clean_slug_name)) and rel == year_hint:
                    best = m
                    break

        if not best:
            exact = []
            for m in candidates:
                m_title = m.get('title') or m.get('name') or m.get('original_title') or ''
                if titleNormalize(m_title) == norm_title or titleNormalize(m_title) == titleNormalize(clean_slug_name):
                    exact.append(m)
            if exact:
                exact.sort(key=lambda x: (x.get('vote_count', 0), x.get('popularity', 0)), reverse=True)
                best = exact[0]

        if not best and year_hint:
            for m in candidates:
                rel = (m.get('release_date') or m.get('first_air_date') or '')[:4]
                if rel == year_hint:
                    best = m
                    break

        if not best and candidates:
            candidates.sort(key=lambda x: (x.get('vote_count', 0), x.get('popularity', 0)), reverse=True)
            best = candidates[0]

        if best:
            m_id = best.get('id')
            m_title = best.get('title') or best.get('name') or title
            poster = best.get('poster_path', '')
            backdrop = best.get('backdrop_path', '')
            overview = best.get('overview', '')
            release_date = best.get('release_date') or best.get('first_air_date') or year_hint or ''
            year = release_date.split('-')[0] if release_date else year_hint
            vote_avg = round(float(best.get('vote_average', 7.0)), 1)
            
            genre_ids = best.get('genre_ids', [])
            genres_list = [genreDict[g] for g in genre_ids if g in genreDict]
            genres_str = ", ".join(genres_list)

            return {
                'movie_id': m_id,
                'title': m_title,
                'poster_path': poster,
                'backdrop_path': backdrop,
                'genres': genres_str,
                'overview': overview,
                'year': year,
                'runtime': 0,
                'vote_average': vote_avg,
                'added_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
        else:
            return {
                'movie_id': abs(hash(slug)) % 10000000,
                'title': title,
                'poster_path': '',
                'backdrop_path': '',
                'genres': 'General',
                'overview': '',
                'year': year_hint,
                'runtime': 0,
                'vote_average': 7.0,
                'added_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }

    with ThreadPoolExecutor(max_workers=10) as executor:
        resolved = list(executor.map(resolve_entry, all_entries))

    # Overwrite watchlist with all exact resolved entries
    df = pd.DataFrame(resolved)
    os.makedirs(os.path.dirname(watchlist_path), exist_ok=True)
    df.to_csv(watchlist_path, index=False)

    return True, f"Successfully synced all {len(df)} movies from @{clean_user}'s Letterboxd Watchlist!"
