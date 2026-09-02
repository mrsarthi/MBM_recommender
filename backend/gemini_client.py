import os
import json
import re
from google import genai
from google.genai import types
from backend.config import GEMINI_API_KEY, gemini_client

VALID_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

# Verified active models with available quota in priority order
CASCADE_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-flash-lite-latest',
    'gemini-3.1-flash-lite',
    'gemini-3.5-flash-lite'
]

def _get_genai_client(active_key=None):
    key = active_key or GEMINI_API_KEY
    if not key or key in ('YOUR_GEMINI_API_KEY_HERE', ''):
        return None
    if active_key == GEMINI_API_KEY and gemini_client is not None:
        return gemini_client
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None

def _extract_year_constraints(user_input):
    """Deterministic extraction of temporal / decade / year bounds from prompt."""
    text = (user_input or '').lower()
    year_min, year_max = None, None

    # Before / Pre X
    m_before = re.search(r'(?:before|pre[- ]?|prior to|older than|earlier than)\s*(?:the\s*)?(\d{4})s?', text)
    if m_before:
        y = int(m_before.group(1))
        year_max = y - 1 if text.find('s') == -1 and not m_before.group(0).endswith('s') else y - 1
        if m_before.group(0).endswith('s') or '00s' in m_before.group(0):
            year_max = y - 1 if y % 100 == 0 else y + 9

    # After / Post X
    m_after = re.search(r'(?:after|post[- ]?|newer than|later than|since)\s*(?:the\s*)?(\d{4})s?', text)
    if m_after:
        y = int(m_after.group(1))
        year_min = y + 1 if not m_after.group(0).endswith('s') else y + 10

    # Specific Decades
    decade_patterns = [
        (r'\b(19[2-9]0|20[0-2]0)s\b', lambda y: (y, y + 9)),
        (r'\b(?:the\s*)?([2-9]0)s\b', lambda y: (1900 + y if y >= 20 else 2000 + y, 1900 + y + 9 if y >= 20 else 2000 + y + 9)),
    ]
    for pat, mapper in decade_patterns:
        m_dec = re.search(pat, text)
        if m_dec and not m_before and not m_after:
            y = int(m_dec.group(1))
            ymin, ymax = mapper(y)
            year_min, year_max = ymin, ymax
            break

    # Year Ranges: "from 1990 to 1999" or "1995-2005"
    m_range = re.search(r'\b(19\d{2}|20\d{2})\s*(?:to|-|until)\s*(19\d{2}|20\d{2})\b', text)
    if m_range:
        year_min, year_max = int(m_range.group(1)), int(m_range.group(2))

    return year_min, year_max

def _extract_reference_entity(user_input):
    """Deterministic extraction of reference movie titles from comparison queries."""
    text = (user_input or '').strip()
    pattern = r'^(?:(?:can you\s+)?(?:recommend|find|suggest|give me)?\s*(?:something|anything|movies?|films?)?\s*(?:like|similar to|in the vein of)\s+)(.+)$'
    m = re.search(pattern, text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text
    
    # Strip trailing temporal clauses
    candidate = re.sub(r'\s*(?:from\s+)?(?:before|prior to|older than|after|post|in|during)?\s*(?:the\s*)?\b(?:\d{4}s?|\d{2}s)\b.*$', '', candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r'\s*(?:from\s+before\s+.*|from\s+after\s+.*)$', '', candidate, flags=re.IGNORECASE).strip()
    
    if candidate and candidate.lower() != text.lower() and len(candidate.split()) <= 6:
        return candidate.strip(' "\'')
    return None

def interpret_query_with_ai(user_input, custom_api_key=None, taste_context=None):
    """
    Parses natural language mood/vibe prompts into TMDB genres, search query, year constraints,
    thematic keywords, reference entity, and suggested titles with vibe pitches.
    Grounded with the user's Letterboxd taste anchors (top directors, 5★ favorites, high affinity genres).
    Cascades across active Gemini models if quota is exhausted on any single model.
    """
    active_key = custom_api_key or GEMINI_API_KEY
    det_ymin, det_ymax = _extract_year_constraints(user_input)
    det_ref_entity = _extract_reference_entity(user_input)

    if not active_key or active_key == 'YOUR_GEMINI_API_KEY_HERE':
        return {
            'genres': _fallback_mood_match(user_input),
            'search_query': user_input.strip(),
            'suggested_titles': [],
            'year_min': det_ymin,
            'year_max': det_ymax,
            'reference_entity': det_ref_entity,
            'thematic_keywords': []
        }

    client = _get_genai_client(active_key)
    if not client:
        return {
            'genres': _fallback_mood_match(user_input),
            'search_query': user_input.strip(),
            'suggested_titles': [],
            'year_min': det_ymin,
            'year_max': det_ymax,
            'reference_entity': det_ref_entity,
            'thematic_keywords': []
        }

    # Format user taste context if available
    taste_prompt_block = ""
    if taste_context and isinstance(taste_context, dict):
        favs = ", ".join(taste_context.get('favorite_movies', [])[:4])
        dirs = ", ".join(taste_context.get('top_directors', [])[:3])
        genres = ", ".join(taste_context.get('top_genres', [])[:3])
        decades = ", ".join(taste_context.get('preferred_decades', [])[:3])
        
        parts = []
        if favs: parts.append(f"- 5-Star Favorites: {favs}")
        if dirs: parts.append(f"- Top-Rated Directors: {dirs}")
        if genres: parts.append(f"- Favorite Genres: {genres}")
        if decades: parts.append(f"- Preferred Decades: {decades}")
        
        if parts:
            taste_prompt_block = (
                f"User Taste Profile (use as reference for aesthetic sensibilities, tone, and storytelling standards):\n"
                + "\n".join(parts) + "\n"
                f"Note: Use these taste anchors to gauge quality and style. Do NOT restrict recommendations only to these directors if others fit the mood better.\n\n"
            )

    prompt = (
        f"You are an elite cinephile movie assistant with deep cinematic knowledge.\n\n"
        f"{taste_prompt_block}"
        f"The user wants movie recommendations for the following prompt / vibe / reference:\n"
        f"\"{user_input}\"\n\n"
        f"Analyze the prompt carefully. If the user mentions a reference myth, concept, or era (e.g. 'the odyssey', 'greek mythology', 'cyberpunk', 'before 2000s'), identify the core narrative tropes and strictly enforce any era/time restrictions.\n\n"
        f"Provide a curated selection of 12-18 specific, high-acclaim movie titles matching this exact aesthetic and time constraint.\n"
        f"Return a clean JSON object with:\n"
        f"- 'genres': list of matching TMDB genres from: {', '.join(VALID_GENRES)}\n"
        f"- 'search_query': semantic keyword search phrase (e.g. 'greek mythology voyage epic quest', 'rainy neo noir')\n"
        f"- 'year_min': integer minimum release year if constrained by prompt (or null)\n"
        f"- 'year_max': integer maximum release year if constrained by prompt (or null, e.g. 1999 if 'before 2000s')\n"
        f"- 'thematic_keywords': list of 3-5 specific thematic keywords (e.g. ['greek mythology', 'heroic voyage', 'epic quest'])\n"
        f"- 'suggested_titles': list of objects with:\n"
        f"    - 'title': exact movie title (e.g. 'Jason and the Argonauts')\n"
        f"    - 'year': release year string (e.g. '1963')\n"
        f"    - 'vibe_pitch': 1 concise sentence explaining why this film matches the mood and cinematic taste\n\n"
        f"Return ONLY valid JSON."
    )

    for model_name in CASCADE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )
            raw = (response.text or '').strip()
            if not raw:
                continue
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()

            data = json.loads(raw)
            data['genres'] = [g for g in data.get('genres', []) if g in VALID_GENRES]
            
            # Enforce deterministic year constraints if AI missed them
            ai_ymin = data.get('year_min')
            ai_ymax = data.get('year_max')
            data['year_min'] = det_ymin if det_ymin is not None else (int(ai_ymin) if ai_ymin and str(ai_ymin).isdigit() else None)
            data['year_max'] = det_ymax if det_ymax is not None else (int(ai_ymax) if ai_ymax and str(ai_ymax).isdigit() else None)
            data['thematic_keywords'] = [str(k).strip() for k in data.get('thematic_keywords', []) if k]

            # Normalize suggested_titles format
            raw_titles = data.get('suggested_titles', [])
            normalized_titles = []
            for item in raw_titles:
                if isinstance(item, dict) and item.get('title'):
                    normalized_titles.append({
                        'title': str(item.get('title', '')).strip(),
                        'year': str(item.get('year', '')).strip(),
                        'vibe_pitch': str(item.get('vibe_pitch', '')).strip()
                    })
                elif isinstance(item, str) and item.strip():
                    normalized_titles.append({
                        'title': item.strip(),
                        'year': '',
                        'vibe_pitch': ''
                    })
            data['suggested_titles'] = normalized_titles

            data['reference_entity'] = det_ref_entity

            if not data['genres'] and not data.get('search_query') and not data.get('suggested_titles'):
                data['genres'] = _fallback_mood_match(user_input)
            if not data.get('search_query'):
                data['search_query'] = user_input.strip()
            return data
        except Exception:
            continue

    # If all models exhausted, use deterministic heuristic parser
    return {
        'genres': _fallback_mood_match(user_input),
        'search_query': user_input.strip(),
        'suggested_titles': [],
        'year_min': det_ymin,
        'year_max': det_ymax,
        'reference_entity': det_ref_entity,
        'thematic_keywords': []
    }

def generate_matchmaker_pitch(movie_dict, user_taste=None, duration_pref='Any', mood_pref='Any', custom_api_key=None):
    """
    Generates a personalized, witty, 2-sentence pitch for 'Pick For Me Tonight'
    highlighting why this specific watchlist movie matches tonight's constraints and user taste.
    """
    active_key = custom_api_key or GEMINI_API_KEY
    title = movie_dict.get('title', 'This film')
    year = movie_dict.get('year') or movie_dict.get('release_date') or ''
    runtime = movie_dict.get('runtime') or 0
    genres = movie_dict.get('genres', [])
    genres_str = ", ".join(genres) if isinstance(genres, list) else str(genres)
    overview = movie_dict.get('overview', '')
    ai_score = movie_dict.get('ai_score', 3.8)

    # Dynamic fallback template
    fallback_pitch = (
        f"Selected as your #1 match tonight with a {int(ai_score * 20)}% personal affinity score. "
        f"At {runtime or 'feature'} min, this {genres_str or 'watchlist'} pick delivers on your mood and unwinds decision fatigue."
    )

    if not active_key or active_key == 'YOUR_GEMINI_API_KEY_HERE':
        return fallback_pitch

    client = _get_genai_client(active_key)
    if not client:
        return fallback_pitch

    taste_context_str = ""
    if user_taste and isinstance(user_taste, dict):
        favs = ", ".join(user_taste.get('favorite_movies', [])[:3])
        dirs = ", ".join(user_taste.get('top_directors', [])[:2])
        if favs or dirs:
            taste_context_str = f"User likes: {favs}. Favorite directors: {dirs}."

    prompt = (
        f"You are a witty, charismatic cinephile AI matchmaker. The user asked MBMR to pick one movie from their watchlist for tonight.\n\n"
        f"Selected Winner: {title} ({year}, {runtime} min, {genres_str})\n"
        f"Synopsis: {overview[:200]}...\n"
        f"{taste_context_str}\n"
        f"Tonight's constraints: Duration: {duration_pref}, Mood: {mood_pref}.\n\n"
        f"Write a sharp, personalized, 2-sentence pitch explaining why they should watch this movie right now.\n"
        f"Return ONLY valid JSON with format: {{\"pitch\": \"...\"}}"
    )

    for model_name in CASCADE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=200,
                    response_mime_type="application/json"
                )
            )
            raw = (response.text or '').strip()
            if not raw:
                continue
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()
            data = json.loads(raw)
            pitch = data.get('pitch', '').strip()
            if pitch:
                return pitch
        except Exception:
            continue

    return fallback_pitch

def filter_and_rank_watchlist_with_ai(user_input, watchlist_movies, custom_api_key=None, taste_context=None):
    """
    Evaluates watchlist movies directly against a natural language mood/vibe query using Gemini.
    Returns a dict mapping int(movie_id) -> {'relevance': float (0.1 to 1.0), 'vibe_pitch': str}.
    Falls back gracefully to local heuristic relevance scoring if AI is unavailable.
    """
    if not watchlist_movies:
        return {}

    active_key = custom_api_key or GEMINI_API_KEY
    if not active_key or active_key == 'YOUR_GEMINI_API_KEY_HERE':
        return _fallback_watchlist_relevance(user_input, watchlist_movies)

    client = _get_genai_client(active_key)
    if not client:
        return _fallback_watchlist_relevance(user_input, watchlist_movies)

    # Format compact watchlist candidate lines (up to 80 items)
    compact_lines = []
    for idx, m in enumerate(watchlist_movies[:80]):
        m_id = m.get('movie_id') or m.get('id')
        title = m.get('title') or 'Untitled'
        year = str(m.get('year') or '').replace('.0', '')
        genres = m.get('genres', '')
        if isinstance(genres, list):
            genres = ", ".join(genres)
        overview = str(m.get('overview') or '')[:160]
        compact_lines.append(f"[{m_id}] \"{title}\" ({year}) | Genres: {genres} | Plot: {overview}")

    candidates_text = "\n".join(compact_lines)

    taste_context_str = ""
    if taste_context and isinstance(taste_context, dict):
        favs = ", ".join(taste_context.get('favorite_movies', [])[:3])
        if favs:
            taste_context_str = f"User's all-time favorites for style context: {favs}.\n"

    prompt = (
        f"You are an expert film curator and taste analyst.\n\n"
        f"The user wants recommendations specifically FROM THEIR WATCHLIST for this mood / vibe:\n"
        f"\"{user_input}\"\n\n"
        f"{taste_context_str}"
        f"Here are the user's Watchlist Candidates:\n"
        f"{candidates_text}\n\n"
        f"Task:\n"
        f"1. Select movies from the watchlist candidates above that genuinely match this specific mood/vibe.\n"
        f"2. For each match, assign a 'relevance' score from 1 to 10 (10 = absolute bullseye for this exact mood, 5 = decent fit; do NOT include poor fits < 5).\n"
        f"3. Write a 1-sentence 'vibe_pitch' specifically explaining why this watchlist film matches their '{user_input}' request.\n\n"
        f"Return ONLY a JSON object with format:\n"
        f"{{\n"
        f"  \"matches\": [\n"
        f"    {{\"movie_id\": <int>, \"relevance\": <int 1-10>, \"vibe_pitch\": \"...\"}}\n"
        f"  ]\n"
        f"}}"
    )

    for model_name in CASCADE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1200,
                    response_mime_type="application/json"
                )
            )
            raw = (response.text or '').strip()
            if not raw:
                continue
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()
            data = json.loads(raw)
            matches_list = data.get('matches', [])
            result_map = {}
            for item in matches_list:
                mid = item.get('movie_id')
                try:
                    mid = int(mid)
                    rel = float(item.get('relevance', 7)) / 10.0
                    pitch = str(item.get('vibe_pitch', '')).strip()
                    result_map[mid] = {'relevance': min(1.0, max(0.1, rel)), 'vibe_pitch': pitch}
                except (ValueError, TypeError):
                    continue
            if result_map:
                return result_map
        except Exception:
            continue

    return _fallback_watchlist_relevance(user_input, watchlist_movies)

def _fallback_watchlist_relevance(user_input, watchlist_movies):
    """
    Deterministic rule-based thematic relevance scoring for watchlist movies.
    Ensures distinct query terms (e.g. 'weird', 'horror', 'niche', 'comedy') produce distinct rankings.
    """
    text = user_input.lower().strip()
    filler = {'some', 'film', 'films', 'movie', 'movies', 'a', 'the', 'an', 'and', 'or', 'for', 'with', 'in', 'on', 'of', 'me', 'my', 'i', 'want', 'good'}
    tokens = [t for t in re.split(r'[^a-z0-9\-]+', text) if t and t not in filler]

    genre_keywords = {
        'horror': 'Horror', 'scary': 'Horror', 'spooky': 'Horror', 'slasher': 'Horror',
        'comedy': 'Comedy', 'funny': 'Comedy', 'hilarious': 'Comedy',
        'thriller': 'Thriller', 'tense': 'Thriller', 'suspense': 'Thriller',
        'sci-fi': 'Science Fiction', 'scifi': 'Science Fiction', 'space': 'Science Fiction', 'dystopian': 'Science Fiction',
        'action': 'Action', 'adrenaline': 'Action',
        'drama': 'Drama', 'emotional': 'Drama', 'sad': 'Drama', 'melancholic': 'Drama',
        'romance': 'Romance', 'romantic': 'Romance', 'love': 'Romance',
        'mystery': 'Mystery', 'detective': 'Mystery', 'whodunnit': 'Mystery',
        'crime': 'Crime', 'gangster': 'Crime', 'noir': 'Crime', 'neo-noir': 'Crime',
        'animation': 'Animation', 'animated': 'Animation', 'anime': 'Animation',
        'fantasy': 'Fantasy', 'magic': 'Fantasy',
        'western': 'Western', 'documentary': 'Documentary'
    }

    target_genres = set()
    for kw, g in genre_keywords.items():
        if kw in text:
            target_genres.add(g)

    weird_terms = {'weird', 'surreal', 'strange', 'bizarre', 'unconventional', 'hallucinatory', 'mindbending', 'mind-bending', 'absurd', 'cult', 'psychedelic', 'trippy', 'body-horror', 'grotesque'}
    is_weird_query = any(w in text for w in weird_terms)

    niche_terms = {'niche', 'indie', 'arthouse', 'obscure', 'underrated', 'hidden', 'gem', 'cult', 'underground', 'experimental', 'foreign'}
    is_niche_query = any(n in text for n in niche_terms)

    results = {}
    for m in watchlist_movies:
        mid = m.get('movie_id') or m.get('id')
        if not mid: continue
        try: mid = int(mid)
        except: continue

        m_genres = str(m.get('genres') or '').lower()
        m_title = str(m.get('title') or '').lower()
        m_overview = str(m.get('overview') or '').lower()
        m_director = str(m.get('director') or '').lower()
        vote_count = int(m.get('vote_count') or 500)

        score = 0.35  # Neutral baseline
        pitch_reasons = []

        # 1. Target Genre Filter / Boost
        if target_genres:
            matched_g = [g for g in target_genres if g.lower() in m_genres]
            if matched_g:
                score += 0.45
                pitch_reasons.append(f"Features {' & '.join(matched_g)} elements")
            else:
                score -= 0.40  # Heavy penalty for missing explicitly requested genres

        # 2. Weird / Surreal Boost
        if is_weird_query:
            weird_match_count = sum(1 for w in weird_terms if (w in m_overview or w in m_title or w in m_genres))
            if any(x in m_genres for x in ['mystery', 'science fiction', 'horror', 'fantasy']):
                score += 0.20
            if weird_match_count > 0:
                score += 0.35
                pitch_reasons.append("Delivers a surreal, mind-bending atmosphere")
            else:
                if any(x in m_genres for x in ['family', 'romance', 'action']):
                    score -= 0.15

        # 3. Niche / Indie Boost
        if is_niche_query:
            if vote_count < 3000:
                score += 0.35
                pitch_reasons.append("A specialized, less-mainstream cinematic gem")
            elif vote_count > 15000:
                score -= 0.25
            if any(x in m_genres for x in ['drama', 'documentary', 'mystery']):
                score += 0.15

        # 4. Keyword token match
        for token in tokens:
            if len(token) > 2:
                if token in m_title:
                    score += 0.35
                    pitch_reasons.append(f"Title matches '{token}'")
                elif token in m_overview:
                    score += 0.20
                    pitch_reasons.append(f"Themes match '{token}'")
                elif token in m_director:
                    score += 0.30
                    pitch_reasons.append(f"Directed by {m.get('director')}")

        final_rel = round(min(1.0, max(0.05, score)), 2)
        if final_rel >= 0.40:
            pitch = f"Matches your search for \"{user_input}\""
            if pitch_reasons:
                pitch += f": {pitch_reasons[0]}."
            else:
                pitch += f" with strong tonal alignment."
            results[mid] = {'relevance': final_rel, 'vibe_pitch': pitch}

    return results

def _fallback_mood_match(user_input):
    fallback_map = {
        'happy': ['Comedy', 'Music', 'Animation', 'Family', 'Romance'],
        'sad': ['Drama', 'Romance'],
        'tense': ['Horror', 'Thriller', 'Mystery', 'Crime'],
        'adventurous': ['Adventure', 'Science Fiction', 'Fantasy', 'Action'],
        'odyssey': ['Adventure', 'Fantasy', 'Action'],
        'myth': ['Fantasy', 'Adventure', 'Action'],
        'mythology': ['Fantasy', 'Adventure', 'Action'],
        'quest': ['Adventure', 'Fantasy', 'Action'],
        'voyage': ['Adventure', 'Drama', 'Fantasy'],
        'calm': ['Documentary', 'Drama', 'History'],
        'nostalgic': ['Drama', 'Romance', 'Fantasy'],
        'excited': ['Action', 'Adventure', 'Comedy'],
        'thoughtful': ['Drama', 'Documentary'],
        'scary': ['Horror', 'Thriller'],
        'intense': ['Action', 'Thriller', 'War'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'romantic': ['Romance', 'Drama', 'Comedy'],
        'mind-bending': ['Science Fiction', 'Mystery', 'Thriller'],
        'psychological': ['Thriller', 'Mystery', 'Drama'],
        'cyberpunk': ['Science Fiction', 'Action', 'Thriller'],
        'sci-fi': ['Science Fiction', 'Mystery', 'Thriller'],
        'noir': ['Crime', 'Mystery', 'Thriller'],
        'indie': ['Drama', 'Romance']
    }
    text = user_input.lower()
    matched_genres = set()

    for g in VALID_GENRES:
        gl = g.lower()
        if gl in text or (gl + 's') in text:
            matched_genres.add(g)
    if 'sci-fi' in text or 'scifi' in text:
        matched_genres.add('Science Fiction')

    for keyword, genres in fallback_map.items():
        if keyword in text:
            matched_genres.update(genres)

    return list(matched_genres) if matched_genres else ['Action', 'Drama', 'Science Fiction']
