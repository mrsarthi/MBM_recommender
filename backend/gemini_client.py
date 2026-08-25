import os
import json
import re
import requests
from backend.config import GEMINI_API_KEY, gemini_model

VALID_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

# Verified active models with available quota in priority order
CASCADE_MODELS = [
    'gemini-flash-lite-latest',
    'gemini-3.1-flash-lite',
    'gemini-3.5-flash-lite',
    'gemini-2.5-flash'
]

def interpret_query_with_ai(user_input, custom_api_key=None):
    """
    Parses natural language mood/vibe prompts into TMDB genres, search query, and suggested titles.
    Cascades across verified active Gemini models if quota is exhausted on any single model.
    """
    active_key = custom_api_key or GEMINI_API_KEY
    if not active_key:
        return {'genres': _fallback_mood_match(user_input), 'search_query': user_input.strip(), 'suggested_titles': []}

    prompt = (
        f"You are an elite cinephile movie assistant. The user wants recommendations for:\n\n"
        f"\"{user_input}\"\n\n"
        f"Return a clean JSON object with:\n"
        f"- 'genres': list of matching TMDB genres from: {', '.join(VALID_GENRES)}\n"
        f"- 'search_query': specific keyword, director, or theme (e.g. 'neo-noir', 'Christopher Nolan')\n"
        f"- 'suggested_titles': 3-5 specific movie titles matching the intent\n\n"
        f"Return ONLY valid JSON."
    )

    for model_name in CASCADE_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
            }
            resp = requests.post(url, json=payload, timeout=3.5)
            if resp.status_code == 200:
                raw = resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                if raw.startswith("```json"): raw = raw[7:-3].strip()
                elif raw.startswith("```"): raw = raw[3:-3].strip()

                data = json.loads(raw)
                data['genres'] = [g for g in data.get('genres', []) if g in VALID_GENRES]
                if not data['genres'] and not data.get('search_query') and not data.get('suggested_titles'):
                    data['genres'] = _fallback_mood_match(user_input)
                if not data.get('search_query'):
                    data['search_query'] = user_input.strip()
                return data
            else:
                # Quota or rate-limit error, cascade to next model in list
                continue
        except Exception:
            continue

    # If all models exhausted, use deterministic heuristic parser
    return {'genres': _fallback_mood_match(user_input), 'search_query': user_input.strip(), 'suggested_titles': []}

def _fallback_mood_match(user_input):
    fallback_map = {
        'happy': ['Comedy', 'Music', 'Animation', 'Family', 'Romance'],
        'sad': ['Drama', 'Romance'],
        'tense': ['Horror', 'Thriller', 'Mystery', 'Crime'],
        'adventurous': ['Adventure', 'Science Fiction', 'Fantasy', 'Action'],
        'calm': ['Documentary', 'Drama', 'History'],
        'nostalgic': ['Drama', 'Romance', 'Fantasy'],
        'excited': ['Action', 'Adventure', 'Comedy'],
        'thoughtful': ['Drama', 'Documentary'],
        'scary': ['Horror', 'Thriller'],
        'intense': ['Action', 'Thriller', 'War'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'romantic': ['Romance', 'Drama', 'Comedy'],
        'mind-bending': ['Science Fiction', 'Mystery', 'Thriller'],
        'sci-fi': ['Science Fiction', 'Mystery', 'Thriller'],
        'noir': ['Crime', 'Mystery', 'Thriller'],
        'indie': ['Drama', 'Romance']
    }
    text = user_input.lower()
    matched_genres = set()

    # A TMDB genre named outright is the strongest possible signal. This was missing,
    # so a plain "thriller" query fell through to the generic default below.
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
