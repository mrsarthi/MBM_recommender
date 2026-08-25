import json
import re
from backend.config import gemini_model

VALID_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

def interpret_query_with_ai(user_input):
    """
    Parses natural language mood/vibe prompts into TMDB genres, search query, and suggested titles.
    """
    if gemini_model:
        try:
            prompt = (
                f"You are an elite cinephile movie assistant. The user wants recommendations for:\n\n"
                f"\"{user_input}\"\n\n"
                f"Return a clean JSON object with:\n"
                f"- 'genres': list of matching TMDB genres from: {', '.join(VALID_GENRES)}\n"
                f"- 'search_query': specific keyword, director, or theme (e.g. 'neo-noir', 'Christopher Nolan')\n"
                f"- 'suggested_titles': 3-5 specific movie titles matching the intent\n\n"
                f"Return ONLY valid JSON."
            )
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()
            
            data = json.loads(raw)
            data['genres'] = [g for g in data.get('genres', []) if g in VALID_GENRES]
            if not data['genres'] and not data.get('search_query') and not data.get('suggested_titles'):
                data['genres'] = _fallback_mood_match(user_input)
            if not data.get('search_query'):
                data['search_query'] = user_input.strip()
            return data
        except Exception as e:
            print(f"⚠️ Gemini query parsing fallback: {e}")
    
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
