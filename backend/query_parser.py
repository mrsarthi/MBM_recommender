import os
import json
import re
from datetime import datetime

VALID_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

LANGUAGE_MAP = {
    'japanese': ['ja'],
    'japan': ['ja'],
    'anime': ['ja'],
    'chinese': ['zh', 'cn'],
    'china': ['zh', 'cn'],
    'hong kong': ['zh', 'cn'],
    'taiwanese': ['zh', 'cn'],
    'wuxia': ['zh', 'cn'],
    'korean': ['ko'],
    'korea': ['ko'],
    'k-drama': ['ko'],
    'french': ['fr'],
    'france': ['fr'],
    'italian': ['it'],
    'italy': ['it'],
    'giallo': ['it'],
    'spanish': ['es'],
    'spain': ['es'],
    'mexican': ['es'],
    'mexico': ['es'],
    'german': ['de'],
    'germany': ['de'],
    'hindi': ['hi'],
    'indian': ['hi'],
    'bollywood': ['hi'],
    'nordic': ['sv', 'da', 'no'],
    'swedish': ['sv'],
    'danish': ['da'],
    'norwegian': ['no'],
    'russian': ['ru'],
    'soviet': ['ru']
}

THEMATIC_KEYWORD_MAP = {
    'samurai': ['samurai', 'sword fight', 'chanbara', 'ronin'],
    'ninja': ['ninja', 'shinobi', 'assassin', 'martial arts'],
    'wuxia': ['wuxia', 'martial arts', 'swordplay', 'kung fu'],
    'kung fu': ['kung fu', 'martial arts', 'hand to hand combat'],
    'sword': ['sword fight', 'swordplay', 'swordsman'],
    'swords': ['sword fight', 'swordplay', 'swordsman'],
    'spaghetti western': ['spaghetti western', 'gunslinger', 'bounty hunter'],
    'western': ['western', 'cowboy', 'outlaw', 'frontier'],
    'giallo': ['giallo', 'murder mystery', 'slasher'],
    'cyberpunk': ['cyberpunk', 'dystopia', 'futuristic', 'artificial intelligence'],
    'slasher': ['slasher', 'serial killer', 'masked killer'],
    'body horror': ['body horror', 'mutation', 'grotesque'],
    'kaiju': ['giant monster', 'kaiju', 'creature feature'],
    'neo-noir': ['neo-noir', 'hardboiled', 'femme fatale', 'detective'],
    'noir': ['film noir', 'private investigator', 'crime noir'],
    'psychological thriller': ['psychological thriller', 'unreliable narrator', 'paranoia'],
    'time travel': ['time travel', 'time loop', 'temporal'],
    'space opera': ['space opera', 'interstellar', 'spaceship'],
    'heist': ['heist', 'bank robbery', 'caper'],
    'whodunit': ['whodunit', 'murder mystery', 'detective'],
    'dark comedy': ['dark comedy', 'black comedy', 'satire'],
    'coming of age': ['coming of age', 'teenage', 'youth'],
    'courtroom': ['courtroom drama', 'legal drama', 'trial'],
    'haunted house': ['haunted house', 'ghost', 'possession'],
    'erotic': ['erotic', 'erotica', 'sensual', 'sexual', 'passion', 'desire', 'provocative', 'affair', 'erotic thriller'],
    'erotica': ['erotic', 'erotica', 'sensual', 'sexual', 'passion', 'desire'],
    'sex': ['erotic', 'sexuality', 'sensual', 'sexual obsession', 'erotic thriller', 'passion', 'adult', 'steamy'],
    'sexual': ['erotic', 'sexuality', 'sensual', 'sexual obsession', 'passion'],
    'sensual': ['sensual', 'erotic', 'passion', 'intimate', 'steamy'],
    'steamy': ['erotic', 'steamy', 'passion', 'sensual', 'affair', 'intimate'],
    'adult': ['erotic', 'adult', 'sexuality', 'provocative', 'mature'],
    'nudity': ['erotic', 'nudity', 'sensual', 'provocative', 'explicit'],
    'gore': ['gore', 'bloody', 'splatter', 'body horror', 'mutilation'],
    'gory': ['gore', 'bloody', 'splatter', 'body horror'],
    'splatter': ['splatter', 'gore', 'bloody', 'slasher'],
    'acclaimed': ['critically acclaimed', 'masterpiece', 'award winning', 'palme d\'or'],
    'critically acclaimed': ['critically acclaimed', 'masterpiece', 'award winning', 'classic'],
    'masterpiece': ['masterpiece', 'critically acclaimed', 'essential cinema']
}

NEGATION_GENRE_MAP = {
    'anime': ('Animation', ['anime', 'manga', 'animation', 'animated', 'japanese animation']),
    'animated': ('Animation', ['animation', 'animated', 'cartoon']),
    'animation': ('Animation', ['animation', 'animated', 'cartoon']),
    'cartoon': ('Animation', ['animation', 'animated', 'cartoon']),
    'romance': ('Romance', ['romance', 'romantic', 'love story']),
    'romantic': ('Romance', ['romance', 'romantic']),
    'comedy': ('Comedy', ['comedy', 'humor', 'funny', 'slapstick']),
    'horror': ('Horror', ['horror', 'scary', 'spooky', 'slasher']),
    'action': ('Action', ['action', 'explosions']),
    'thriller': ('Thriller', ['thriller', 'suspense']),
    'sci-fi': ('Science Fiction', ['sci-fi', 'scifi', 'science fiction']),
    'scifi': ('Science Fiction', ['sci-fi', 'scifi', 'science fiction']),
    'science fiction': ('Science Fiction', ['sci-fi', 'scifi', 'science fiction']),
    'drama': ('Drama', ['drama']),
    'fantasy': ('Fantasy', ['fantasy']),
    'crime': ('Crime', ['crime', 'gangster', 'mafia']),
    'mystery': ('Mystery', ['mystery', 'whodunit']),
    'documentary': ('Documentary', ['documentary', 'docuseries']),
    'musical': ('Music', ['musical', 'music']),
    'music': ('Music', ['musical', 'music']),
    'western': ('Western', ['western', 'cowboy']),
    'family': ('Family', ['family', 'kids', 'children']),
    'kids': ('Family', ['family', 'kids']),
    'war': ('War', ['war', 'military']),
    'history': ('History', ['history', 'historical', 'period piece']),
    'historical': ('History', ['history', 'historical', 'period piece'])
}

def _extract_negations(user_input):
    """
    Deterministic extraction of negative / excluded genres and keywords from prompt.
    E.g. 'not anime', 'no romance', 'without horror', 'aren't animated', 'non-anime'.
    """
    text = (user_input or '').lower()
    excluded_genres = set()
    excluded_keywords = set()

    clauses = re.split(r'[,;.]|\band\b|\bor\b', text)
    neg_pattern = r'\b(?:not|no|without|non[- ]|isn\'t|aren\'t|exclude|excluding|never|zero)\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,2})'

    for clause in clauses:
        clause = clause.strip()
        for match in re.finditer(neg_pattern, clause):
            phrase_matched = match.group(1).strip()
            words = phrase_matched.split()
            for i in range(1, min(4, len(words) + 1)):
                subphrase = ' '.join(words[:i]).strip()
                if subphrase in NEGATION_GENRE_MAP:
                    genre, kws = NEGATION_GENRE_MAP[subphrase]
                    excluded_genres.add(genre)
                    excluded_keywords.update(kws)

    for term in ['anime', 'animated', 'fiction']:
        if re.search(r'\bnon[- ]' + term + r'\b', text):
            if term in NEGATION_GENRE_MAP:
                genre, kws = NEGATION_GENRE_MAP[term]
                excluded_genres.add(genre)
                excluded_keywords.update(kws)

    return {
        'genres': list(excluded_genres),
        'keywords': list(excluded_keywords)
    }

def _extract_languages(user_input, excluded_keywords=None):
    """Deterministic extraction of ISO-639-1 language codes from prompt."""
    text = (user_input or '').lower()
    languages = set()
    ex_kws = set(excluded_keywords or [])
    for name, codes in LANGUAGE_MAP.items():
        if name in ex_kws:
            continue
        if re.search(r'\b' + re.escape(name) + r'\b', text):
            languages.update(codes)
    return list(languages)

def _extract_thematic_keywords(user_input, excluded_keywords=None):
    """Deterministic extraction of cinephile sub-genre keyword tags."""
    text = (user_input or '').lower()
    keywords = set()
    ex_kws = set(excluded_keywords or [])
    for term, kw_list in THEMATIC_KEYWORD_MAP.items():
        if term in ex_kws:
            continue
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            keywords.update(kw_list)
    return list(keywords)

def _is_upcoming_query(user_input):
    """Returns True if the prompt specifies unreleased/upcoming/anticipated future cinema."""
    text = (user_input or '').lower()
    return bool(re.search(r'\b(?:upcoming|anticipated|coming soon|future|unreleased|in theaters|next year|this year)\b', text))

def _extract_year_constraints(user_input):
    """Deterministic extraction of temporal / decade / year bounds from prompt."""
    text = (user_input or '').lower()
    year_min, year_max = None, None
    current_year = datetime.now().year

    # Check forward-looking phrases
    if _is_upcoming_query(text):
        year_min = current_year

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

    # Colloquial era phrases
    if year_max is None and year_min is None:
        if re.search(r'\b(?:old|classic|vintage|golden age|retro|older)\b', text):
            year_max = 1999
        elif re.search(r'\b(?:modern|recent|new|latest)\b', text):
            year_min = 2015

    return year_min, year_max


def _extract_reference_entity(user_input):
    """Deterministic extraction of reference movie titles from comparison queries."""
    text = (user_input or '').strip()
    pattern = r'^(?:(?:can you\s+)?(?:recommend|find|suggest|give me)?\s*(?:something|anything|movies?|films?)?\s*(?:like|similar to|in the vein of)\s+)(.+)$'
    m = re.search(pattern, text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text
    
    # Strip trailing conjunctions, year/decade qualifiers, temporal prepositions
    candidate = re.sub(r'\s*(?:but|and|however|yet)?\s*(?:from\s+)?(?:before|prior to|older than|after|post|in|during)?\s*(?:the\s*)?\b(?:\d{4}s?|\d{2}s)\b.*$', '', candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r'\s*(?:from\s+before\s+.*|from\s+after\s+.*|but\s+before\s+.*|but\s+after\s+.*)$', '', candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r'\s+\b(?:but|and|or|yet|with|without)\b\s*$', '', candidate, flags=re.IGNORECASE).strip()
    
    if candidate and candidate.lower() != text.lower() and len(candidate.split()) <= 6:
        return candidate.strip(' "\'')
    return None

def _fallback_mood_match(user_input, excluded_genres=None):
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
        'horror': ['Horror'],
        'gore': ['Horror'],
        'gory': ['Horror'],
        'splatter': ['Horror'],
        'slasher': ['Horror'],
        'body horror': ['Horror', 'Science Fiction'],
        'mutilation': ['Horror'],
        'nudity': ['Thriller', 'Drama', 'Romance'],
        'nude': ['Drama', 'Romance'],
        'intense': ['Action', 'Thriller', 'War'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'romantic': ['Romance', 'Drama', 'Comedy'],
        'mind-bending': ['Science Fiction', 'Mystery', 'Thriller'],
        'psychological': ['Thriller', 'Mystery', 'Drama'],
        'cyberpunk': ['Science Fiction', 'Action', 'Thriller'],
        'sci-fi': ['Science Fiction', 'Mystery', 'Thriller'],
        'noir': ['Crime', 'Mystery', 'Thriller'],
        'indie': ['Drama', 'Romance'],
        'sex': ['Romance', 'Drama', 'Thriller'],
        'erotic': ['Thriller', 'Romance', 'Drama'],
        'erotica': ['Romance', 'Drama', 'Thriller'],
        'steamy': ['Romance', 'Drama', 'Thriller'],
        'sensual': ['Romance', 'Drama'],
        'adult': ['Drama', 'Thriller', 'Romance'],
        'affair': ['Drama', 'Romance', 'Thriller'],
        'passion': ['Romance', 'Drama']
    }
    text = (user_input or '').lower()
    matched_genres = set()
    ex_genres = set(excluded_genres or [])

    for g in VALID_GENRES:
        gl = g.lower()
        if (gl in text or (gl + 's') in text) and g not in ex_genres:
            matched_genres.add(g)
    if ('sci-fi' in text or 'scifi' in text) and 'Science Fiction' not in ex_genres:
        matched_genres.add('Science Fiction')

    for keyword, genres in fallback_map.items():
        if keyword in text:
            for g in genres:
                if g not in ex_genres:
                    matched_genres.add(g)

    if matched_genres:
        return list(matched_genres)

    if text.strip():
        return []

    safe_defaults = [g for g in ['Action', 'Drama', 'Science Fiction'] if g not in ex_genres]
    return safe_defaults if safe_defaults else ['Drama']

def _fallback_watchlist_relevance(user_input, watchlist_movies):
    """
    Deterministic rule-based thematic relevance scoring for watchlist movies.
    """
    text = (user_input or '').lower().strip()
    negations = _extract_negations(text)
    excluded_genres = set(negations.get('genres', []))
    excluded_kws = set(negations.get('keywords', []))

    filler = {'some', 'film', 'films', 'movie', 'movies', 'a', 'the', 'an', 'and', 'or', 'for', 'with', 'in', 'on', 'of', 'me', 'my', 'i', 'want', 'good', 'not', 'no', 'without'}
    tokens = [t for t in re.split(r'[^a-z0-9\-]+', text) if t and t not in filler and t not in excluded_kws]

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
        if kw in text and kw not in excluded_kws and g not in excluded_genres:
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

        if excluded_genres and any(eg.lower() in m_genres for eg in excluded_genres):
            continue
        if excluded_kws and any(ek in m_title or ek in m_overview or ek in m_genres for ek in excluded_kws):
            continue

        score = 0.35
        pitch_reasons = []

        if target_genres:
            matched_g = [g for g in target_genres if g.lower() in m_genres]
            if matched_g:
                score += 0.45
                pitch_reasons.append(f"Features {' & '.join(matched_g)} elements")
            else:
                score -= 0.40

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

        if is_niche_query:
            if vote_count < 3000:
                score += 0.35
                pitch_reasons.append("A specialized, less-mainstream cinematic gem")
            elif vote_count > 15000:
                score -= 0.25
            if any(x in m_genres for x in ['drama', 'documentary', 'mystery']):
                score += 0.15

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

CASCADE_MODELS = ['models/gemini-2.5-flash', 'models/gemini-1.5-flash']
GEMINI_API_KEY = None

def _get_genai_client(custom_api_key=None):
    key = custom_api_key or os.getenv('GEMINI_API_KEY') or GEMINI_API_KEY
    if not key or key == 'YOUR_GEMINI_API_KEY_HERE':
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None

def filter_and_rank_watchlist_with_ai(user_input, watchlist_movies, taste_context=None, custom_api_key=None):
    """Deterministic rule-based watchlist filtering and thematic ranking."""
    return _fallback_watchlist_relevance(user_input, watchlist_movies)

def generate_matchmaker_pitch(movie, taste_context=None, user_taste=None, duration_pref=None, mood_pref=None, custom_api_key=None, _client_override=None, **kwargs):
    """Generates a compelling recommendation pitch for the winning matchmaker pick."""
    taste = taste_context or user_taste
    title = movie.get('title') or 'This film'
    year = str(movie.get('year') or '').replace('.0', '')
    genres = movie.get('genres') or ''
    runtime = movie.get('runtime') or ''
    score = movie.get('ai_score') or 3.8
    if isinstance(genres, list):
        genres = ", ".join(genres)
    
    client = _client_override or _get_genai_client(custom_api_key)
    if client:
        try:
            for model_name in CASCADE_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=f"Write a 1-sentence recommendation pitch for {title} ({year})."
                    )
                    raw = (resp.text or '').strip()
                    if raw.startswith("```json"): raw = raw[7:-3].strip()
                    elif raw.startswith("```"): raw = raw[3:-3].strip()
                    try:
                        d = json.loads(raw)
                        if isinstance(d, dict) and d.get('pitch'):
                            return d['pitch']
                    except Exception:
                        pass
                    if raw:
                        return raw
                except Exception:
                    continue
        except Exception:
            pass

    rt_str = f" ({runtime} min)" if runtime else ""
    return f"'{title}' ({year}) matches your taste profile with a high affinity score ({score:.1f}★){rt_str} and features {genres or 'stellar storytelling'}."

def _clean_search_query(user_input, ref_entity=None, is_upcoming=False):
    """
    Cleans conversational noise, query framing, and meta-intents from search_query
    so TMDB title search is only invoked for genuine film titles.
    """
    text = (user_input or '').strip()
    if ref_entity:
        return ref_entity

    # If it's a generic upcoming query, title search should be empty
    if is_upcoming and re.search(r'^(?:most\s+anticipated\s+)?(?:upcoming|anticipated|coming\s+soon|new)\s*(?:movies|films)?$', text, re.IGNORECASE):
        return ""

    # Check if this is a vibe/mood prompt rather than a movie title
    mood_patterns = [
        r'\b(?:something|anything|movies?|films?)\s+(?:like|similar to|in the vein of)\b',
        r'\b(?:most\s+anticipated|upcoming|coming\s+soon)\b',
        r'\b(?:gore|slasher|splatter|horror|nudity|erotic|steamy)\s+(?:movies?|films?)\b',
        r'\b(?:movies?|films?)\s+with\s+(?:nudity|gore|violence)\b'
    ]
    for pat in mood_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return ""

    # Strip conversational prefixes
    cleaned = re.sub(r'^(?:can you\s+)?(?:recommend|find|suggest|give me|show me|looking for)\s+', '', text, flags=re.IGNORECASE).strip()
    # Strip temporal constraints at end
    cleaned = re.sub(r'\s*(?:but|and)?\s*(?:from\s+)?(?:before|prior to|older than|after|post|in|during)\s*(?:the\s*)?\b(?:\d{4}s?|\d{2}s)\b.*$', '', cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned if len(cleaned) >= 2 else ""

def _extract_compound_keyword_groups(user_input):
    """
    Detects when multiple orthogonal thematic concepts are demanded in one query
    (e.g., 'gore' AND 'nudity', or 'cyberpunk' AND 'heist').
    Returns keyword groups that must be intersected with AND in discovery.
    """
    text = (user_input or '').lower()
    groups = []
    
    if re.search(r'\b(?:gore|gory|splatter|body horror|mutilation)\b', text):
        groups.append(['gore', 'splatter', 'body horror'])
        
    if re.search(r'\b(?:nudity|nude|erotic|erotica|sex|sexual|steamy|sensual)\b', text):
        groups.append(['nudity', 'erotic', 'explicit'])
        
    if re.search(r'\b(?:time travel|time loop|temporal)\b', text):
        groups.append(['time travel', 'time loop'])
        
    if re.search(r'\b(?:cyberpunk|dystopia|dystopian)\b', text):
        groups.append(['cyberpunk', 'dystopia'])
        
    if re.search(r'\b(?:samurai|wuxia|kung fu|martial arts)\b', text):
        groups.append(['samurai', 'martial arts'])

    return groups if len(groups) >= 2 else []

def interpret_query(user_input, taste_context=None, custom_api_key=None, _client_override=None, **kwargs):
    """
    Semantic NLP query engine with deterministic rule resolution and optional LLM cascade.
    Extracts TMDB genres, search query, year bounds, ISO language codes,
    thematic keywords, reference entity, excluded genres/keywords, and upcoming release intent.
    """
    text = (user_input or '').strip()
    negations = _extract_negations(text)
    excluded_genres = negations['genres']
    excluded_keywords = negations['keywords']

    det_ymin, det_ymax = _extract_year_constraints(text)
    is_upcoming = _is_upcoming_query(text)
    det_ref_entity = _extract_reference_entity(text)
    det_langs = _extract_languages(text, excluded_keywords=excluded_keywords)
    det_kws = _extract_thematic_keywords(text, excluded_keywords=excluded_keywords)
    det_genres = _fallback_mood_match(text, excluded_genres=excluded_genres)
    det_compound_groups = _extract_compound_keyword_groups(text)
    det_search_query = _clean_search_query(text, ref_entity=det_ref_entity, is_upcoming=is_upcoming)

    client = _client_override or _get_genai_client(custom_api_key)
    if client:
        taste_str = ""
        if taste_context and isinstance(taste_context, dict):
            favs = ", ".join(taste_context.get('favorite_movies', [])[:3])
            if favs:
                taste_str = f"User Favorites: {favs}\n"
        prompt = (
            f"Analyze this movie recommendation request:\n"
            f"\"{text}\"\n\n"
            f"{taste_str}"
            f"Return ONLY a JSON object with format:\n"
            f"{{\n"
            f"  \"genres\": [\"Genre1\", ...],\n"
            f"  \"search_query\": \"...\",\n"
            f"  \"suggested_titles\": [\"Title1\", ...]\n"
            f"}}"
        )
        for model_name in CASCADE_MODELS:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                raw = (resp.text or '').strip()
                if raw.startswith("```json"): raw = raw[7:-3].strip()
                elif raw.startswith("```"): raw = raw[3:-3].strip()
                data = json.loads(raw)
                llm_sq = data.get('search_query') or ''
                # Clean conversational LLM output if it echoed the whole prompt
                final_sq = _clean_search_query(llm_sq, ref_entity=det_ref_entity, is_upcoming=is_upcoming) if llm_sq else det_search_query
                return {
                    'genres': data.get('genres') if 'genres' in data else det_genres,
                    'search_query': final_sq,
                    'suggested_titles': data.get('suggested_titles') or [],
                    'year_min': det_ymin,
                    'year_max': det_ymax,
                    'is_upcoming': is_upcoming,
                    'languages': det_langs,
                    'reference_entity': det_ref_entity,
                    'thematic_keywords': det_kws,
                    'compound_keyword_groups': det_compound_groups,
                    'excluded_genres': excluded_genres,
                    'excluded_keywords': excluded_keywords
                }
            except Exception:
                continue

    return {
        'genres': det_genres,
        'search_query': det_search_query,
        'suggested_titles': [],
        'year_min': det_ymin,
        'year_max': det_ymax,
        'is_upcoming': is_upcoming,
        'languages': det_langs,
        'reference_entity': det_ref_entity,
        'thematic_keywords': det_kws,
        'compound_keyword_groups': det_compound_groups,
        'excluded_genres': excluded_genres,
        'excluded_keywords': excluded_keywords
    }

interpret_query_with_ai = interpret_query
