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

_NEGATION_TRIGGERS = {
    'not', 'no', 'without', 'non', "isn't", "aren't", "don't", 'exclude',
    'excluding', 'never', 'zero', 'avoid', 'skip', 'minus'
}
# Words that end a negated phrase ("without nudity BUT with gore").
_NEGATION_STOPS = {
    'but', 'with', 'that', 'which', 'who', 'where', 'while', 'yet', 'though',
    'although', 'please', 'from', 'before', 'after', 'like', 'about', 'set',
    'in', 'by', 'starring', 'directed'
}
_NEGATION_CONJUNCTIONS = {'and', 'or', 'nor'}
_DEGREE_WORDS = {'too', 'very', 'so', 'overly', 'really', 'particularly', 'that'}
# Nouns that close a negated phrase without being part of it ("no gore scenes").
_NEGATION_TAIL_FILLER = {
    'movie', 'movies', 'film', 'films', 'scene', 'scenes', 'content', 'stuff',
    'elements', 'element', 'ones', 'please', 'at', 'all', 'too', 'much', 'any', 'the', 'a', 'an'
}
_POSITIVE_FILLER = {
    'a', 'an', 'the', 'and', 'or', 'some', 'something', 'anything', 'movie', 'movies',
    'film', 'films', 'that', 'are', 'is', 'have', 'has', 'with', 'me', 'give', 'show',
    'recommend', 'find', 'want', 'i', 'please', 'any', 'good', 'which', 'contain',
    'contains', 'containing', 'feature', 'features', 'featuring', 'can', 'you', 'to', 'watch'
}


def _extract_negated_terms(user_input):
    """
    Splits a prompt into what the user excluded and what they actually want.

    Returns (negated_terms, positive_text):
      'movies without nudity'                 -> (['nudity'], 'movies')
      'nolan movies that are not time travel' -> (['time travel'], 'nolan movies that are')
      'thrillers without gore or nudity'      -> (['gore', 'nudity'], 'thrillers')
    Negated words must never become search terms, so discovery uses positive_text.
    """
    text = (user_input or '').strip()
    tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'&\-]*|[,;.]", text))
    terms = []
    drop = set()

    def take_term(start):
        """Collects up to 3 words of a negated term starting at token index start."""
        words, i = [], start
        while i < len(tokens) and len(words) < 3:
            w = tokens[i].group(0).lower()
            if w in ',;.' or w in _NEGATION_STOPS or w in _NEGATION_CONJUNCTIONS or w in _NEGATION_TRIGGERS:
                break
            drop.add(i)
            if w not in _NEGATION_TAIL_FILLER:
                words.append(w)
            i += 1
        return ' '.join(words), i

    i = 0
    while i < len(tokens):
        word = tokens[i].group(0).lower()
        if word.startswith('non-') and len(word) > 4:
            drop.add(i)
            terms.append(word[4:])
            i += 1
            continue
        if word not in _NEGATION_TRIGGERS:
            i += 1
            continue
        # "not too long", "not very scary" qualify degree; nothing is excluded.
        if i + 1 < len(tokens) and tokens[i + 1].group(0).lower() in _DEGREE_WORDS:
            i += 2
            continue
        drop.add(i)
        term, i = take_term(i + 1)
        if term:
            terms.append(term)
        # Chains: "without gore or nudity", "no horror, romance or comedy"
        while i + 1 < len(tokens):
            sep = tokens[i].group(0).lower()
            nxt = tokens[i + 1].group(0).lower()
            if sep not in _NEGATION_CONJUNCTIONS and sep != ',':
                break
            if nxt in _NEGATION_STOPS or nxt in _NEGATION_TRIGGERS or nxt in ',;.':
                break
            drop.add(i)
            term, i = take_term(i + 1)
            if not term:
                break
            terms.append(term)

    kept = [t.group(0) for idx, t in enumerate(tokens) if idx not in drop and t.group(0) not in ',;.']
    # Drop words left dangling by the removal ("dark sci-fi with [no romance]")
    while kept and kept[-1].lower() in {'with', 'that', 'are', 'is', 'and', 'or', 'but', 'which', 'who', 'have', 'has'}:
        kept.pop()
    positive = ' '.join(kept)
    deduped = list(dict.fromkeys(t for t in terms if t))
    return deduped, positive.strip()


def _has_positive_content(positive_text):
    """True when the non-negated part of a prompt still asks for something."""
    words = re.findall(r"[a-z0-9][a-z0-9'\-]*", (positive_text or '').lower())
    return any(w not in _POSITIVE_FILLER for w in words)


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

    # Second pass: capture arbitrary negated terms not in genre map as excluded keywords.
    # e.g. "not time travel" -> excluded_keywords: ["time travel", "temporal"]
    for clause in clauses:
        clause = clause.strip()
        for match in re.finditer(neg_pattern, clause):
            phrase_matched = match.group(1).strip()
            # Skip if already handled via genre map
            words = phrase_matched.split()
            already_handled = False
            for i in range(1, min(4, len(words) + 1)):
                sub = ' '.join(words[:i]).strip()
                if sub in NEGATION_GENRE_MAP:
                    already_handled = True
                    break
            if already_handled:
                continue
            # Strip leading articles ("the violent ones" -> "violent ones")
            phrase_matched = re.sub(r'^(?:the\s+|a\s+|an\s+)', '', phrase_matched).strip()
            if not phrase_matched:
                continue
            # Add the full phrase and its thematic keyword expansions
            excluded_keywords.add(phrase_matched)
            # Expand via THEMATIC_KEYWORD_MAP if applicable
            for term, kw_list in THEMATIC_KEYWORD_MAP.items():
                if phrase_matched.startswith(term) or term in phrase_matched:
                    excluded_keywords.update(kw_list)

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
    if not m:
        return None
    candidate = m.group(1).strip()
    
    # Strip trailing conjunctions, year/decade qualifiers, temporal prepositions
    candidate = re.sub(r'\s*(?:but|and|however|yet)?\s*(?:from\s+)?(?:before|prior to|older than|after|post|in|during)?\s*(?:the\s*)?\b(?:\d{4}s?|\d{2}s)\b.*$', '', candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r'\s*(?:from\s+before\s+.*|from\s+after\s+.*|but\s+before\s+.*|but\s+after\s+.*)$', '', candidate, flags=re.IGNORECASE).strip()
    # Trailing exclusions belong to the request, not the title ("Drive without gore").
    # Only cut at a negation preceded by a space, so titles like "No Country for Old Men" survive.
    candidate = re.sub(r'\s+(?:but\s+|and\s+)?(?:without|with\s+no|minus|excluding|but\s+not|but\s+no|that\s+(?:is|are)n\'t|that\s+(?:is|are)\s+not)\b.*$', '', candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r'\s+\b(?:but|and|or|yet|with|without)\b\s*$', '', candidate, flags=re.IGNORECASE).strip()
    
    if candidate and len(candidate.split()) <= 6:
        # Don't return generic words like "movies" or "films" as reference entities
        if candidate.lower().strip(' "\'') in ('movies', 'films', 'movie', 'film'):
            return None
        return candidate.strip(' "\'')
    return None


# ── Person Entity Detection ──
# Full names are unambiguous and match anywhere. Surnames and nicknames are common
# words too ("park", "stone", "hardy"), so they only count in a person position:
# "nolan movies", "nolan's films", "directed by nolan", "starring stone".
KNOWN_PEOPLE = {
    'Christopher Nolan': ['christopher nolan', 'chris nolan'],
    'Quentin Tarantino': ['quentin tarantino'],
    'Martin Scorsese': ['martin scorsese'],
    'Stanley Kubrick': ['stanley kubrick'],
    'Steven Spielberg': ['steven spielberg'],
    'James Cameron': ['james cameron'],
    'Akira Kurosawa': ['akira kurosawa'],
    'Alfred Hitchcock': ['alfred hitchcock'],
    'Orson Welles': ['orson welles'],
    'David Fincher': ['david fincher'],
    'Denis Villeneuve': ['denis villeneuve'],
    'Guillermo del Toro': ['guillermo del toro', 'guillermo deltoro'],
    'George Lucas': ['george lucas'],
    'David Lynch': ['david lynch'],
    'Wes Anderson': ['wes anderson'],
    'Paul Thomas Anderson': ['paul thomas anderson'],
    'Ridley Scott': ['ridley scott'],
    'George Miller': ['george miller'],
    'Joel Coen': ['coen brothers', 'the coens', 'joel coen', 'ethan coen'],
    'Woody Allen': ['woody allen'],
    'Hayao Miyazaki': ['hayao miyazaki'],
    'Bong Joon-ho': ['bong joon ho', 'bong joon-ho', 'bong joonho'],
    'Park Chan-wook': ['park chan wook', 'park chan-wook', 'park chanwook'],
    'Alfonso Cuarón': ['alfonso cuaron', 'alfonso cuarón'],
    'Darren Aronofsky': ['darren aronofsky'],
    'Gaspar Noé': ['gaspar noe', 'gaspar noé'],
    'François Truffaut': ['francois truffaut', 'françois truffaut'],
    'Jean-Luc Godard': ['jean-luc godard', 'jean luc godard'],
    'Abbas Kiarostami': ['abbas kiarostami'],
    'Greta Gerwig': ['greta gerwig'],
    'Noah Baumbach': ['noah baumbach'],
    'Barry Jenkins': ['barry jenkins'],
    'Yorgos Lanthimos': ['yorgos lanthimos'],
    'Chloé Zhao': ['chloe zhao', 'chloé zhao'],
    'Lars von Trier': ['lars von trier'],
    'Jean-Pierre Jeunet': ['jean-pierre jeunet', 'jean pierre jeunet'],
    'Nicolas Winding Refn': ['nicolas winding refn'],
    'Zhang Yimou': ['zhang yimou'],
    'Wong Kar-wai': ['wong kar-wai', 'wong kar wai'],
    'John Woo': ['john woo'],
    'Sofia Coppola': ['sofia coppola'],
    'Francis Ford Coppola': ['francis ford coppola'],
    'Ari Aster': ['ari aster'],
    'Jordan Peele': ['jordan peele'],
    'Robert Eggers': ['robert eggers'],
    'Scarlett Johansson': ['scarlett johansson', 'scarlett johanson', 'scarlet johansson', 'scarlett johnson'],
    'Leonardo DiCaprio': ['leonardo dicaprio', 'leo dicaprio', 'leonardo di caprio'],
    'Johnny Depp': ['johnny depp'],
    'Tom Hanks': ['tom hanks'],
    'Brad Pitt': ['brad pitt'],
    'Denzel Washington': ['denzel washington', 'denzen washington'],
    'Keanu Reeves': ['keanu reeves'],
    'Christian Bale': ['christian bale'],
    'Robert Downey Jr.': ['robert downey jr', 'robert downey junior', 'robert downey'],
    'Chris Evans': ['chris evans'],
    'Chris Hemsworth': ['chris hemsworth'],
    'Tom Holland': ['tom holland'],
    'Chris Pratt': ['chris pratt'],
    'Gal Gadot': ['gal gadot'],
    'Saoirse Ronan': ['saoirse ronan'],
    'Emma Stone': ['emma stone'],
    'Tom Hardy': ['tom hardy'],
    'Heath Ledger': ['heath ledger'],
    'Al Pacino': ['al pacino'],
    'Robert De Niro': ['robert de niro', 'robert deniro'],
    'Michael Keaton': ['michael keaton'],
    'Kate Winslet': ['kate winslet'],
    'Paul Newman': ['paul newman'],
    'Robert Redford': ['robert redford'],
    'Daniel Day-Lewis': ['daniel day-lewis', 'daniel day lewis'],
    'Zhang Ziyi': ['zhang ziyi'],
    'Meryl Streep': ['meryl streep'],
    'Cate Blanchett': ['cate blanchett'],
    'Natalie Portman': ['natalie portman'],
    'Ryan Gosling': ['ryan gosling'],
    'Joaquin Phoenix': ['joaquin phoenix'],
    'Florence Pugh': ['florence pugh'],
    'Timothée Chalamet': ['timothee chalamet', 'timothée chalamet'],
    'Margot Robbie': ['margot robbie'],
    'Tom Cruise': ['tom cruise'],
    'Lady Gaga': ['lady gaga'],
}
# Single-word names that only count in a person position
KNOWN_SURNAMES = {
    'nolan': 'Christopher Nolan', 'tarantino': 'Quentin Tarantino', 'scorsese': 'Martin Scorsese',
    'kubrick': 'Stanley Kubrick', 'spielberg': 'Steven Spielberg', 'cameron': 'James Cameron',
    'kurosawa': 'Akira Kurosawa', 'hitchcock': 'Alfred Hitchcock', 'welles': 'Orson Welles',
    'fincher': 'David Fincher', 'villeneuve': 'Denis Villeneuve', 'del toro': 'Guillermo del Toro',
    'lucas': 'George Lucas', 'lynch': 'David Lynch', 'coen': 'Joel Coen', 'coens': 'Joel Coen',
    'miyazaki': 'Hayao Miyazaki', 'bong': 'Bong Joon-ho', 'park': 'Park Chan-wook',
    'cuaron': 'Alfonso Cuarón', 'aronofsky': 'Darren Aronofsky', 'noe': 'Gaspar Noé',
    'truffaut': 'François Truffaut', 'godard': 'Jean-Luc Godard', 'kiarostami': 'Abbas Kiarostami',
    'gerwig': 'Greta Gerwig', 'baumbach': 'Noah Baumbach', 'lanthimos': 'Yorgos Lanthimos',
    'von trier': 'Lars von Trier', 'jeunet': 'Jean-Pierre Jeunet', 'refn': 'Nicolas Winding Refn',
    'yimou': 'Zhang Yimou', 'peele': 'Jordan Peele', 'aster': 'Ari Aster', 'eggers': 'Robert Eggers',
    'johansson': 'Scarlett Johansson', 'johanson': 'Scarlett Johansson', 'scarjo': 'Scarlett Johansson',
    'dicaprio': 'Leonardo DiCaprio', 'depp': 'Johnny Depp', 'hanks': 'Tom Hanks', 'pitt': 'Brad Pitt',
    'denzel': 'Denzel Washington', 'keanu': 'Keanu Reeves', 'bale': 'Christian Bale',
    'pacino': 'Al Pacino', 'de niro': 'Robert De Niro', 'deniro': 'Robert De Niro',
    'winslet': 'Kate Winslet', 'streep': 'Meryl Streep', 'blanchett': 'Cate Blanchett',
    'gosling': 'Ryan Gosling', 'chalamet': 'Timothée Chalamet', 'pugh': 'Florence Pugh',
    'ronan': 'Saoirse Ronan', 'gadot': 'Gal Gadot', 'portman': 'Natalie Portman',
    'phoenix': 'Joaquin Phoenix', 'cruise': 'Tom Cruise', 'robbie': 'Margot Robbie',
    'stone': 'Emma Stone', 'hardy': 'Tom Hardy', 'ledger': 'Heath Ledger', 'keaton': 'Michael Keaton',
}

_PERSON_SUFFIX = r"(?:'s|s')?\s+(?:movies?|films?|flicks?|filmography|pictures|collection)\b"
_DIRECTOR_CUES = r"(?:directed by|direction of|helmed by|a (?:movie|film) by|(?:movies?|films?) by|works? (?:of|by))"
_ACTOR_CUES = r"(?:starring|featuring|stars|acted by|with (?:actor|actress)|played by)"
_NAME = r"([a-z][a-z'.\-]+(?:\s+[a-z][a-z'.\-]+){0,3})"
# Words that end a name: "directed by nolan about time travel"
_NAME_STOPS = {'about', 'from', 'with', 'before', 'after', 'that', 'which', 'set', 'in', 'and', 'or', 'but',
               'movies', 'movie', 'films', 'film', 'where', 'without', 'not', 'no', 'like', 'for', 'during',
               'is', 'are', 'was', 'were', 'please'}
_NOT_NAME_START = {'the', 'a', 'an', 'some', 'any', 'good', 'best', 'new', 'old', 'my', 'all', 'other', 'more'}


def _trim_name(candidate):
    words = []
    for w in candidate.split():
        if w in _NAME_STOPS:
            break
        words.append(w)
    return ' '.join(words[:3])


def _detect_person(user_input):
    """
    Finds a director/actor in a prompt.
    Returns {'name', 'role' ('director' | 'actor' | None), 'matched', 'confident'} or None.
    'matched' is the prompt text naming the person. Confident matches (known names, or a
    name after "directed by"/"starring") are stripped before keyword parsing; guesses
    ("upcoming highly anticipated movies", "Blade Runner") are left in the prompt and only
    used once the recommender confirms the name with TMDB.
    """
    text = (user_input or '').strip()
    low = text.lower()
    if not low:
        return None

    role = None
    if re.search(r'\b' + _DIRECTOR_CUES + r'\s', low):
        role = 'director'
    elif re.search(r'\b' + _ACTOR_CUES + r'\s', low):
        role = 'actor'

    # 1. A known full name anywhere ("christopher nolan", "scarlett johanson")
    best = None
    for canonical, aliases in KNOWN_PEOPLE.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', low) and (best is None or len(alias) > len(best[1])):
                best = (canonical, alias)
    if best:
        return {'name': best[0], 'role': role, 'matched': best[1], 'confident': True}

    # 2. A known surname in a person position ("nolan movies", "directed by nolan")
    cues = r'(?:' + _DIRECTOR_CUES + '|' + _ACTOR_CUES + r'|by)'
    for surname, canonical in sorted(KNOWN_SURNAMES.items(), key=lambda kv: -len(kv[0])):
        s = re.escape(surname)
        if re.search(r'\b' + s + _PERSON_SUFFIX, low) or re.search(r'\b' + cues + r'\s+' + s + r'\b', low):
            return {'name': canonical, 'role': role, 'matched': surname, 'confident': True}

    # 3. Any other name after a cue ("directed by sean baker", "starring anya taylor-joy")
    #    or in front of "movies"/"films" ("david lynch movies"). Topic words are rejected
    #    ("haunted house movies"); the recommender then confirms the name with TMDB.
    from backend.concepts import analyze_concepts, FILLER
    candidates = []
    for cue, cue_role in ((_DIRECTOR_CUES, 'director'), (_ACTOR_CUES, 'actor')):
        for m in re.finditer(r'\b' + cue + r'\s+' + _NAME, low):
            candidates.append((_trim_name(m.group(1)), cue_role, True))
    for m in re.finditer(r'\b' + _NAME + _PERSON_SUFFIX, low):
        # Only the words right before "movies", up to a stop word ("sad lynch movies" -> "sad lynch")
        words = m.group(1).split()
        tail = []
        for w in reversed(words):
            if w in _NAME_STOPS:
                break
            tail.insert(0, w)
        candidates.append((' '.join(tail[-3:]), role, False))

    for name, cand_role, after_cue in candidates:
        words = name.split()
        if not words or words[0] in _NEGATION_TRIGGERS or words[0] in _NOT_NAME_START:
            continue
        # A single word before "movies" is usually a topic ("heist movies") unless capitalized
        if len(words) == 1 and not after_cue and not re.search(r'\b' + re.escape(words[0].capitalize()) + r'\b', text):
            continue
        covered = analyze_concepts(name)['covered']
        if any(w in FILLER or w in covered or w in LANGUAGE_MAP for w in words):
            continue
        original = re.search(re.escape(name), text, re.IGNORECASE)
        display = original.group(0) if original else name
        if display.islower():
            display = ' '.join(w[:1].upper() + w[1:] for w in display.split())
        return {'name': display, 'role': cand_role, 'matched': name, 'confident': after_cue}

    # 4. The whole prompt is a capitalized name ("David Lynch", "Emma Stone")
    m = re.fullmatch(r"([A-Z][a-z'\-]+(?:\s+(?:van|von|de|del|da|di|le)?\s*[A-Z][a-z'\-]+){1,2})", text)
    if m and m.group(1).split()[0].lower() not in _NEGATION_TRIGGERS | _NOT_NAME_START:
        return {'name': m.group(1), 'role': None, 'matched': m.group(1).lower(), 'confident': False}
    return None


def _extract_person_entity(user_input):
    """Canonical person name in a prompt, or None. See _detect_person."""
    found = _detect_person(user_input)
    return found['name'] if found else None


def _strip_person_from_prompt(user_input, person_name):
    """
    Removes the person entity from the raw query string before keyword parsing,
    so that the surname doesn't get misinterpreted as a mood/genre keyword.
    """
    if not person_name:
        return user_input
    text = user_input or ''
    # Strip full name (case-insensitive)
    text = re.sub(re.escape(person_name), '', text, flags=re.IGNORECASE)
    # Strip surname (last word of the name) — e.g. "Nolan" when name is "Christopher Nolan"
    surname = person_name.split()[-1] if person_name else ''
    if surname and len(surname) > 2:
        text = re.sub(r'\b' + re.escape(surname) + r'\b', '', text, flags=re.IGNORECASE)
    # Remove possessive patterns like "nolan's movies"
    text = re.sub(r"'s\s+movie", " movie", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmovie\s+by\s+", "movie ", text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s{2,}', ' ', text).strip()
    return cleaned


def _extract_runtime_constraints(user_input):
    """
    Deterministic extraction of runtime constraints from prompt.
    Detects "under 90 min", "less than 90 min", "under 1.5 hours", "short movies" etc.
    Returns runtime_max (int or None).
    """
    text = (user_input or '').lower().strip()

    # Pattern: under/less than X min(s)
    m = re.search(r'(?:under|less than|shorter than|below|at most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*(?:mins?|minutes?|m)\b', text)
    if m:
        val = float(m.group(1))
        # If it's a word like "ninety", we won't catch it here, but numbers work
        return int(val) if val == int(val) else int(val)

    # Pattern: under X hours / less than X hours
    m = re.search(r'(?:under|less than|shorter than|below)\s*(\d+(?:\.\d+)?)\s*(?:hours?|hr|hrs|h)\b', text)
    if m:
        hours = float(m.group(1))
        return int(hours * 60)

    # Pattern: "90 minutes or less", "at most 2 hours"
    m = re.search(r'(?:at most|no more than|within)\s*(\d+(?:\.\d+)?)\s*(mins?|minutes?|hours?|hr|hrs|h)\b', text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith('h'):
            return int(val * 60)
        return int(val)

    # Pattern: "X min or shorter" / "X minutes or less"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mins?|minutes?)\s*or\s*(?:shorter|less)', text)
    if m:
        return int(float(m.group(1)))

    # Pattern: "quick movies", "short films", "short movies", "quick films", "under 2 hours" already covered
    if re.search(r'\b(?:short|quick|fast|brief|rapid)\s+(?:movie|film|movies|films)\b', text):
        # Default short threshold: 90 minutes
        return 90

    # Pattern: "less than 2 hours" already covered above
    # Pattern: "1.5 hours" without qualifier but with "under"
    m = re.search(r'under\s+(\d+(?:\.\d+)?)\s*hours?', text)
    if m:
        return int(float(m.group(1)) * 60)

    return None


def _extract_quality_constraints(user_input):
    """
    Deterministic extraction of quality constraints from prompt.
    Detects "critically acclaimed", "masterpiece", "top rated", "award winning" etc.
    Returns vote_average_min (float or None).
    """
    text = (user_input or '').lower().strip()

    # Acclaimed / masterpiece / award winning / top rated patterns
    if re.search(r'\b(?:critically\s+acclaimed|acclaimed|masterpiece|award\s*winning|award\s+nominated|top\s+rated|highly\s+rated|best\s+picture|palme\s+d\'or|oscar|nominated|won\s+(?:an\s+)?award)\b', text):
        return 7.5

    if re.search(r'\b(?:classics?|essential\s+cinema|must-see|must\s+see)\b', text):
        return 7.5

    if re.search(r'\b(?:legendary|gREATEST|all.?time\s+best)\b', text):
        return 8.0

    return None


def _clean_search_query_with_person(user_input, person_name):
    """
    Clean the search query removing the person entity so TMDB title search
    doesn't pick up "Christopher Nolan" as a movie title.
    When a person is detected, the search query is set to empty — person-based
    discovery uses TMDB's /discover/movie?with_people parameter instead.
    """
    if person_name:
        return ""
    text = user_input or ''
    return text.strip()


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
        'anime': ['Animation'],
        'animated': ['Animation'],
        'animation': ['Animation'],
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

def rank_watchlist_relevance(user_input, watchlist_movies):
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

def generate_matchmaker_pitch(movie):
    """Builds the recommendation pitch for the winning matchmaker pick."""
    title = movie.get('title') or 'This film'
    year = str(movie.get('year') or '').replace('.0', '')
    genres = movie.get('genres') or ''
    runtime = movie.get('runtime') or ''
    score = movie.get('ai_score') or 3.8
    if isinstance(genres, list):
        genres = ", ".join(genres)

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

    if re.search(r'\b(?:heist|bank robbery|caper)\b', text):
        groups.append(['heist', 'caper'])

    return groups if len(groups) >= 2 else []

def interpret_query(user_input):
    """
    Deterministic rule-based query parser.
    Extracts TMDB genres, search query, year bounds, ISO language codes,
    thematic keywords, reference entity, excluded genres/keywords, and upcoming release intent.
    """
    text = (user_input or '').strip()

    # Reference entity extraction first — "something like X" / "similar to X"
    # must not be confused with a person name ("The Matrix" has two caps words).
    det_ref_entity = _extract_reference_entity(text)

    # A referenced title is not a negation ("something like No Country for Old Men"),
    # so negations are only read from the rest of the prompt.
    negation_source = text
    if det_ref_entity:
        negation_source = re.sub(re.escape(det_ref_entity), ' ', text, count=1, flags=re.IGNORECASE)
    negations = _extract_negations(negation_source)
    excluded_genres = negations['genres']
    excluded_keywords = negations['keywords']

    # Person entity extraction (director/actor name or surname)
    found_person = None if det_ref_entity else _detect_person(text)
    det_person = found_person['name'] if found_person else None
    text_with_person = text
    if found_person and found_person['confident']:
        # Remove exactly what named the person (typos included) and the cue before it
        text = re.sub(r"\b(?:directed by|starring|featuring|movies by|films by|by)?\s*" + re.escape(found_person['matched']) + r"(?:'s)?\b",
                      ' ', text, count=1, flags=re.IGNORECASE)
        text = _strip_person_from_prompt(text, det_person)

    # Everything that describes what to find works on the non-negated part only,
    # so "without nudity" can never turn "nudity" into a genre, keyword or title search.
    placeholder = 'refentityplaceholder'
    protected = re.sub(re.escape(det_ref_entity), placeholder, text, count=1, flags=re.IGNORECASE) if det_ref_entity else text
    negated_terms, positive_text = _extract_negated_terms(protected)
    if det_ref_entity:
        negated_terms = [t for t in negated_terms if placeholder not in t]
        positive_text = positive_text.replace(placeholder, det_ref_entity)
    # Title search keeps the detected person: the detector can mistake a title
    # ("Fight Club") for a name, and the full text still finds the film.
    positive_query = _extract_negated_terms(text_with_person)[1] if det_person else positive_text
    excluded_keywords = list(dict.fromkeys(list(excluded_keywords) + negated_terms))
    has_positive = _has_positive_content(positive_text)

    det_ymin, det_ymax = _extract_year_constraints(text)
    is_upcoming = _is_upcoming_query(positive_text)
    det_langs = _extract_languages(positive_text, excluded_keywords=excluded_keywords)
    det_kws = _extract_thematic_keywords(positive_text, excluded_keywords=excluded_keywords)
    det_genres = _fallback_mood_match(positive_text, excluded_genres=excluded_genres) if has_positive else []
    det_compound_groups = _extract_compound_keyword_groups(positive_text)
    det_runtime_max = _extract_runtime_constraints(text)
    det_vote_avg_min = _extract_quality_constraints(positive_text)
    # Clean search query — strip person name so TMDB title search is not confused
    if det_person:
        det_search_query = _clean_search_query_with_person(text, det_person)
    elif det_ref_entity:
        det_search_query = _clean_search_query(text, ref_entity=det_ref_entity, is_upcoming=is_upcoming)
    elif has_positive:
        det_search_query = _clean_search_query(positive_text, is_upcoming=is_upcoming)
    else:
        det_search_query = ''

    return {
        'positive_query': positive_query,
        'negated_terms': negated_terms,
        'exclusion_only': not has_positive and not det_person and not det_ref_entity,
        'genres': det_genres,
        'search_query': det_search_query,
        'year_min': det_ymin,
        'year_max': det_ymax,
        'is_upcoming': is_upcoming,
        'languages': det_langs,
        'reference_entity': det_ref_entity,
        'thematic_keywords': det_kws,
        'compound_keyword_groups': det_compound_groups,
        'excluded_genres': excluded_genres,
        'excluded_keywords': excluded_keywords,
        'person': det_person,
        'person_role': found_person['role'] if found_person else None,
        'person_matched': found_person['matched'] if found_person else None,
        'runtime_max': det_runtime_max,
        'vote_average_min': det_vote_avg_min
    }

