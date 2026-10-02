"""
Search relevance benchmark (live TMDB, not part of the unit test run).

Each case states what a correct result must satisfy, checked against TMDB's own
metadata (genres, keyword tags, release year, language, credits). The score for
a prompt is precision@10: the share of the top 10 results that pass every check.

Run:  .venv\\Scripts\\python.exe tests\\query_benchmark.py [--set tuned|heldout|all] [--user USERNAME]
          [--source all|watchlist] [--verbose] [--only TEXT]

--source watchlist ranks USERNAME's watchlist instead of discovering new films. Scores
there are capped by what the watchlist contains, so compare runs, not absolute values.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import TMDB_KEY, TMDB_BASE_URL
from backend.query_parser import interpret_query
from backend.recommender import analyze, http_session, titleNormalize

GENRES = {28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime', 99: 'Documentary',
          18: 'Drama', 10751: 'Family', 14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
          9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction', 10770: 'TV Movie', 53: 'Thriller',
          10752: 'War', 37: 'Western'}

SPORT_TAGS = ['sport', 'football', 'boxing', 'basketball', 'baseball', 'soccer', 'tennis', 'racing',
              'olympic', 'hockey', 'surfing', 'wrestling', 'skateboard', 'golf', 'running', 'swimming']

# genres: list of groups; a result must have at least one genre from every group
CASES = [
    {'prompt': 'scary movies without gore', 'genres': [['Horror']], 'forbid_tags': ['gore']},
    {'prompt': 'dark sci-fi with no romance', 'genres': [['Science Fiction']], 'forbid_genres': ['Romance']},
    {'prompt': 'funny animated movies', 'genres': [['Animation'], ['Comedy']]},
    {'prompt': '90s crime thrillers', 'genres': [['Crime', 'Thriller']], 'years': (1990, 1999)},
    {'prompt': 'korean revenge thrillers', 'language': 'ko', 'tags': ['revenge']},
    {'prompt': 'time travel movies', 'tags': ['time travel', 'time loop', 'time machine']},
    {'prompt': 'heist movies', 'tags': ['heist', 'robbery', 'thief']},
    {'prompt': 'zombie comedies', 'genres': [['Comedy']], 'tags': ['zombie']},
    {'prompt': 'space horror', 'genres': [['Horror'], ['Science Fiction']]},
    {'prompt': 'romantic comedies from the 2000s', 'genres': [['Romance'], ['Comedy']], 'years': (2000, 2009)},
    {'prompt': 'feel-good sports movies', 'tags': SPORT_TAGS, 'forbid_genres': ['Horror']},
    {'prompt': 'christmas movies', 'tags': ['christmas']},
    {'prompt': 'coming of age movies', 'tags': ['coming of age', 'teenager', 'adolescence', 'growing up']},
    {'prompt': 'mind-bending sci-fi', 'genres': [['Science Fiction']]},
    {'prompt': 'westerns before 1970', 'genres': [['Western']], 'years': (None, 1969)},
    {'prompt': 'war movies', 'genres': [['War']]},
    {'prompt': 'slow burn psychological thrillers', 'genres': [['Thriller', 'Mystery', 'Horror', 'Drama']]},
    {'prompt': 'christopher nolan movies', 'person': 525},
    {'prompt': 'japanese animated movies', 'genres': [['Animation']], 'language': 'ja'},
    {'prompt': 'cozy mystery movies', 'genres': [['Mystery']], 'forbid_genres': ['Horror']},
    {'prompt': 'gritty crime dramas', 'genres': [['Crime']]},
    {'prompt': 'superhero movies', 'tags': ['superhero', 'super power', 'based on comic']},
    {'prompt': 'movies about artificial intelligence', 'tags': ['artificial intelligence', 'robot', 'android', 'a.i.', 'cyborg']},
    {'prompt': 'haunted house movies', 'genres': [['Horror', 'Thriller', 'Mystery']], 'tags': ['haunted house', 'haunting', 'ghost', 'haunted']},
    {'prompt': 'dystopian movies', 'tags': ['dystopia', 'dystopian', 'post-apocalyptic', 'totalitarian']},
    {'prompt': 'musicals', 'genres': [['Music']]},
    {'prompt': 'vampire movies', 'tags': ['vampire']},
    {'prompt': 'feel good movies', 'forbid_genres': ['Horror', 'War', 'Crime'],
     'genres': [['Comedy', 'Family', 'Romance', 'Animation', 'Music', 'Drama', 'Adventure']]},
]

# Held-out prompts: not used while tuning the concept lexicon, to catch overfitting.
HELDOUT = [
    {'prompt': 'something creepy to watch tonight', 'genres': [['Horror', 'Thriller']]},
    {'prompt': 'lighthearted family adventure', 'genres': [['Family', 'Adventure', 'Animation', 'Comedy']], 'forbid_genres': ['Horror']},
    {'prompt': 'bleak war dramas', 'genres': [['War']]},
    {'prompt': '80s action movies', 'genres': [['Action']], 'years': (1980, 1989)},
    {'prompt': 'french romantic dramas', 'genres': [['Romance']], 'language': 'fr'},
    {'prompt': 'gangster movies', 'tags': ['mafia', 'gangster', 'organized crime', 'mob', 'cartel', 'crime boss']},
    {'prompt': 'movies about aliens', 'tags': ['alien', 'extraterrestrial', 'ufo']},
    {'prompt': 'sad movies that will make me cry', 'genres': [['Drama', 'Romance']]},
    {'prompt': 'spy thrillers', 'tags': ['spy', 'espionage', 'secret agent', 'cia', 'mi6', 'kgb']},
    {'prompt': 'courtroom dramas', 'tags': ['court', 'lawyer', 'trial', 'attorney', 'judge', 'legal']},
    {'prompt': 'survival movies in the wilderness', 'tags': ['survival', 'wilderness', 'stranded', 'forest', 'mountain', 'plane crash']},
    {'prompt': 'dinosaur movies', 'tags': ['dinosaur']},
    {'prompt': 'boxing movies', 'tags': ['boxing', 'boxer']},
    {'prompt': 'satirical comedies', 'genres': [['Comedy']]},
    {'prompt': 'psychological horror', 'genres': [['Horror']]},
    {'prompt': 'spielberg movies', 'person': 488},
    {'prompt': 'greta gerwig films', 'person': 45400},
    {'prompt': 'movies starring tom hanks', 'person': 31},
    {'prompt': 'historical epics', 'genres': [['History', 'War', 'Adventure', 'Drama']]},
    {'prompt': 'disaster movies', 'tags': ['disaster', 'earthquake', 'tsunami', 'volcano', 'asteroid', 'flood', 'storm', 'catastrophe']},
    {'prompt': 'detective noir', 'genres': [['Crime', 'Mystery', 'Thriller']]},
    {'prompt': 'post-apocalyptic movies', 'tags': ['apocalyp', 'end of the world', 'nuclear', 'wasteland']},
    {'prompt': 'submarine movies', 'tags': ['submarine']},
    {'prompt': 'movies set in tokyo', 'tags': ['tokyo']},
    {'prompt': 'Inception', 'first_id': 27205},
    {'prompt': 'Heat', 'first_id': 949},
    {'prompt': 'Alien', 'first_id': 348},
]

_tag_cache = {}
_credit_cache = {}


def movie_tags(mid):
    if mid not in _tag_cache:
        r = http_session.get(f"{TMDB_BASE_URL}/movie/{mid}/keywords", params={'api_key': TMDB_KEY}, timeout=8).json()
        _tag_cache[mid] = [k['name'].lower() for k in r.get('keywords', [])]
    return _tag_cache[mid]


def person_movie_ids(pid):
    if pid not in _credit_cache:
        r = http_session.get(f"{TMDB_BASE_URL}/person/{pid}/movie_credits", params={'api_key': TMDB_KEY}, timeout=8).json()
        _credit_cache[pid] = {c['id'] for c in r.get('crew', []) if c.get('job') == 'Director'} | \
                             {c['id'] for c in r.get('cast', [])}
    return _credit_cache[pid]


def genre_names(m):
    names = {GENRES.get(g) for g in m.get('genre_ids', [])}
    if isinstance(m.get('genres'), list):
        names |= {g if isinstance(g, str) else g.get('name') for g in m['genres']}
    elif isinstance(m.get('genres'), str):
        names |= {g.strip() for g in m['genres'].split(',')}
    return {n for n in names if n}


def check(m, case):
    """Returns a list of failed checks (empty means the result is correct)."""
    fails = []
    genres = genre_names(m)
    for group in case.get('genres', []):
        if not genres & set(group):
            fails.append(f"genre∉{group}")
    if genres & set(case.get('forbid_genres', [])):
        fails.append('forbidden genre')
    if 'tags' in case or 'forbid_tags' in case:
        tags = movie_tags(m['id'])
        if 'tags' in case and not any(any(t in tag for t in case['tags']) for tag in tags):
            fails.append('no matching tag')
        if any(any(t in tag for t in case.get('forbid_tags', [])) for tag in tags):
            fails.append('forbidden tag')
    lo, hi = case.get('years', (None, None))
    year = str(m.get('release_date') or m.get('year') or '')[:4]
    if (lo or hi) and year.isdigit():
        if (lo and int(year) < lo) or (hi and int(year) > hi):
            fails.append(f'year {year}')
    if case.get('language') and m.get('original_language') and m['original_language'] != case['language']:
        fails.append(f"lang {m['original_language']}")
    if case.get('person') and m['id'] not in person_movie_ids(case['person']):
        fails.append('not in filmography')
    return fails


def run(user=None, verbose=False, only=None, cases=None, source='all'):
    model = (None, None, None, None)
    watched_t, watched_i = set(), set()
    if user:
        from backend.in_memory_model import get_or_train_user_model
        from backend.db import get_user_diary
        model = get_or_train_user_model(user)
        rows, _, _ = get_user_diary(user)
        watched_t = {titleNormalize(r['title']) for r in rows if r.get('title')}
        watched_i = {r['movie_id'] for r in rows if r.get('movie_id')}

    scores = []
    for case in (cases or CASES):
        if only and only not in case['prompt']:
            continue
        prompt = case['prompt']
        if source == 'watchlist' and ('person' in case or 'first_id' in case):
            continue
        picks = analyze(watched_t, watched_i, [], interpret_query(prompt), *model,
                        raw_prompt=prompt, source=source, username=user, tmdb_key=TMDB_KEY)
        top = picks[:10]
        if 'first_id' in case:
            top = picks[:1]
            results = [(m, [] if m['id'] == case['first_id'] else ['wrong top result']) for m in top]
            p10 = 1.0 if results and not results[0][1] else 0.0
        else:
            results = [(m, check(m, case)) for m in top]
            p10 = sum(1 for _, f in results if not f) / 10 if top else 0.0
        scores.append(p10)
        print(f"{p10:4.1f}  {prompt}")
        if verbose:
            for m, f in results:
                mark = '✓' if not f else '✗'
                print(f"        {mark} {m.get('title', '')[:40]:40} {str(m.get('release_date', ''))[:4]} {', '.join(f)}")
    print(f"\nmean precision@10: {sum(scores) / max(1, len(scores)):.3f} over {len(scores)} prompts")
    return scores


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--user')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--set', choices=['tuned', 'heldout', 'all'], default='tuned')
    ap.add_argument('--source', choices=['all', 'watchlist'], default='all',
                    help='watchlist ranks the --user watchlist instead of discovering new films')
    args = ap.parse_args()
    chosen = {'tuned': CASES, 'heldout': HELDOUT, 'all': CASES + HELDOUT}[args.set]
    run(args.user, args.verbose, args.only, chosen, args.source)
