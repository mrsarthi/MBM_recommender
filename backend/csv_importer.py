import io
import re
import pandas as pd

from backend.db import (
    get_or_create_user, upsert_movies_batch, upsert_user_diary,
    upsert_user_watchlist, stable_movie_id
)
from backend.in_memory_model import train_user_model_in_memory

# Columns that only appear in a profile CSV this app itself enriched. When they are
# present the film metadata is already complete and TMDB never needs to be called.
ENRICHED_COLUMNS = {'movie_id', 'genres', 'overview'}


def _clean(val, default=''):
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in ('nan', 'none', ''):
        return default
    return s


def _slug_from_uri(uri):
    """Letterboxd exports use short boxd.it links; full URLs carry the film slug."""
    m = re.search(r'/film/([^/]+)/?', str(uri or ''))
    return m.group(1) if m else ''


def _normalize_date(val):
    """
    Coerces a date cell to YYYY-MM-DD, or '' if it is not a usable date.

    Real exports occasionally carry a malformed cell (a shifted column, a stray URL).
    Those must not reach the database, where watched_date is a VARCHAR(30).
    """
    s = _clean(val)
    if not s:
        return ''
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return m.group(0)
    try:
        parsed = pd.to_datetime(s, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except Exception:
        pass
    return ''


def _cache_key(slug, title, year):
    """Stable identity for a film so re-imports reuse the same movies row."""
    return slug or "{0}-{1}".format(re.sub(r'[^a-z0-9]+', '-', str(title).lower()).strip('-'), year or '')


def parse_letterboxd_csv(csv_content, is_watchlist=False):
    """
    Parses any Letterboxd export CSV (ratings.csv, diary.csv, watched.csv,
    watchlist.csv) or a profile CSV this app previously enriched.

    Returns (entries, enriched_records). `enriched_records` is non-empty only for the
    already-enriched format, in which case the films need no TMDB lookup at all.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        raise ValueError("Failed to parse CSV: {0}".format(e))

    df.columns = [c.strip() for c in df.columns]

    title_col = next((c for c in ('Name', 'Title') if c in df.columns), None)
    if not title_col:
        raise ValueError("CSV must contain a 'Name' or 'Title' column.")

    year_col = next((c for c in ('Year', 'Release Date') if c in df.columns), None)
    rating_col = next((c for c in ('Rating', 'Rating10') if c in df.columns), None)
    # diary.csv has both 'Date' (logged) and 'Watched Date'; the latter is the real one.
    date_col = next((c for c in ('Watched Date', 'Date') if c in df.columns), None)
    uri_col = next((c for c in ('Letterboxd URI', 'URI') if c in df.columns), None)

    is_enriched = ENRICHED_COLUMNS.issubset(set(df.columns))

    entries = []
    enriched_records = []

    for _, row in df.iterrows():
        title = _clean(row.get(title_col))
        if not title:
            continue

        year = _clean(row.get(year_col)) if year_col else ''
        year = year.split('-')[0].replace('.0', '').strip()

        rating = None
        if rating_col:
            raw = row.get(rating_col)
            if pd.notna(raw) and _clean(raw):
                try:
                    r_val = float(raw)
                    # Rating10 is a 0-10 scale; Letterboxd stars are 0-5.
                    if rating_col == 'Rating10':
                        r_val = r_val / 2.0
                    rating = round(max(0.5, min(5.0, r_val)), 1)
                except (TypeError, ValueError):
                    rating = None

        watched_date = _normalize_date(row.get(date_col)) if date_col else ''

        slug = _slug_from_uri(row.get(uri_col)) if uri_col else ''
        key = _cache_key(slug, title, year)

        if is_enriched:
            try:
                m_id = int(float(row.get('movie_id')))
            except (TypeError, ValueError):
                m_id = stable_movie_id(key)
            enriched_records.append({
                'movie_id': m_id,
                'title': title,
                'year': year,
                'genres': _clean(row.get('genres'), 'General'),
                'overview': _clean(row.get('overview')),
                'director': _clean(row.get('director')),
                'cast': _clean(row.get('cast')),
                'keywords': _clean(row.get('keywords')),
                'runtime': 0,
                'vote_average': 7.0,
                'poster_path': _clean(row.get('poster_path')),
                'backdrop_path': _clean(row.get('backdrop_path')),
                'letterboxd_slug': slug,
            })
            entries.append({
                'slug': key, 'title': title, 'year_hint': year,
                'rating': rating, 'watched_date': watched_date,
                'resolved_id': m_id,
            })
        else:
            entries.append({
                'slug': key, 'title': title, 'year_hint': year,
                'rating': rating, 'watched_date': watched_date,
            })

    if not entries:
        raise ValueError("No valid movie entries found in the CSV.")

    return entries, enriched_records


def import_letterboxd_csv(username, csv_content, is_watchlist=False, tmdb_key=None,
                          job_id=None, progress_cb=None):
    """
    Imports a Letterboxd export CSV into Neon for the given user.

    Films are matched against TMDB so posters, genres, overviews, directors and
    keywords are populated - without that the diary renders as blank cards and the
    taste model has nothing but a title to learn from.
    """
    # Imported here to avoid a circular import at module load (jobs imports this module).
    from backend.jobs import resolve_entries

    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user:
        return False, "Username is required."

    user = get_or_create_user(clean_user)
    if not user:
        return False, "User creation failed."

    try:
        entries, enriched_records = parse_letterboxd_csv(csv_content, is_watchlist=is_watchlist)
    except ValueError as e:
        return False, str(e)

    if progress_cb:
        progress_cb(20, "Parsed {0} films from CSV...".format(len(entries)))

    slug_to_id = {}
    movie_records = []

    if enriched_records:
        # Already-enriched profile export: no network calls needed.
        movie_records = enriched_records
        slug_to_id = {e['slug']: e['resolved_id'] for e in entries if e.get('resolved_id')}
    else:
        movie_records, slug_to_id = resolve_entries(
            entries, tmdb_key, job_id=job_id, base_progress=25, span=50,
            label='watchlist films' if is_watchlist else 'diary films'
        )

    if progress_cb:
        progress_cb(80, "Saving {0} films to the database...".format(len(movie_records)))

    if movie_records:
        upsert_movies_batch(movie_records)

    links = []
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    for e in entries:
        m_id = slug_to_id.get(e['slug'])
        if not m_id:
            continue
        if is_watchlist:
            links.append({'movie_id': m_id, 'added_date': e.get('watched_date') or today})
        else:
            links.append({
                'movie_id': m_id,
                'rating': e.get('rating'),
                'watched_date': e.get('watched_date') or ''
            })

    if not links:
        return False, "Could not match any films from the CSV."

    if is_watchlist:
        count = upsert_user_watchlist(user['id'], links)
        return True, "Imported {0} films into @{1}'s watchlist.".format(count, clean_user)

    count = upsert_user_diary(user['id'], links)
    if progress_cb:
        progress_cb(92, "Calibrating your AI taste model...")
    model, _c, _v, _e = train_user_model_in_memory(clean_user)
    note = "AI model calibrated" if model is not None else "Not enough rated films to train yet"
    return True, "Imported {0} diary films for @{1}. {2}.".format(count, clean_user, note)
