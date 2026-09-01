import os
import sys
import shutil
from dotenv import load_dotenv
from google import genai

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

TMDB_KEY = os.getenv('TMDB_key')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

LETTERBOXD_USERNAME = os.getenv('LETTERBOXD_USERNAME', '')
DATABASE_URL = os.getenv('DATABASE_URL', '')

import hashlib
import base64

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    derived = hashlib.pbkdf2_hmac('sha256', (DATABASE_URL or 'mbmr_secure_instance').encode('utf-8'), b'mbmr_fixed_salt_2026_prod', 100000)
    ENCRYPTION_KEY = base64.urlsafe_b64encode(derived).decode('utf-8')

SESSION_SECRET = os.getenv('SESSION_SECRET')
if not SESSION_SECRET:
    SESSION_SECRET = hashlib.sha256((ENCRYPTION_KEY + "_session_salt").encode('utf-8')).hexdigest()

if not TMDB_KEY or TMDB_KEY == 'YOUR_TMDB_API_KEY_HERE':
    print("[WARN] TMDB_key not set in .env. Search, posters, and ripple recommendations require a valid TMDB key.")
    print("       Get a free TMDB API key at: https://www.themoviedb.org/settings/api")

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[OK] Google GenAI client initialized.")
    except Exception as e:
        print(f"[WARN] Gemini init warning: {e}")
else:
    print("[INFO] GEMINI_API_KEY not set. Using local mood & genre heuristics fallback.")

gemini_model = gemini_client

def get_user_data_path(filename):
    appdata = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'MBM_Recommender')
    os.makedirs(appdata, exist_ok=True)
    full_path = os.path.join(appdata, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

CONFIG_FILE = get_user_data_path('config.json')
APP_MEMORY_FILE = get_user_data_path('app_memory_ids.csv')
PROFILE_PATH = get_user_data_path('user_data/user_profile.csv')
FEATURES_PATH = get_user_data_path('user_data/user_profile_features.csv')
MODEL_PATH = get_user_data_path('user_data/personal_ai_model.pkl')
COLUMNS_PATH = get_user_data_path('user_data/model_columns.pkl')
VECTORIZER_PATH = get_user_data_path('user_data/summary_vectorizer.pkl')
ENCODERS_PATH = get_user_data_path('user_data/feature_encoders.pkl')
WATCHLIST_PATH = get_user_data_path('user_data/watchlist.csv')

def get_user_profile_path(username=None):
    clean = (username or '').strip().lstrip('@').lower()
    if clean and clean != 'guest':
        p = get_user_data_path(f'user_data/profiles/{clean}_profile.csv')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    if LETTERBOXD_USERNAME:
        return PROFILE_PATH
    return get_user_data_path('user_data/empty_profile.csv')

def get_user_watchlist_path(username=None):
    clean = (username or '').strip().lstrip('@').lower()
    if clean and clean != 'guest':
        p = get_user_data_path(f'user_data/profiles/{clean}_watchlist.csv')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    if LETTERBOXD_USERNAME:
        return WATCHLIST_PATH
    return get_user_data_path('user_data/empty_watchlist.csv')

# Sync fallback from workspace user_data if AppData empty and developer username is set
if LETTERBOXD_USERNAME:
    local_profile = os.path.join(BASE_DIR, 'user_data', 'user_profile.csv')
    if os.path.exists(local_profile) and (not os.path.exists(PROFILE_PATH) or os.path.getsize(PROFILE_PATH) == 0):
        try: shutil.copy2(local_profile, PROFILE_PATH)
        except Exception: pass

    local_watchlist = os.path.join(BASE_DIR, 'user_data', 'watchlist.csv')
    if os.path.exists(local_watchlist) and (not os.path.exists(WATCHLIST_PATH) or os.path.getsize(WATCHLIST_PATH) == 0):
        try: shutil.copy2(local_watchlist, WATCHLIST_PATH)
        except Exception: pass
