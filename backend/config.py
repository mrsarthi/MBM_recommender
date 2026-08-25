import os
import sys
import shutil
from dotenv import load_dotenv
import google.generativeai as genai

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

TMDB_KEY = os.getenv('TMDB_key')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("[OK] Gemini AI initialized (gemini-2.5-flash).")
    except Exception as e:
        print(f"[WARN] Gemini init warning: {e}")
else:
    print("[WARN] GEMINI_API_KEY not set. Using mood mapping fallback.")

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

# Sync fallback from workspace user_data if AppData empty
local_profile = os.path.join(BASE_DIR, 'user_data', 'user_profile.csv')
if os.path.exists(local_profile) and (not os.path.exists(PROFILE_PATH) or os.path.getsize(PROFILE_PATH) == 0):
    try:
        shutil.copy2(local_profile, PROFILE_PATH)
    except Exception: pass

local_watchlist = os.path.join(BASE_DIR, 'user_data', 'watchlist.csv')
if os.path.exists(local_watchlist) and (not os.path.exists(WATCHLIST_PATH) or os.path.getsize(WATCHLIST_PATH) == 0):
    try:
        shutil.copy2(local_watchlist, WATCHLIST_PATH)
    except Exception: pass
