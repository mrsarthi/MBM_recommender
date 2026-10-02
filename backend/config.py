import os
import sys
import hashlib
import base64
from dotenv import load_dotenv

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

TMDB_KEY = os.getenv('TMDB_KEY') or os.getenv('TMDB_key') or os.getenv('TMDB_API_KEY', '')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

LETTERBOXD_USERNAME = os.getenv('LETTERBOXD_USERNAME', '')
DATABASE_URL = os.getenv('DATABASE_URL', '')

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    derived = hashlib.pbkdf2_hmac('sha256', (DATABASE_URL or 'mbmr_secure_instance').encode('utf-8'), b'mbmr_fixed_salt_2026_prod', 100000)
    ENCRYPTION_KEY = base64.urlsafe_b64encode(derived).decode('utf-8')

SESSION_SECRET = os.getenv('SESSION_SECRET')
if not SESSION_SECRET:
    SESSION_SECRET = hashlib.sha256((ENCRYPTION_KEY + "_session_salt").encode('utf-8')).hexdigest()

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

def get_user_data_path(filename):
    appdata = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'MBM_Recommender')
    os.makedirs(appdata, exist_ok=True)
    full_path = os.path.join(appdata, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

PROFILE_PATH = get_user_data_path('user_data/user_profile.csv')
FEATURES_PATH = get_user_data_path('user_data/user_profile_features.csv')
MODEL_PATH = get_user_data_path('user_data/personal_ai_model.pkl')
COLUMNS_PATH = get_user_data_path('user_data/model_columns.pkl')
VECTORIZER_PATH = get_user_data_path('user_data/summary_vectorizer.pkl')
ENCODERS_PATH = get_user_data_path('user_data/feature_encoders.pkl')
