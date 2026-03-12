import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import traceback
import re
import json
import webbrowser
import io
import threading
import shutil
from PIL import Image, ImageTk
import joblib
import numpy as np
import requests_cache
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import google.generativeai as genai

# Use a non-interactive backend for Tkinter embedding
matplotlib.use('TkAgg')

class NullWriter:
    def write(self, text): pass
    def flush(self): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = NullWriter()
elif hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if sys.stderr is None:
    sys.stderr = NullWriter()
elif hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Local Imports for ML Pipeline
from data_handling.import_letterboxd import process_letterboxd_import
from featureEngineering import feature_engineering
from modelTrain import train_personal_model

# --- Optimization: Cache TMDB API calls ---
# This makes the app lightweight and fast by not re-downloading movie data it has already seen.
requests_cache.install_cache('tmdb_cache', backend='sqlite', expire_after=86400) # Cache for 24 hours

# --- App Version ---
APP_VERSION = "3.2.6"
GITHUB_REPO = "mrsarthi/MBM_recommender"

# --- 1. Path & Config Setup ---

def get_path(relative_path):
    """ 
    Robust path finder:
    1. If running as a frozen .exe, looks in the temporary bundle (_MEIPASS).
    2. If running as a script, looks in the folder containing this script.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not frozen, use the folder where this app.py file is located
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    return os.path.join(base_path, relative_path)

# Load API Key
load_dotenv(dotenv_path=get_path('.env'))
key = os.getenv('TMDB_key')
baseUrl = "https://api.themoviedb.org/3"

# --- Gemini AI Setup ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
        print("✅ Gemini AI initialized.")
    except Exception as e:
        print(f"⚠️ Gemini init failed: {e}")
else:
    print("⚠️ GEMINI_API_KEY not set. Using fallback mood buttons.")

# File Paths — User data persists in %APPDATA% across exe updates
def get_user_data_path(filename):
    """Returns a path inside %APPDATA%/MBM_Recommender/ for persistent user data."""
    appdata = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'MBM_Recommender')
    os.makedirs(appdata, exist_ok=True)
    # Ensure subdirectories exist
    full_path = os.path.join(appdata, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def _get_exe_relative_path(filename):
    """Old path logic — used only for migration."""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

def _migrate_old_data():
    """One-time migration: copy user data from old exe-relative location to AppData."""
    migration_marker = get_user_data_path('.migrated')
    if os.path.exists(migration_marker):
        return  # Already migrated
    
    files_to_migrate = [
        'config.json',
        'app_memory_ids.csv',
        'tmdb_cache.sqlite',
        'user_data/personal_ai_model.pkl',
        'user_data/model_columns.pkl',
        'user_data/summary_vectorizer.pkl',
        'user_data/user_profile.csv',
        'user_data/user_profile_features.csv',
    ]
    
    migrated_any = False
    for f in files_to_migrate:
        old_path = _get_exe_relative_path(f)
        new_path = get_user_data_path(f)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(old_path, new_path)
                print(f"  Migrated: {f}")
                migrated_any = True
            except Exception as e:
                print(f"  Failed to migrate {f}: {e}")
    
    if migrated_any:
        print("✅ Data migration to AppData complete.")
    
    # Write marker so we don't re-migrate
    with open(migration_marker, 'w') as marker:
        marker.write('migrated')

# Run migration on startup
_migrate_old_data()

CONFIG_FILE = get_user_data_path('config.json')
APP_MEMORY_FILE = get_user_data_path('app_memory_ids.csv')

MODEL_PATH = get_user_data_path('user_data/personal_ai_model.pkl')
COLUMNS_PATH = get_user_data_path('user_data/model_columns.pkl')
VECTORIZER_PATH = get_user_data_path('user_data/summary_vectorizer.pkl')


# --- 2. Helper Functions ---

def titleNormalize(title):
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def watchedMovies(letterboxd_path, app_memory_path):
    """
    Loads watched movies. 
    Also identifies 'Hated Movies' (Rating <= 2.5) for the Veto System.
    """
    watchedSet_titles = set()
    watchedSet_ids = set()
    hated_movies = set()
    
    # 1. Load User/Friend CSV
    try:
        if letterboxd_path and os.path.exists(letterboxd_path):
            df = pd.read_csv(letterboxd_path)
            df.columns = [c.strip() for c in df.columns]
            
            for index, row in df.iterrows():
                col_name = 'Name' if 'Name' in df.columns else 'Title'
                if col_name in row:
                    title = titleNormalize(row[col_name])
                    watchedSet_titles.add(title)
                    
                    # VETO LOGIC
                    if 'Rating' in row and pd.notna(row['Rating']):
                        try:
                            if float(row['Rating']) <= 2.5:
                                hated_movies.add(title)
                        except: pass
            print(f"Loaded {len(watchedSet_titles)} movies and {len(hated_movies)} hated movies from CSV.")
    except Exception as e:
        print(f"Warning: Could not read watched file: {e}")

    # 2. Load App Memory
    try:
        if os.path.exists(app_memory_path) and os.path.getsize(app_memory_path) > 0:
            memLogged = pd.read_csv(app_memory_path)
            if 'movie_id' in memLogged.columns:
                memory_set_ids = set(memLogged['movie_id'].astype(int))
                watchedSet_ids.update(memory_set_ids)
        else:
            with open(app_memory_path, 'w', newline='', encoding='utf-8') as f:
                f.write('movie_id,title\n')
    except Exception as e:
        print(f"Warning: Could not read app memory: {e}")
    
    return watchedSet_titles, watchedSet_ids, hated_movies

def load_ai_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            columns = joblib.load(COLUMNS_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("✅ AI Model, Columns, and Vectorizer Loaded Successfully.")
            return model, columns, vectorizer
        else:
            print("⚠️ Model files not found. Using standard popularity sorting.")
            return None, None, None
    except Exception as e:
        print(f"⚠️ Error loading AI: {e}")
        return None, None, None

# Valid TMDB genre list for Gemini to pick from
VALID_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

def get_genres_from_ai(user_input):
    """
    Uses Gemini to interpret a user's mood/description and return matching TMDB genres.
    Falls back to a simple keyword match if Gemini is unavailable.
    """
    if gemini_model:
        try:
            prompt = (
                f"You are a movie recommendation assistant who deeply understands internet culture, "
                f"memes, sarcasm, Gen-Z slang, and film community jargon.\n\n"
                f"The user describes their mood or what they want to watch:\n\n"
                f"\"{user_input}\"\n\n"
                f"IMPORTANT CONTEXT for interpreting user input:\n"
                f"- 'absolute cinema' or 'peak cinema' = critically acclaimed masterpieces (Drama, History, War)\n"
                f"- 'brainrot' or 'turn my brain off' = mindless fun (Action, Comedy, Animation)\n"
                f"- 'cozy vibes' or 'comfort movie' = warm, feel-good (Family, Comedy, Romance, Animation)\n"
                f"- 'edgy' or 'messed up' = dark, disturbing (Thriller, Horror, Crime)\n"
                f"- 'crying in the club' or 'in my feels' = emotional, tearjerker (Drama, Romance)\n"
                f"- 'kino' = artsy, high-quality cinema (Drama, History, Mystery)\n"
                f"- 'based' = bold, unconventional picks (Crime, Thriller, War, Western)\n"
                f"- 'mid' = user is bored of average stuff, suggest niche or standout genres\n"
                f"- Understand sarcasm: 'nothing too scary' with a wink might still mean Thriller\n"
                f"- Understand vibe descriptions: 'rainy day', 'late night', '3am energy' etc.\n\n"
                f"Select the most relevant genres from this EXACT list:\n"
                f"{', '.join(VALID_GENRES)}\n\n"
                f"Rules:\n"
                f"- Return ONLY genre names from the list above, separated by commas.\n"
                f"- Choose 2-5 genres that best match the user's intent (not just literal words).\n"
                f"- Do NOT include any explanation, formatting, or extra text.\n"
                f"- Example output: Action, Thriller, Science Fiction"
            )
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            print(f"🤖 Gemini Response: {raw}")
            
            # Parse and validate genres
            parsed = [g.strip() for g in raw.split(',')]
            valid = [g for g in parsed if g in VALID_GENRES]
            
            if valid:
                print(f"✅ Matched Genres: {valid}")
                return valid
            else:
                print("⚠️ Gemini returned no valid genres. Using fallback.")
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}. Using fallback.")
    
    # --- Fallback: Simple keyword matching ---
    return _fallback_mood_match(user_input)

def _fallback_mood_match(user_input):
    """Basic keyword-to-genre mapping when Gemini is unavailable."""
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
        'lighthearted': ['Comedy', 'Family', 'Animation'],
        'intense': ['Action', 'Thriller', 'War'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'romantic': ['Romance', 'Drama', 'Comedy'],
        'suspenseful': ['Thriller', 'Horror', 'Mystery']
    }
    text = user_input.lower()
    matched_genres = set()
    for keyword, genres in fallback_map.items():
        if keyword in text:
            matched_genres.update(genres)
    
    if matched_genres:
        print(f"📌 Fallback matched: {list(matched_genres)}")
        return list(matched_genres)
    
    # Default if nothing matches
    print("📌 No keywords matched. Defaulting to popular genres.")
    return ['Action', 'Comedy', 'Drama', 'Thriller']

# --- 3. Core Logic (Prediction & Analysis) ---

def predict_score(model, model_columns, vectorizer, genres, context, overview):
    # 1. Initialize Input
    input_data = {col: 0 for col in model_columns}
    
    # 2. Set Context
    if f'context_{context}' in input_data:
        input_data[f'context_{context}'] = 1
        
    # 3. Set Genres
    for g in genres:
        if f'genre_{g}' in input_data:
            input_data[f'genre_{g}'] = 1
            
    # 4. Set Default Rating
    input_data['rating_encoded'] = 2 

    # 5. Process Summary
    if vectorizer and overview:
        try:
            overview_text = str(overview)
            tfidf_matrix = vectorizer.transform([overview_text])
            feature_names = [f"summary_{w}" for w in vectorizer.get_feature_names_out()]
            dense_vector = tfidf_matrix.toarray()[0]
            
            for name, value in zip(feature_names, dense_vector):
                if name in input_data:
                    input_data[name] = value
        except Exception as e:
            pass

    # 6. Predict
    df_input = pd.DataFrame([input_data])
    df_input = df_input[model_columns] 
    return model.predict(df_input)[0]

def analyze(watchedSet_titles, watchedSet_ids, hated_movies, desiredGenre, ai_model, ai_columns, ai_vectorizer, user_context):
    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }
    idToGenre = {v: k for k, v in genreDict.items()}

    if desiredGenre:
        targetGenreIds = []
        for name in desiredGenre:
            genreId = genreDict.get(name)
            if genreId: targetGenreIds.append(str(genreId))

        if not targetGenreIds: return []

        genreIdString = "|".join(targetGenreIds)
        print(f"Searching TMDB for genres: {genreIdString}")

        # Fetch candidates
        discoverUrl = f"{baseUrl}/discover/movie"
        discoverParams = {
            'api_key': key, 'with_genres': genreIdString,
            'vote_average.gte': 5.5, 'vote_count.gte': 100, 
            'sort_by': 'popularity.desc', 'language': 'en-US', 'page': 1
        }

        results = []
        for _ in range(2): 
            resp = requests.get(discoverUrl, params=discoverParams)
            if resp.status_code == 200:
                results.extend(resp.json().get('results', []))
                discoverParams['page'] += 1
            else: break

        finalPicks = []
        for movie in results:
            title_norm = titleNormalize(movie['title'])
            movie_id = movie['id']
            
            # Filter: Already Watched?
            if (title_norm not in watchedSet_titles) and (movie_id not in watchedSet_ids):
                
                # --- AI PREDICTION ---
                if ai_model:
                    genres = [idToGenre[g] for g in movie.get('genre_ids', []) if g in idToGenre]
                    overview = movie.get('overview', '')
                    score = predict_score(ai_model, ai_columns, ai_vectorizer, genres, user_context, overview)
                    
                    # --- VETO SYSTEM ---
                    is_vetoed = False
                    for hated in hated_movies:
                        if (hated in title_norm) or (title_norm in hated):
                            print(f"🚫 Vetoing '{movie['title']}' because user hated '{hated}'")
                            score -= 3.0 
                            is_vetoed = True
                            break
                    
                    movie['ai_score'] = score
                else:
                    movie['ai_score'] = 0
                
                finalPicks.append(movie)

        print(f"Found {len(finalPicks)} candidate movies (sorting deferred to UI).")

        return finalPicks
    return []


# --- 4. GUI Class ---

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
    def flush(self): pass

class App(ctk.CTk):
    COLORS = {
        'bg_main': '#121212',          # Deep Dark Background
        'bg_card': '#1E1E1E',          # Card Surface
        'bg_card_hover': '#2C2C2C',    # Slightly Lighter
        'accent': '#9B6BDB',           # Deeper Purple Accent
        'accent_hover': '#7E4FC4',
        'text_main': '#E0E0E0',
        'text_sub': '#B0B0B0',
        'btn_text': '#1A1A1A',         # Dark text for colored buttons
        'success': '#03DAC6',          # Teal
        'danger': '#CF6679',
        'score_high': '#03DAC6',
        'score_med': '#FFB74D',
        'score_low': '#CF6679'
    }

    def __init__(self, watched_path, initialWatchedSet_titles, initialWatchedSet_ids, initialHated):
        super().__init__()
        
        self.watched_path = watched_path 
        self.watchedSet_titles = initialWatchedSet_titles
        self.watchedSet_ids = initialWatchedSet_ids
        self.hated_movies = initialHated
        
        # Load AI
        self.ai_model, self.ai_columns, self.ai_vectorizer = load_ai_model()
        
        self.title("Mood Movie Recommender AI")
        self.geometry("1000x850")
        self.configure(fg_color=self.COLORS['bg_main'])

        self.gemini_available = gemini_model is not None
        self.current_results = {}
        self.current_search_results = {}
        self.poster_base_url = "https://image.tmdb.org/t/p/w200"
        self.new_logs_count = 0  # Track new logs for auto-retrain prompt
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1) # Let the main frame expand
        
        # Main Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(2, weight=1)

        # System Font
        self.main_font = ('Segoe UI', 12)
        self.header_font = ('Segoe UI', 14, 'bold')
        self.title_font = ('Segoe UI', 20, 'bold')

        # --- Check if Model Exists. If not, show Onboarding. ---
        if self.ai_model is None:
            self.show_onboarding()
        else:
            self.show_main_app()

    def show_onboarding(self):
        """Builds the Welcome / Import Screen"""
        self.onboard_frame = ctk.CTkFrame(self.main_container, fg_color=self.COLORS['bg_card'], corner_radius=15)
        self.onboard_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.6)
        
        ctk.CTkLabel(self.onboard_frame, text="Welcome to Personal AI Recommender", font=('Segoe UI', 24, 'bold'), text_color=self.COLORS['text_main']).pack(pady=(40, 10))
        ctk.CTkLabel(self.onboard_frame, text="To get started, please import your Letterboxd Data Export.", font=('Segoe UI', 14), text_color=self.COLORS['text_sub']).pack(pady=(0, 20))
        ctk.CTkLabel(self.onboard_frame, text="We will analyze your taste and build a custom Neural ML model specifically for you.", font=('Segoe UI', 12, 'italic'), text_color=self.COLORS['accent']).pack(pady=(0, 40))
        
        self.import_btn = ctk.CTkButton(self.onboard_frame, text="Import Letterboxd Data (.zip)", command=self._on_import_zip, 
                      height=50, width=250, font=('Segoe UI', 14, 'bold'), 
                      fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'], 
                      text_color=self.COLORS['btn_text'], corner_radius=25)
        self.import_btn.pack(pady=10)
        
        self.skip_import_btn = ctk.CTkButton(self.onboard_frame, text="Start Fresh (Opt Out)", command=self._on_skip_import, 
                      height=40, width=250, font=('Segoe UI', 12, 'bold'), 
                      fg_color="transparent", hover_color=self.COLORS['bg_card_hover'], border_width=1, border_color=self.COLORS['text_sub'], text_color=self.COLORS['text_sub'], corner_radius=20)
        self.skip_import_btn.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(self.onboard_frame, text="", font=('Segoe UI', 12), text_color=self.COLORS['success'])
        self.status_label.pack(pady=20)
        
        self.progress = ctk.CTkProgressBar(self.onboard_frame, width=300, progress_color=self.COLORS['accent'])
        self.progress.pack(pady=10)
        self.progress.set(0)
        self.progress.pack_forget() # Hide initially

    def show_main_app(self):
        """Builds the main application UI"""
        # Clear container in case we came from onboarding
        for w in self.main_container.winfo_children(): w.destroy()
        
        # --- Build UI ---
        self.setup_top_bar(row=0)
        self.setup_mood_area(row=1)
        self.setup_main_tabs(row=2)
        
        # Update Banner (row 3, hidden by default)
        self.update_banner = ctk.CTkFrame(self.main_container, fg_color='#2D1F4E', corner_radius=8, height=40)
        self.update_banner.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.update_banner.grid_remove()  # Hidden by default
        
        self.update_label = ctk.CTkLabel(self.update_banner, text="", font=('Segoe UI', 12), text_color=self.COLORS['text_main'])
        self.update_label.pack(side="left", padx=15, pady=8)
        
        self.update_btn = ctk.CTkButton(self.update_banner, text="Download Update", width=130, height=28,
                                         fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'],
                                         text_color=self.COLORS['btn_text'], font=('Segoe UI', 11, 'bold'),
                                         corner_radius=14, command=self._on_download_update)
        self.update_btn.pack(side="right", padx=15, pady=8)
        
        # Redirect Console AFTER all UI elements are built preventing Tkinter threaded crashes
        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)
        
        # Check for updates in background
        threading.Thread(target=self._check_for_updates, daemon=True).start()

    def setup_top_bar(self, row):
        """Top bar with File selection and Context"""
        container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        container.grid(row=row, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        # 1. File Section Card
        file_frame = ctk.CTkFrame(container, fg_color=self.COLORS['bg_card'], corner_radius=10)
        file_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        file_inner = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_inner.pack(padx=20, pady=15, anchor="w", fill="x")
        
        ctk.CTkLabel(file_inner, text="Watched History", font=self.header_font, text_color=self.COLORS['accent']).pack(anchor="w")
        
        self.file_path_var = tk.StringVar(value=os.path.basename(self.watched_path) if self.watched_path else "Not Selected")
        
        path_row = ctk.CTkFrame(file_inner, fg_color="transparent")
        path_row.pack(fill="x", pady=(5,0))
        
        ctk.CTkLabel(path_row, textvariable=self.file_path_var, font=('Segoe UI', 12), text_color=self.COLORS['text_sub']).pack(side="left")
        ctk.CTkButton(path_row, text="Change File", command=self._on_browse_click, width=80, height=28, 
                      fg_color=self.COLORS['bg_card_hover'], hover_color=self.COLORS['accent'], 
                      font=('Segoe UI', 11, 'bold')).pack(side="left", padx=10)

        # 2. Context Section Card
        ctx_frame = ctk.CTkFrame(container, fg_color=self.COLORS['bg_card'], corner_radius=10)
        ctx_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        ctx_inner = ctk.CTkFrame(ctx_frame, fg_color="transparent")
        ctx_inner.pack(padx=20, pady=15, anchor="w", fill="x")
        
        ctk.CTkLabel(ctx_inner, text="Context (Who are you with?)", font=self.header_font, text_color=self.COLORS['accent']).pack(anchor="w")
        
        self.context_var = ctk.StringVar(value="Alone")
        ctx_menu = ctk.CTkOptionMenu(ctx_inner, variable=self.context_var, 
                                     values=["Alone", "Friends", "Family", "Partner", "Other"], 
                                     width=200, height=30, fg_color=self.COLORS['bg_card_hover'], 
                                     button_color=self.COLORS['accent'], button_hover_color=self.COLORS['accent_hover'],
                                     dropdown_fg_color=self.COLORS['bg_card'], dropdown_text_color=self.COLORS['text_main'])
        ctx_menu.pack(pady=(5,0), anchor="w")

    def setup_mood_area(self, row):
        """Mood input area with free-form text and generate button"""
        container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        container.grid(row=row, column=0, sticky="ew", padx=20, pady=5)
        
        # Header Row
        head_row = ctk.CTkFrame(container, fg_color="transparent")
        head_row.pack(fill="x")
        
        header_text = "💬 Describe your mood" if self.gemini_available else "💬 What are you in the mood for?"
        ctk.CTkLabel(head_row, text=header_text, font=self.title_font, text_color=self.COLORS['text_main']).pack(side="left")
        
        # Gemini badge
        if self.gemini_available:
            ctk.CTkLabel(head_row, text="Powered by Gemini ✨", font=('Segoe UI', 11, 'italic'), 
                         text_color=self.COLORS['success']).pack(side="left", padx=15)
        
        # Generate Button (Right Aligned)
        self.generate_btn = ctk.CTkButton(head_row, text="✨ Generate Recommendations", command=self._on_analyze_click, 
                      height=40, font=('Segoe UI', 12, 'bold'), 
                      fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'],
                      text_color=self.COLORS['btn_text'], corner_radius=20)
        self.generate_btn.pack(side="right")

        # Mood Input Card
        input_card = ctk.CTkFrame(container, fg_color=self.COLORS['bg_card'], corner_radius=10)
        input_card.pack(fill='x', expand=True, pady=(10, 5))
        
        self.mood_input = ctk.CTkTextbox(input_card, height=70, wrap=tk.WORD, 
                                         font=('Segoe UI', 13), 
                                         fg_color=self.COLORS['bg_main'], 
                                         text_color=self.COLORS['text_main'],
                                         border_width=1, border_color=self.COLORS['bg_card_hover'],
                                         corner_radius=8)
        self.mood_input.pack(fill='x', padx=15, pady=12)
        self.mood_input.insert('1.0', 'e.g. "I feel nostalgic and want something sci-fi but not too dark"')
        
        # Placeholder behavior
        self._mood_has_placeholder = True
        self.mood_input.bind('<FocusIn>', self._on_mood_focus_in)
        self.mood_input.bind('<FocusOut>', self._on_mood_focus_out)
        self.mood_input.configure(text_color=self.COLORS['text_sub'])  # Placeholder style

        # Quick mood chips (optional shortcuts)
        chips_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        chips_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        quick_moods = ["😊 Happy", "😢 Sad", "😱 Scared", "🤔 Thoughtful", "🔥 Excited", "💕 Romantic", "🌌 Adventurous"]
        for mood in quick_moods:
            ctk.CTkButton(chips_frame, text=mood, width=90, height=28,
                          font=('Segoe UI', 10), corner_radius=14,
                          fg_color=self.COLORS['bg_card_hover'], hover_color=self.COLORS['accent'],
                          text_color=self.COLORS['text_sub'],
                          command=lambda m=mood: self._insert_mood_chip(m)
                          ).pack(side="left", padx=3)

    def _on_mood_focus_in(self, event):
        if self._mood_has_placeholder:
            self.mood_input.delete('1.0', tk.END)
            self.mood_input.configure(text_color=self.COLORS['text_main'])
            self._mood_has_placeholder = False

    def _on_mood_focus_out(self, event):
        if not self.mood_input.get('1.0', tk.END).strip():
            self.mood_input.insert('1.0', 'e.g. "I feel nostalgic and want something sci-fi but not too dark"')
            self.mood_input.configure(text_color=self.COLORS['text_sub'])
            self._mood_has_placeholder = True

    def _insert_mood_chip(self, mood_text):
        """Insert a quick mood chip into the text input"""
        # Remove emoji prefix for cleaner text
        clean = mood_text.split(' ', 1)[1] if ' ' in mood_text else mood_text
        if self._mood_has_placeholder:
            self.mood_input.delete('1.0', tk.END)
            self.mood_input.configure(text_color=self.COLORS['text_main'])
            self._mood_has_placeholder = False
        
        current = self.mood_input.get('1.0', tk.END).strip()
        if current:
            self.mood_input.insert(tk.END, f", {clean}")
        else:
            self.mood_input.insert('1.0', f"I'm feeling {clean.lower()}")

    def setup_main_tabs(self, row):
        """Tabs for Results, Log, and Console"""
        self.notebook = ctk.CTkTabview(self.main_container, fg_color=self.COLORS['bg_main'], 
                                       segmented_button_fg_color=self.COLORS['bg_card'],
                                       segmented_button_selected_color=self.COLORS['accent'],
                                       segmented_button_unselected_color=self.COLORS['bg_card'],
                                       segmented_button_selected_hover_color=self.COLORS['accent_hover'])
        
        self.notebook.grid(row=row, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        
        self.notebook.add('My Taste')
        self.notebook.add('Recommendations')
        self.notebook.add('Log a Movie')
        self.notebook.add('System Log')
        
        self.setup_my_taste_tab(self.notebook.tab('My Taste'))
        self.setup_results_tab(self.notebook.tab('Recommendations'))
        self.setup_log_tab(self.notebook.tab('Log a Movie'))
        self.setup_console_tab(self.notebook.tab('System Log'))

    def setup_my_taste_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # We need a frame to hold the plots
        chart_frame = ctk.CTkFrame(parent, fg_color="transparent")
        chart_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.columnconfigure(1, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        # Matplotlib Figure (Dark Theme)
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 5), facecolor=self.COLORS['bg_card'])
        
        try:
            # 1. Load User Data
            df = pd.DataFrame()
            if self.watched_path and os.path.exists(self.watched_path):
                df = pd.read_csv(self.watched_path)
            
            if df.empty or 'Rating' not in df.columns:
                ctk.CTkLabel(chart_frame, text="Not enough rating data yet. Log some movies!", 
                             font=self.header_font).pack(pady=50)
                return

            # Clean and process data mapping
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
            df = df.dropna(subset=['Rating'])
            
            # --- Subplot 1: Rating Distribution (Histogram) ---
            ax1 = fig.add_subplot(121, facecolor=self.COLORS['bg_main'])
            ax1.hist(df['Rating'], bins=np.arange(0.25, 5.5, 0.5), color=self.COLORS['accent'], edgecolor='black', alpha=0.8)
            ax1.set_title("Your Rating Distribution", fontsize=12, color=self.COLORS['text_main'], pad=15)
            ax1.set_xlabel("Stars", color=self.COLORS['text_sub'])
            ax1.set_ylabel("Count", color=self.COLORS['text_sub'])
            ax1.set_xticks(np.arange(0.5, 5.5, 0.5))
            ax1.tick_params(colors=self.COLORS['text_sub'])
            for spine in ax1.spines.values():
                spine.set_color('#333333')

            # --- Subplot 2: Top Genres (Bar Chart - if available) ---
            ax2 = fig.add_subplot(122, facecolor=self.COLORS['bg_main'])
            
            if 'genres' in df.columns and df['genres'].notna().any():
                # Process comma-separated genres
                genres_series = df['genres'].dropna().str.split(',').explode().str.strip()
                genre_counts = genres_series.value_counts().head(8) # Top 8
                
                # Horizontal Bar Chart
                y_pos = np.arange(len(genre_counts))
                bars = ax2.barh(y_pos, genre_counts.values, align='center', color=self.COLORS['success'], alpha=0.8)
                ax2.set_yticks(y_pos, labels=genre_counts.index)
                ax2.invert_yaxis()  # top genre at the top
                ax2.set_title("Your Top Genres", fontsize=12, color=self.COLORS['text_main'], pad=15)
                ax2.set_xlabel("Movies Watched", color=self.COLORS['text_sub'])
                ax2.tick_params(colors=self.COLORS['text_sub'])
            else:
                ax2.text(0.5, 0.5, 'Genre data not available in export.', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax2.transAxes, color=self.COLORS['text_sub'])
                ax2.axis('off')

            plt.tight_layout()
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
            
        except Exception as e:
            ctk.CTkLabel(chart_frame, text=f"Error generating charts: {e}", text_color=self.COLORS['danger']).pack(pady=50)

    def setup_results_tab(self, parent):
        parent.columnconfigure(0, weight=1) # List
        parent.columnconfigure(1, weight=1) # Preview
        parent.rowconfigure(1, weight=1)  # Results row gets weight
        
        # Sort Controls Row
        sort_bar = ctk.CTkFrame(parent, fg_color="transparent")
        sort_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
        ctk.CTkLabel(sort_bar, text="Sort by:", font=('Segoe UI', 12), 
                     text_color=self.COLORS['text_sub']).pack(side="left", padx=(5, 8))
        
        self.sort_var = ctk.StringVar(value="TMDB Score")
        self.sort_toggle = ctk.CTkSegmentedButton(
            sort_bar, values=["TMDB Score", "AI Prediction"],
            variable=self.sort_var,
            command=self._on_sort_change,
            font=('Segoe UI', 11, 'bold'),
            selected_color=self.COLORS['accent'],
            selected_hover_color=self.COLORS['accent_hover'],
            unselected_color=self.COLORS['bg_card'],
            unselected_hover_color=self.COLORS['bg_card_hover'],
            text_color=[self.COLORS['btn_text'], self.COLORS['text_main']],
            corner_radius=8
        )
        self.sort_toggle.pack(side="left", padx=5)
        
        # Left: List
        self.results_scroll = ctk.CTkScrollableFrame(parent, fg_color=self.COLORS['bg_card'], label_text="Your Top Picks", 
                                                     label_font=self.header_font)
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=(0,10), pady=10)
        
        # Right: Details
        self.res_preview = ctk.CTkFrame(parent, fg_color=self.COLORS['bg_card'], corner_radius=10)
        self.res_preview.grid(row=1, column=1, sticky="nsew", pady=10)
        self.res_preview.columnconfigure(0, weight=1)
        self.res_preview.rowconfigure(2, weight=1)

        # Details Content
        self.res_poster = ctk.CTkLabel(self.res_preview, text="", width=220, height=330, fg_color=self.COLORS['bg_main'], corner_radius=8)
        self.res_poster.grid(row=0, column=0, pady=20)
        
        self.res_score = ctk.CTkLabel(self.res_preview, text="Select a movie...", font=('Segoe UI', 18, 'bold'), text_color=self.COLORS['text_sub'])
        self.res_score.grid(row=1, column=0, pady=(0, 10))
        
        self.res_text = ctk.CTkTextbox(self.res_preview, wrap=tk.WORD, height=10, state='disabled', 
                                       fg_color="transparent", font=('Segoe UI', 11), text_color=self.COLORS['text_main'])
        self.res_text.grid(row=2, column=0, sticky="nsew", padx=20, pady=5)
        
        btn_frame = ctk.CTkFrame(self.res_preview, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=20)
        
        ctk.CTkButton(btn_frame, text="Log Movie", command=lambda: self._on_log_movie("res"), 
                      fg_color=self.COLORS['success'], hover_color='#02c4b3', 
                      text_color=self.COLORS['btn_text'], width=120).pack(side="left", padx=10)
        
        ctk.CTkButton(btn_frame, text="View on TMDB", command=self._on_view_details, 
                      fg_color=self.COLORS['bg_card_hover'], hover_color="#3A3A3A", width=120).pack(side="left", padx=10)

    def setup_log_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Search Bar
        search_bar = ctk.CTkFrame(parent, fg_color="transparent")
        search_bar.grid(row=0, column=0, sticky="ew", pady=10)
        
        self.search_entry = ctk.CTkEntry(search_bar, placeholder_text="Enter movie title...", width=300, 
                                         fg_color=self.COLORS['bg_card'], border_color=self.COLORS['bg_card_hover'])
        self.search_entry.pack(side="left", padx=10)
        
        ctk.CTkButton(search_bar, text="Search", command=self._on_tmdb_search, 
                      fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'],
                      text_color=self.COLORS['btn_text']).pack(side="left")

        # Split View
        log_split = ctk.CTkFrame(parent, fg_color="transparent")
        log_split.grid(row=1, column=0, sticky="nsew")
        log_split.columnconfigure(0, weight=1)
        log_split.columnconfigure(1, weight=1)
        log_split.rowconfigure(0, weight=1)
        
        self.search_scroll = ctk.CTkScrollableFrame(log_split, fg_color=self.COLORS['bg_card'], label_text="Search Results")
        self.search_scroll.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        
        # Details
        log_preview = ctk.CTkFrame(log_split, fg_color=self.COLORS['bg_card'])
        log_preview.grid(row=0, column=1, sticky="nsew")
        log_preview.columnconfigure(0, weight=1)
        log_preview.rowconfigure(1, weight=1)
        
        self.log_poster = ctk.CTkLabel(log_preview, text="", width=150, height=225, fg_color=self.COLORS['bg_main'], corner_radius=8)
        self.log_poster.grid(row=0, column=0, pady=20)
        
        self.log_text = ctk.CTkTextbox(log_preview, wrap=tk.WORD, height=10, state='disabled', 
                                       fg_color="transparent", font=('Segoe UI', 11))
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=20, pady=5)
        
        ctk.CTkButton(log_preview, text="Log & Rate Movie", command=lambda: self._on_log_movie("log"), 
                      fg_color=self.COLORS['success'], hover_color='#02c4b3',
                      text_color=self.COLORS['btn_text']).grid(row=2, column=0, pady=20)

    def setup_console_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1) # The textbox gets the weight
        
        # Action Bar
        action_bar = ctk.CTkFrame(parent, fg_color="transparent")
        action_bar.grid(row=0, column=0, sticky="ew", pady=(5, 10))
        
        ctk.CTkLabel(action_bar, text="System Log & AI Controls", font=self.header_font, text_color=self.COLORS['accent']).pack(side="left", padx=5)
        
        self.retrain_btn = ctk.CTkButton(action_bar, text="⚡ Retrain AI Model", command=self._on_retrain_ai, 
                                         fg_color=self.COLORS['bg_card_hover'], hover_color=self.COLORS['accent'], 
                                         height=30, width=150)
        self.retrain_btn.pack(side="right", padx=5)

        self.console_output = ctk.CTkTextbox(parent, wrap=tk.WORD, font=('Consolas', 10), state='disabled', 
                                             fg_color=self.COLORS['bg_main'], text_color="#00FF00")
        self.console_output.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # --- Event Handlers & Logic ---

    def _on_skip_import(self):
        user_csv_path = get_user_data_path('user_data/user_profile.csv')
        os.makedirs(os.path.dirname(user_csv_path), exist_ok=True)
        if not os.path.exists(user_csv_path):
            with open(user_csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write('Name,Year,Rating\n')
        
        self.watched_path = user_csv_path
        self._save_config(user_csv_path)
        self.watchedSet_titles, self.watchedSet_ids, self.hated_movies = watchedMovies(user_csv_path, APP_MEMORY_FILE)
        
        self.show_main_app()

    def _on_import_zip(self):
        zip_path = filedialog.askopenfilename(title="Select your Letterboxd Export Zip", filetypes=(("Zip Files", "*.zip"),))
        if not zip_path: return
        
        self.import_btn.configure(state="disabled", text="Processing...")
        self.progress.pack(pady=10)
        self.progress.set(0.1)
        self.status_label.configure(text="Extracting & Fetching TMDB Data... (This takes a few minutes)")
        
        # Run ML Pipeline in background thread to keep UI responsive
        threading.Thread(target=self._run_ml_pipeline, args=(zip_path,), daemon=True).start()

    def _run_ml_pipeline(self, zip_path):
        user_csv_path = get_user_data_path('user_data/user_profile.csv')
        features_path = get_user_data_path('user_data/user_profile_features.csv')
        
        # Define the callback that hydrates the UI's progress bar (starts at 10% visually, scales to 60%)
        def tmdb_progress(current, total):
            if total > 0:
                percent = current / total
                # Map 0-100% TMDB fetch to 0.1->0.6 on the UI progress bar
                ui_progress = 0.1 + (0.5 * percent)
                self._update_onboard_status(f"Fetching TMDB Data: {current}/{total} movies...", progress=ui_progress)

        # 1. Import
        success = process_letterboxd_import(zip_path, output_csv_path=user_csv_path, progress_callback=tmdb_progress)
        if not success:
            self._update_onboard_status("Failed to import Zip. Check TMDB API key.", error=True)
            return
            
        self._update_onboard_status("Data Hydrated! Engineering NLP Features...", progress=0.7)
        
        # 2. Feature Engineering
        success = feature_engineering(input_file=user_csv_path, output_file=features_path, vectorizer_path=VECTORIZER_PATH)
        if not success:
            self._update_onboard_status("Failed to engineer features.", error=True)
            return
            
        self._update_onboard_status("Features Created. Training Neural Pathways...", progress=0.85)
        
        # 3. Train Model
        success = train_personal_model(input_file=features_path, model_path=MODEL_PATH, columns_path=COLUMNS_PATH)
        if not success:
            self._update_onboard_status("Failed to train model. Need at least 15 ratings.", error=True)
            return
            
        self._update_onboard_status("AI Training Complete! Booting...", progress=1.0)
        
        # Reload Models and launch app
        self.ai_model, self.ai_columns, self.ai_vectorizer = load_ai_model()
        self.watched_path = user_csv_path
        self._save_config(user_csv_path)
        self.watchedSet_titles, self.watchedSet_ids, self.hated_movies = watchedMovies(user_csv_path, APP_MEMORY_FILE)
        
        # Back to main thread for UI changes
        self.after(1500, self.show_main_app)

    def _update_onboard_status(self, text, progress=None, error=False):
        """Thread-safe UI updater"""
        def update():
            self.status_label.configure(text=text, text_color=self.COLORS['danger'] if error else self.COLORS['success'])
            if progress is not None:
                self.progress.set(progress)
            if error:
                self.import_btn.configure(state="normal", text="Try Again")
                self.progress.pack_forget()
            self.update_idletasks()
        self.after(0, update)

    def _save_config(self, path):
        with open(CONFIG_FILE, 'w') as f: json.dump({'watched_path': path}, f)

    def _check_for_updates(self):
        """Background thread: check GitHub releases for a newer version."""
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                latest_tag = data.get('tag_name', '').lstrip('v')
                if latest_tag and latest_tag != APP_VERSION:
                    # Compare version tuples
                    try:
                        latest_parts = tuple(int(x) for x in latest_tag.split('.'))
                        current_parts = tuple(int(x) for x in APP_VERSION.split('.'))
                        if latest_parts > current_parts:
                            download_url = data.get('html_url', f'https://github.com/{GITHUB_REPO}/releases/latest')
                            self._latest_release_url = download_url
                            self.after(0, lambda: self._show_update_banner(latest_tag))
                    except ValueError:
                        pass  # Malformed version tag, skip
        except Exception:
            pass  # Fail silently — user is offline or rate limited

    def _show_update_banner(self, version):
        """Show the update banner on the main thread."""
        self.update_label.configure(text=f"🚀 Update v{version} available!")
        self.update_banner.grid()  # Show banner

    def _on_download_update(self):
        """Open the GitHub releases page."""
        url = getattr(self, '_latest_release_url', f'https://github.com/{GITHUB_REPO}/releases/latest')
        webbrowser.open_new_tab(url)

    def _on_browse_click(self):
        path = filedialog.askopenfilename(title="Select 'watched.csv'", filetypes=(("CSV", "*.csv"),))
        if path:
            self.file_path_var.set(os.path.basename(path))
            self.watched_path = path
            self._save_config(path)
            self.watchedSet_titles, self.watchedSet_ids, self.hated_movies = watchedMovies(path, APP_MEMORY_FILE)

    def _on_analyze_click(self):
        try:
            self.console_output.configure(state='normal')
            self.console_output.delete('1.0', tk.END)
            for w in self.results_scroll.winfo_children(): w.destroy()
            self.current_results.clear()
            self._clear_preview(self.res_poster, self.res_text, self.res_score)
            
            # Get mood text from the input box
            mood_text = self.mood_input.get('1.0', tk.END).strip()
            ctx = self.context_var.get()
            
            if not mood_text or self._mood_has_placeholder:
                print("Error: Please describe your mood or what you'd like to watch.")
                return

            print(f"🧠 Understanding mood: \"{mood_text}\" | Context: {ctx.upper()}...")
            
            # Disable the button while processing
            self.generate_btn.configure(state="disabled", text="🔄 Thinking...")
            self.update_idletasks()
            
            # Run Gemini call in background to keep UI responsive
            threading.Thread(target=self._run_gemini_analysis, args=(mood_text, ctx), daemon=True).start()
            
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            self.generate_btn.configure(state="normal", text="✨ Generate Recommendations")
    
    def _run_gemini_analysis(self, mood_text, ctx):
        """Background thread: get genres from Gemini, then fetch TMDB results."""
        try:
            genres = get_genres_from_ai(mood_text)
            
            if not genres:
                print("No genres could be determined. Try rephrasing.")
                self.after(0, lambda: self.generate_btn.configure(state="normal", text="✨ Generate Recommendations"))
                return
            
            print(f"🎬 Searching TMDB for: {', '.join(genres)}")
            
            picks = analyze(
                self.watchedSet_titles, 
                self.watchedSet_ids, 
                self.hated_movies,
                genres, 
                self.ai_model, 
                self.ai_columns, 
                self.ai_vectorizer, 
                ctx
            )
            
            # Schedule UI updates on the main thread
            self.after(0, lambda: self._display_results(picks))
            
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="✨ Generate Recommendations"))
    
    def _on_sort_change(self, value):
        """Re-sort and redisplay results when sort toggle changes."""
        if hasattr(self, '_last_picks') and self._last_picks:
            self._display_results(self._last_picks)

    def _display_results(self, picks):
        """Populate results UI on the main thread."""
        self.generate_btn.configure(state="normal", text="✨ Generate Recommendations")
        
        # Store picks for re-sorting
        self._last_picks = picks
        
        # Clear existing results
        for w in self.results_scroll.winfo_children(): w.destroy()
        self.current_results.clear()
        
        if picks:
            # Sort based on user's toggle selection
            sort_mode = self.sort_var.get()
            if sort_mode == "AI Prediction":
                sorted_picks = sorted(picks, key=lambda x: x.get('ai_score', 0), reverse=True)
                print(f"\nSorted {len(picks)} recommendations by AI Prediction.")
            else:
                sorted_picks = sorted(picks, key=lambda x: x.get('vote_average', 0), reverse=True)
                print(f"\nSorted {len(picks)} recommendations by TMDB Score.")
            
            for m in sorted_picks[:30]:
                year = m['release_date'].split('-')[0] if m.get('release_date') else "N/A"
                score = m.get('ai_score', 0)
                tmdb_avg = m.get('vote_average', 0)
                
                text_label = f"{m['title']} ({year})"
                
                # Color coding logic
                btn_color = self.COLORS['bg_card_hover']
                text_color = self.COLORS['text_main']
                
                # Build score badges
                badges = []
                if score > 4.2: 
                    badges.append(f"★ {score:.1f}")
                    text_color = self.COLORS['score_high']
                elif score < 3.0: 
                    text_color = self.COLORS['score_low']
                elif score > 0:
                    badges.append(f"★ {score:.1f}")
                
                if tmdb_avg > 0:
                    badges.append(f"TMDB {tmdb_avg:.1f}")
                
                if badges:
                    text_label += f"  {'  |  '.join(badges)}"

                btn = ctk.CTkButton(self.results_scroll, text=text_label, anchor="w", 
                                    fg_color=btn_color, hover_color="#3A3A3A",
                                    text_color=text_color,
                                    font=('Segoe UI', 12),
                                    command=lambda x=m: self._on_result_click(x, "res"),
                                    height=35)
                btn.pack(fill='x', padx=5, pady=3)
                self.current_results[btn] = m 
        else:
            print("No results found. Try describing your mood differently.")
        
        self.notebook.set('Recommendations')
        self.console_output.configure(state='disabled')

    def _on_tmdb_search(self):
        q = self.search_entry.get()
        if not q: return
        for w in self.search_scroll.winfo_children(): w.destroy()
        self.current_search_results.clear()
        self._clear_preview(self.log_poster, self.log_text, None)
        
        try:
            print(f"Searching: {q}...")
            res = requests.get(f"{baseUrl}/search/movie", params={'api_key':key,'query':q}).json().get('results',[])
            for m in res[:15]:
                if m['id'] not in self.watchedSet_ids:
                    year = m['release_date'].split('-')[0] if m.get('release_date') else "N/A"
                    btn = ctk.CTkButton(self.search_scroll, text=f"{m['title']} ({year})", anchor="w", 
                                        fg_color=self.COLORS['bg_card_hover'], 
                                        command=lambda x=m: self._on_result_click(x, "log"))
                    btn.pack(fill='x', padx=5, pady=2)
                    self.current_search_results[btn] = m
        except Exception as e: print(e)

    def _on_result_click(self, movie, mode):
        target_btn = None
        source_dict = self.current_results if mode == "res" else self.current_search_results
        
        for btn, m in source_dict.items():
            if m == movie:
                target_btn = btn
                break
        
        if mode == "res":
            self.selected_result_btn = target_btn
            poster, text_w, score_w = self.res_poster, self.res_text, self.res_score
            score = movie.get('ai_score', 0)
            tmdb_avg = movie.get('vote_average', 0)
            if score_w: 
                score_text = f"AI Match: {score:.1f} / 5.0"
                if tmdb_avg > 0:
                    score_text += f"  •  TMDB: {tmdb_avg:.1f} / 10"
                score_w.configure(text=score_text)
                if score >= 4.0: score_w.configure(text_color=self.COLORS['score_high'])
                elif score <= 2.5: score_w.configure(text_color=self.COLORS['score_low'])
                else: score_w.configure(text_color=self.COLORS['score_med'])
        else:
            self.selected_search_btn = target_btn
            poster, text_w, score_w = self.log_poster, self.log_text, None
            
        self._update_text(text_w, movie.get('overview', ''))
        if movie.get('poster_path'): self._load_img(movie['poster_path'], poster)
        else: poster.configure(image=None, text="No Image")

    def _load_img(self, path, label):
        try:
            d = requests.get(f"{self.poster_base_url}{path}").content
            img = Image.open(io.BytesIO(d))
            # Ratio preserve
            w, h = img.size
            target_w = label.cget("width")
            ratio = target_w / w
            target_h = int(h * ratio)
            
            img.thumbnail((target_w, target_h))
            c_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            label.configure(image=c_img, text="")
            label.image = c_img 
        except: label.configure(image=None, text="Error")

    def _update_text(self, w, t):
        w.configure(state='normal')
        w.delete('1.0', tk.END)
        w.insert(tk.END, t)
        w.configure(state='disabled')

    def _clear_preview(self, l, t, score_label=None):
        l.configure(image=None, text="")
        self._update_text(t, "")
        if score_label: score_label.configure(text="Select a movie...")

    def _on_log_movie(self, mode):
        # 1. Determine selected movie based on tab
        if mode == "res":
            if not getattr(self, 'selected_result_btn', None): return
            m = self.current_results.get(self.selected_result_btn)
            btn_ref = self.selected_result_btn
        else:
            if not getattr(self, 'selected_search_btn', None): return
            m = self.current_search_results.get(self.selected_search_btn)
            btn_ref = self.selected_search_btn
            
        if not m: return

        # 2. Build Custom Rating Dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Log: {m['title']}")
        dialog.geometry("400x300")
        dialog.configure(fg_color=self.COLORS['bg_card'])
        dialog.attributes('-topmost', True)
        dialog.grab_set()  # Focus exclusively on this window
        
        # Center Dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(dialog, text=f"How many stars for", font=('Segoe UI', 14), text_color=self.COLORS['text_sub']).pack(pady=(20, 5))
        ctk.CTkLabel(dialog, text=m['title'], font=('Segoe UI', 18, 'bold'), text_color=self.COLORS['accent']).pack()
        
        # Rating Slider
        rating_var = tk.DoubleVar(value=3.0)
        
        def update_label(value):
            v_val = float(value)
            rating_label.configure(text=f"Rating: {v_val:.1f} ★")

        slider = ctk.CTkSlider(dialog, from_=0.5, to=5.0, number_of_steps=9, variable=rating_var, command=update_label,
                               button_color=self.COLORS['accent'], button_hover_color=self.COLORS['accent_hover'],
                               progress_color=self.COLORS['accent'])
        slider.pack(pady=20, padx=40, fill="x")
        
        rating_label = ctk.CTkLabel(dialog, text="Rating: 3.0 ★", font=('Segoe UI', 16, 'bold'), text_color=self.COLORS['score_med'])
        rating_label.pack()

        def submit_log():
            final_rating = rating_var.get()
            self._process_movie_log(m, final_rating, btn_ref, mode)
            dialog.destroy()

        ctk.CTkButton(dialog, text="Save Log", command=submit_log, fg_color=self.COLORS['success'], hover_color='#02c4b3', text_color=self.COLORS['btn_text']).pack(pady=20)

    def _process_movie_log(self, m, rating, btn_ref, mode):
        # Prevent double logging
        if self.watchedSet_ids and m['id'] in self.watchedSet_ids: return
        
        # 1. Update in-memory sets instantly
        title_norm = titleNormalize(m['title'])
        if self.watchedSet_ids is None: self.watchedSet_ids = set()
        
        self.watchedSet_ids.add(m['id'])
        self.watchedSet_titles.add(title_norm)
        
        # Immediate Veto for low ratings!
        if rating <= 2.5:
            self.hated_movies.add(title_norm)
            print(f"Added {m['title']} to Veto List.")
            
        print(f"Logged '{m['title']}' with {rating} stars.")

        # 2. Append to general memory (so it won't be recommended again)
        try:
            with open(APP_MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{m["id"]},"{m["title"]}"\n')
        except Exception as e:
            print(f"Failed to save to memory file: {e}")

        # 3. Append to Active Watched History for Machine Learning Models
        year = m.get('release_date', 'N/A').split('-')[0]
        try:
            target_path = self.watched_path if self.watched_path else get_user_data_path('user_data/user_profile.csv')
            
            # Ensure directory exists just in case
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if os.path.exists(target_path):
                # We need to maintain the format used by the personal ML pipeline: 
                # Name/Title, Year, Rating (others can be blank or TMDB API fetching happens later)
                with open(target_path, 'a', newline='', encoding='utf-8') as f:
                    f.write(f'"{m["title"]}",{year},{rating}\n')
                print(f"Saved to user dataset: {target_path}")
            else:
                # Create a barebones one if none exists
                with open(target_path, 'w', newline='', encoding='utf-8') as f:
                    f.write('Name,Year,Rating\n')
                    f.write(f'"{m["title"]}",{year},{rating}\n')
                print(f"Created and saved to new dataset: {target_path}")
        except Exception as e:
            print(f"Failed to save to watched history CSV: {e}")

        # 4. Clean up UI
        btn_ref.destroy()
        if mode == "res":
            self._clear_preview(self.res_poster, self.res_text, self.res_score)
        else:
            self._clear_preview(self.log_poster, self.log_text)
            
        messagebox.showinfo("Logged", f"Saved {rating} ★ for '{m['title']}'")
        
        # 5. Continuous Learning Check
        self.new_logs_count += 1
        if self.new_logs_count >= 5:
            if messagebox.askyesno("Retrain AI?", "You've logged 5 new movies! Would you like to retrain your Personal AI to learn from these new ratings?"):
                self.new_logs_count = 0
                self.notebook.set('System Log')
                self._on_retrain_ai()

    def _on_view_details(self):
        if not getattr(self, 'selected_result_btn', None): return
        m = self.current_results.get(self.selected_result_btn)
        if m: webbrowser.open_new_tab(f"https://www.themoviedb.org/movie/{m['id']}")

    def _on_retrain_ai(self):
        if not self.watched_path or not os.path.exists(self.watched_path):
            messagebox.showwarning("No Data", "No watched data available to train on.")
            return
            
        self.retrain_btn.configure(state="disabled", text="Training...")
        self.console_output.configure(state='normal')
        self.console_output.insert(tk.END, "\n--- Initiating Personal AI Retraining Sequence ---\n")
        self.console_output.see(tk.END)
        self.console_output.configure(state='disabled')
        
        threading.Thread(target=self._run_retraining_thread, daemon=True).start()
        
    def _run_retraining_thread(self):
        features_path = get_user_data_path('user_data/user_profile_features.csv')
        
        # 1. Feature Engineer
        print("Extracting NLP Features & Updating Matrix...")
        success = feature_engineering(input_file=self.watched_path, output_file=features_path, vectorizer_path=VECTORIZER_PATH)
        
        if success:
            # 2. Train Model
            print("Training Neural Decision Trees...")
            train_success = train_personal_model(input_file=features_path, model_path=MODEL_PATH, columns_path=COLUMNS_PATH)
            
            if train_success:
                print("✅ Retraining Complete! Reloading Neural Pathways...")
                # 3. Reload into app memory safely
                def reload():
                    self.ai_model, self.ai_columns, self.ai_vectorizer = load_ai_model()
                    self.retrain_btn.configure(state="normal", text="⚡ Retrain AI Model")
                    messagebox.showinfo("Success", "AI successfully retrained on your latest taste profile!")
                self.after(0, reload)
                return
                
        # Handle Failure
        def fail():
            print("❌ Retraining failed. Check logs.")
            self.retrain_btn.configure(state="normal", text="⚡ Retrain AI Model")
            messagebox.showerror("Error", "Failed to retrain model. You may need more ratings.")
        self.after(0, fail)

    # Deprecated in favor of _process_movie_log

# --- 5. Main Execution ---

if __name__ == "__main__":
    if key is None:
        ctk.set_appearance_mode("dark")
        messagebox.showerror("Error", "TMDB_key missing in .env")
        sys.exit()

    ctk.set_appearance_mode("dark")
    w_path = None
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: w_path = json.load(f).get('watched_path')
        except: pass
    
    if not w_path or not os.path.exists(w_path):
        w_path = None

    t, i, h = watchedMovies(w_path, APP_MEMORY_FILE)
    
    app = App(w_path, t, i, h)
    app.mainloop()