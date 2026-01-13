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
from PIL import Image, ImageTk
import joblib
import numpy as np

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

# File Paths
def get_user_data_path(filename):
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

CONFIG_FILE = get_user_data_path('config.json')
APP_MEMORY_FILE = get_user_data_path('app_memory_ids.csv')

MODEL_PATH = get_path('models/personal_ai_model.pkl')
COLUMNS_PATH = get_path('models/model_columns.pkl')
VECTORIZER_PATH = get_path('models/summary_vectorizer.pkl')


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

def askForMood():
    return {
        'happy': ['Comedy', 'Music', 'Animation', 'Family', 'Romance'],
        'sad': ['Drama', 'Romance'],
        'tense': ['Horror', 'Thriller', 'Mystery', 'Crime'],
        'adventurous': ['Adventure', 'Science Fiction', 'Fantasy', 'Action'],
        'calm': ['Documentary', 'Drama', 'History'],
        'nostalgic': ['Drama', 'Romance', 'Fantasy'],
        'excited': ['Action', 'Adventure', 'Comedy'],
        'thoughtful': ['Drama', 'Documentary', 'Biography'],
        'scary': ['Horror', 'Thriller'],
        'lighthearted': ['Comedy', 'Family', 'Animation'],
        'intense': ['Action', 'Thriller', 'War'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'uplifting': ['Comedy', 'Family', 'Musical'],
        'romantic': ['Romance', 'Drama', 'Comedy'],
        'suspenseful': ['Thriller', 'Horror', 'Mystery']
    }

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

        # Sort by AI Score
        if ai_model:
            finalPicks.sort(key=lambda x: x['ai_score'], reverse=True)
            print(f"Sorted {len(finalPicks)} movies by AI preference.")

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
    # --- Theme Configuration ---
    COLORS = {
        'bg_main': '#121212',          # Deep Dark Background
        'bg_card': '#1E1E1E',          # Card Surface
        'bg_card_hover': '#2C2C2C',    # Slightly Lighter
        'accent': '#BB86FC',           # Purple Accent
        'accent_hover': '#9965f4',
        'text_main': '#E0E0E0',
        'text_sub': '#B0B0B0',
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

        self.mood_map = askForMood()
        self.current_results = {}
        self.current_search_results = {}
        self.poster_base_url = "https://image.tmdb.org/t/p/w200"
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # Main content area expands

        # System Font
        self.main_font = ('Segoe UI', 12)
        self.header_font = ('Segoe UI', 14, 'bold')
        self.title_font = ('Segoe UI', 20, 'bold')

        # --- Build UI ---
        self.setup_top_bar(row=0)
        self.setup_mood_area(row=1)
        self.setup_main_tabs(row=2)
        
        # Redirect Console
        # Note: We do this after setup_main_tabs so self.console_output exists
        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)

    def setup_top_bar(self, row):
        """Top bar with File selection and Context"""
        container = ctk.CTkFrame(self, fg_color=self.COLORS['bg_card'], corner_radius=10)
        container.grid(row=row, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        # 1. File Section
        file_frame = ctk.CTkFrame(container, fg_color="transparent")
        file_frame.pack(side="left", padx=15, pady=10)
        
        ctk.CTkLabel(file_frame, text="Watched History", font=self.header_font, text_color=self.COLORS['accent']).pack(anchor="w")
        
        self.file_path_var = tk.StringVar(value=os.path.basename(self.watched_path) if self.watched_path else "Not Selected")
        
        path_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        path_row.pack(fill="x", pady=(5,0))
        
        ctk.CTkLabel(path_row, textvariable=self.file_path_var, font=('Segoe UI', 11), text_color=self.COLORS['text_sub']).pack(side="left")
        ctk.CTkButton(path_row, text="Change", command=self._on_browse_click, width=60, height=24, 
                      fg_color=self.COLORS['bg_card_hover'], hover_color=self.COLORS['accent'], 
                      font=('Segoe UI', 10)).pack(side="left", padx=10)

        # Divider
        ctk.CTkFrame(container, width=2, height=40, fg_color=self.COLORS['bg_main']).pack(side="left", padx=10, pady=10)

        # 2. Context Section
        ctx_frame = ctk.CTkFrame(container, fg_color="transparent")
        ctx_frame.pack(side="left", padx=15, pady=10)
        
        ctk.CTkLabel(ctx_frame, text="Company", font=self.header_font, text_color=self.COLORS['accent']).pack(anchor="w")
        
        self.context_var = ctk.StringVar(value="Alone")
        ctx_menu = ctk.CTkOptionMenu(ctx_frame, variable=self.context_var, 
                                     values=["Alone", "Friends", "Family", "Partner", "Other"], 
                                     width=140, fg_color=self.COLORS['bg_card_hover'], 
                                     button_color=self.COLORS['accent'], button_hover_color=self.COLORS['accent_hover'])
        ctx_menu.pack(pady=(5,0))

    def setup_mood_area(self, row):
        """Mood selection and Run Button"""
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=row, column=0, sticky="ew", padx=20, pady=5)
        
        # Header
        head_row = ctk.CTkFrame(container, fg_color="transparent")
        head_row.pack(fill="x")
        ctk.CTkLabel(head_row, text="How are you feeling?", font=self.title_font, text_color=self.COLORS['text_main']).pack(side="left")
        
        # Run Button (Right Aligned)
        ctk.CTkButton(head_row, text="✨ Generate Recommendations", command=self._on_analyze_click, 
                      height=40, font=('Segoe UI', 12, 'bold'), 
                      fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'],
                      corner_radius=20).pack(side="right")

        # Mood Scroller
        self.mood_frame_scrollable = ctk.CTkScrollableFrame(container, height=60, orientation="horizontal", 
                                                          fg_color="transparent", scrollbar_button_color=self.COLORS['bg_card'])
        self.mood_frame_scrollable.pack(fill='x', expand=True, pady=(10, 5))
        
        self.selected_mood_var = tk.StringVar(value="happy") # Default
        
        # Create Mood Chips
        for mood in self.mood_map.keys():
            # Using Radio Button styled to look minimal
            rb = ctk.CTkRadioButton(self.mood_frame_scrollable, text=mood.title(), variable=self.selected_mood_var, value=mood,
                                    font=('Segoe UI', 12), text_color=self.COLORS['text_main'],
                                    fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover'])
            rb.pack(side="left", padx=15)

    def setup_main_tabs(self, row):
        """Tabs for Results, Log, and Console"""
        self.notebook = ctk.CTkTabview(self, fg_color=self.COLORS['bg_main'], 
                                       segmented_button_fg_color=self.COLORS['bg_card'],
                                       segmented_button_selected_color=self.COLORS['accent'],
                                       segmented_button_unselected_color=self.COLORS['bg_card'],
                                       segmented_button_selected_hover_color=self.COLORS['accent_hover'])
        
        self.notebook.grid(row=row, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        self.notebook.add('Recommendations')
        self.notebook.add('Library Search')
        self.notebook.add('System Log')
        
        self.setup_results_tab(self.notebook.tab('Recommendations'))
        self.setup_log_tab(self.notebook.tab('Library Search'))
        self.setup_console_tab(self.notebook.tab('System Log'))

    def setup_results_tab(self, parent):
        parent.columnconfigure(0, weight=1) # List
        parent.columnconfigure(1, weight=1) # Preview
        parent.rowconfigure(0, weight=1)
        
        # Left: List
        self.results_scroll = ctk.CTkScrollableFrame(parent, fg_color=self.COLORS['bg_card'], label_text="Your Top Picks", 
                                                     label_font=self.header_font)
        self.results_scroll.grid(row=0, column=0, sticky="nsew", padx=(0,10), pady=10)
        
        # Right: Details
        self.res_preview = ctk.CTkFrame(parent, fg_color=self.COLORS['bg_card'], corner_radius=10)
        self.res_preview.grid(row=0, column=1, sticky="nsew", pady=10)
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
        
        ctk.CTkButton(btn_frame, text="Mark as Seen", command=self._on_mark_as_seen, 
                      fg_color=self.COLORS['success'], hover_color="#00b3a1", width=120).pack(side="left", padx=10)
        
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
                      fg_color=self.COLORS['accent'], hover_color=self.COLORS['accent_hover']).pack(side="left")

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
        
        ctk.CTkButton(log_preview, text="Add to History & Train", command=self._on_add_to_watched, 
                      fg_color=self.COLORS['success'], hover_color="#00b3a1").grid(row=2, column=0, pady=20)

    def setup_console_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        self.console_output = ctk.CTkTextbox(parent, wrap=tk.WORD, font=('Consolas', 10), state='disabled', 
                                             fg_color=self.COLORS['bg_main'], text_color="#00FF00")
        self.console_output.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # --- Event Handlers & Logic ---

    def _save_config(self, path):
        with open(CONFIG_FILE, 'w') as f: json.dump({'watched_path': path}, f)

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
            
            mood = self.selected_mood_var.get()
            ctx = self.context_var.get()
            
            if not mood: print("Error: Select mood."); return

            print(f"Analyzing for Mood: {mood.upper()} | Context: {ctx.upper()}...")
            
            picks = analyze(
                self.watchedSet_titles, 
                self.watchedSet_ids, 
                self.hated_movies,
                self.mood_map.get(mood), 
                self.ai_model, 
                self.ai_columns, 
                self.ai_vectorizer, 
                ctx
            )
            
            if picks:
                print(f"\nFound {len(picks)} recommendations.")
                for m in picks[:30]:
                    year = m['release_date'].split('-')[0] if m.get('release_date') else "N/A"
                    score = m.get('ai_score', 0)
                    
                    text_label = f"{m['title']} ({year})"
                    
                    # Color coding logic
                    btn_color = self.COLORS['bg_card_hover']
                    text_color = self.COLORS['text_main']
                    
                    if score > 4.2: 
                        text_label += f"  ★ {score:.1f}"
                        text_color = self.COLORS['score_high']
                    elif score < 3.0: 
                        text_color = self.COLORS['score_low']
                    elif score > 0:
                        text_label += f"  ★ {score:.1f}"

                    btn = ctk.CTkButton(self.results_scroll, text=text_label, anchor="w", 
                                        fg_color=btn_color, hover_color="#3A3A3A",
                                        text_color=text_color,
                                        font=('Segoe UI', 12),
                                        command=lambda x=m: self._on_result_click(x, "res"),
                                        height=35)
                    btn.pack(fill='x', padx=5, pady=3)
                    self.current_results[btn] = m 
            else:
                print("No results found. Try a different mood.")
            
            self.notebook.set('Recommendations') # Switch tab
            self.console_output.configure(state='disabled')
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

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
            if score_w: 
                score_w.configure(text=f"AI Match: {score:.1f} / 5.0")
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

    def _on_mark_as_seen(self):
        if not getattr(self, 'selected_result_btn', None): return
        m = self.current_results.get(self.selected_result_btn)
        if m:
            self._add_memory(m)
            self.selected_result_btn.destroy()
            self._clear_preview(self.res_poster, self.res_text, self.res_score)

    def _on_add_to_watched(self):
        if not getattr(self, 'selected_search_btn', None): return
        m = self.current_search_results.get(self.selected_search_btn)
        if m:
            self._add_memory(m)
            self.selected_search_btn.destroy()
            self._clear_preview(self.log_poster, self.log_text)

    def _on_view_details(self):
        if not getattr(self, 'selected_result_btn', None): return
        m = self.current_results.get(self.selected_result_btn)
        if m: webbrowser.open_new_tab(f"https://www.themoviedb.org/movie/{m['id']}")

    def _add_memory(self, m):
        if self.watchedSet_ids and m['id'] in self.watchedSet_ids: return
        if self.watchedSet_ids: self.watchedSet_ids.add(m['id'])
        else: self.watchedSet_ids = {m['id']}
        try:
            with open(APP_MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{m["id"]},"{m["title"]}"\n')
            print(f"Logged '{m['title']}'.")
            messagebox.showinfo("Logged", f"Marked '{m['title']}' as watched.")
        except: pass

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
        root_temp = ctk.CTk()
        root_temp.withdraw()
        w_path = filedialog.askopenfilename(title="Select watched.csv", filetypes=(("CSV","*.csv"),))
        root_temp.destroy()
        
        if not w_path:
            w_path = get_path('my_watched_movies.csv')
            if not os.path.exists(w_path):
                with open(w_path, 'w') as f: f.write("Name,Year,Rating\n")
            print("Using fresh/empty history.")
            
        with open(CONFIG_FILE,'w') as f: json.dump({'watched_path':w_path}, f)

    t, i, h = watchedMovies(w_path, APP_MEMORY_FILE)
    
    app = App(w_path, t, i, h)
    app.mainloop()