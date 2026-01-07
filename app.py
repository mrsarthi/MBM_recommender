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
import joblib  # For loading the AI Brain
import numpy as np

# --- 1. Path & Config Setup ---
# --- REPLACEMENT CODE ---
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
# Load API Key
load_dotenv(dotenv_path=get_path('.env'))
key = os.getenv('TMDB_key')
baseUrl = "https://api.themoviedb.org/3"

# File Paths (User Data - Keep these outside the exe usually, but for dev this works)
# If you want watched.csv to stay next to the app, use this:
def get_user_data_path(filename):
    if getattr(sys, 'frozen', False):
        # Next to the .exe
        base = os.path.dirname(sys.executable)
    else:
        # Next to the script
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

CONFIG_FILE = get_user_data_path('config.json')
APP_MEMORY_FILE = get_user_data_path('app_memory_ids.csv')

# AI Model Paths (These use get_path because they are bundled resources)
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
            # Normalize column names just in case
            df.columns = [c.strip() for c in df.columns]
            
            for index, row in df.iterrows():
                # Letterboxd usually has 'Name' or 'Title'
                col_name = 'Name' if 'Name' in df.columns else 'Title'
                if col_name in row:
                    title = titleNormalize(row[col_name])
                    watchedSet_titles.add(title)
                    
                    # VETO LOGIC: If rating is low, add to hated list
                    # Letterboxd rating is 0.5-5.0
                    if 'Rating' in row and pd.notna(row['Rating']):
                        try:
                            if float(row['Rating']) <= 2.5:
                                hated_movies.add(title)
                        except: pass
            print(f"Loaded {len(watchedSet_titles)} movies and {len(hated_movies)} hated movies from CSV.")
    except Exception as e:
        print(f"Warning: Could not read watched file: {e}")

    # 2. Load App Memory (Internal Log)
    try:
        if os.path.exists(app_memory_path) and os.path.getsize(app_memory_path) > 0:
            memLogged = pd.read_csv(app_memory_path)
            if 'movie_id' in memLogged.columns:
                memory_set_ids = set(memLogged['movie_id'].astype(int))
                watchedSet_ids.update(memory_set_ids)
        else:
            # Create if missing
            with open(app_memory_path, 'w', newline='', encoding='utf-8') as f:
                f.write('movie_id,title\n')
    except Exception as e:
        print(f"Warning: Could not read app memory: {e}")
    
    return watchedSet_titles, watchedSet_ids, hated_movies

def load_ai_model():
    """ Loads the Brain (Model), the Blueprint (Columns), and the Translator (Vectorizer). """
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
    """ 
    Predicts rating based on Genres + Context + Plot Summary 
    """
    # 1. Initialize Input Row with 0s
    input_data = {col: 0 for col in model_columns}
    
    # 2. Set Context (One-Hot)
    if f'context_{context}' in input_data:
        input_data[f'context_{context}'] = 1
        
    # 3. Set Genres (Multi-Hot)
    for g in genres:
        if f'genre_{g}' in input_data:
            input_data[f'genre_{g}'] = 1
            
    # 4. Set Default Rating (PG-13 / Unknown = 2)
    # We default this to save an API call per movie, as it's minor
    input_data['rating_encoded'] = 2 

    # 5. Process Summary (TF-IDF) - The "Reading" Part
    if vectorizer and overview:
        try:
            overview_text = str(overview)
            # Transform text to numbers
            tfidf_matrix = vectorizer.transform([overview_text])
            feature_names = [f"summary_{w}" for w in vectorizer.get_feature_names_out()]
            dense_vector = tfidf_matrix.toarray()[0]
            
            # Map numbers to the correct columns
            for name, value in zip(feature_names, dense_vector):
                if name in input_data:
                    input_data[name] = value
        except Exception as e:
            # If text processing fails, we just continue with 0s for summary
            pass

    # 6. Predict
    df_input = pd.DataFrame([input_data])
    # Reorder columns to match training EXACTLY
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

        # Fetch candidates (Loose filters to let AI decide)
        discoverUrl = f"{baseUrl}/discover/movie"
        discoverParams = {
            'api_key': key, 'with_genres': genreIdString,
            'vote_average.gte': 5.5, 'vote_count.gte': 100, 
            'sort_by': 'popularity.desc', 'language': 'en-US', 'page': 1
        }

        results = []
        for _ in range(2): # Fetch 2 pages (~40 movies)
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
                    # Get Genres Names
                    genres = [idToGenre[g] for g in movie.get('genre_ids', []) if g in idToGenre]
                    # Get Plot
                    overview = movie.get('overview', '')
                    
                    # Predict
                    score = predict_score(ai_model, ai_columns, ai_vectorizer, genres, user_context, overview)
                    
                    # --- VETO SYSTEM (Friend Logic) ---
                    # Check if this movie title sounds like a movie they hated
                    # e.g. Hated "Matrix", new movie "Matrix Reloaded" -> Penalty
                    is_vetoed = False
                    for hated in hated_movies:
                        # Simple fuzzy match
                        if (hated in title_norm) or (title_norm in hated):
                            print(f"🚫 Vetoing '{movie['title']}' because user hated '{hated}'")
                            score -= 3.0 # Heavy Penalty
                            is_vetoed = True
                            break
                    
                    if not is_vetoed:
                        # Bonus: If it's a "perfect match" movie, boost it slightly
                        pass 

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
    def __init__(self, watched_path, initialWatchedSet_titles, initialWatchedSet_ids, initialHated):
        super().__init__()
        
        self.watched_path = watched_path 
        self.watchedSet_titles = initialWatchedSet_titles
        self.watchedSet_ids = initialWatchedSet_ids
        self.hated_movies = initialHated
        
        # Load AI
        self.ai_model, self.ai_columns, self.ai_vectorizer = load_ai_model()
        
        self.title("Mood Movie Recommender v4.0 (Robust AI)")
        self.geometry("950x800")
        self.configure(fg_color='#2E2E2E')

        self.mood_map = askForMood()
        self.current_results = {}
        self.current_search_results = {}
        self.poster_base_url = "https://image.tmdb.org/t/p/w200"
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Top Frame ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(top_frame, text="Watched File:", font=('Segoe UI', 11, 'bold')).pack(side="left")
        self.file_path_var = tk.StringVar(value=self.watched_path if self.watched_path else "None")
        ctk.CTkEntry(top_frame, textvariable=self.file_path_var, state='readonly', width=200).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Change", command=self._on_browse_click, width=60).pack(side="left", padx=5)

        # Context Dropdown
        ctk.CTkLabel(top_frame, text="  Who are you with?", font=('Segoe UI', 11, 'bold')).pack(side="left", padx=(15, 5))
        self.context_var = ctk.StringVar(value="Alone")
        ctx_menu = ctk.CTkOptionMenu(top_frame, variable=self.context_var, values=["Alone", "Friends", "Family", "Partner", "Other"], width=100)
        ctx_menu.pack(side="left")

        # --- Mood Frame ---
        mood_frame = ctk.CTkFrame(self, fg_color="transparent")
        mood_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        ctk.CTkLabel(mood_frame, text="Select your mood:", font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        self.mood_frame_scrollable = ctk.CTkScrollableFrame(mood_frame, height=90, orientation="horizontal", fg_color="#3C3C3C")
        self.mood_frame_scrollable.pack(fill='x', expand=True, pady=5)
        
        self.selected_mood_var = tk.StringVar()
        for mood in self.mood_map.keys():
            ctk.CTkRadioButton(self.mood_frame_scrollable, text=mood.title(), variable=self.selected_mood_var, value=mood).pack(side="left", padx=10)

        # --- Run Button ---
        ctk.CTkButton(self, text="Generate Recommendations", command=self._on_analyze_click, height=40, font=('Segoe UI', 12, 'bold')).grid(row=2, column=0, pady=10, padx=15)

        # --- Tab View ---
        notebook = ctk.CTkTabview(self, fg_color="#3C3C3C")
        notebook.grid(row=3, column=0, sticky="nsew", padx=15, pady=(5, 15))
        notebook.add('Results')
        notebook.add('Log a Movie')
        notebook.add('Log')
        
        # RESULTS TAB
        res_tab = notebook.tab('Results')
        res_tab.columnconfigure(0, weight=1)
        res_tab.columnconfigure(1, weight=1)
        res_tab.rowconfigure(0, weight=1)
        
        self.results_scroll = ctk.CTkScrollableFrame(res_tab, fg_color="#2B2B2B", label_text="Ranked for You")
        self.results_scroll.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        
        res_preview = ctk.CTkFrame(res_tab, fg_color="#2B2B2B")
        res_preview.grid(row=0, column=1, sticky="nsew")
        res_preview.columnconfigure(0, weight=1)
        res_preview.rowconfigure(2, weight=1)

        self.res_poster = ctk.CTkLabel(res_preview, text="Select a movie...", width=200, height=300, fg_color="#1E1E1E", corner_radius=5)
        self.res_poster.grid(row=0, column=0, pady=10)
        self.res_score = ctk.CTkLabel(res_preview, text="", font=('Segoe UI', 14, 'bold'), text_color="#00CC66")
        self.res_score.grid(row=1, column=0, pady=(0, 5))
        self.res_text = ctk.CTkTextbox(res_preview, wrap=tk.WORD, height=10, state='disabled', fg_color="#1E1E1E")
        self.res_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        btn_frame = ctk.CTkFrame(res_preview, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=10)
        ctk.CTkButton(btn_frame, text="Mark Seen", command=self._on_mark_as_seen, width=90, fg_color="#D9534F", hover_color="#C9302C").pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="View Details", command=self._on_view_details, width=90).pack(side="left", padx=5)

        # LOG TAB
        log_tab = notebook.tab('Log a Movie')
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(1, weight=1)
        
        search_bar = ctk.CTkFrame(log_tab, fg_color="transparent")
        search_bar.grid(row=0, column=0, sticky="ew", pady=10)
        self.search_entry = ctk.CTkEntry(search_bar, placeholder_text="Movie title...", width=300)
        self.search_entry.pack(side="left", padx=10)
        ctk.CTkButton(search_bar, text="Search", command=self._on_tmdb_search).pack(side="left")

        log_split = ctk.CTkFrame(log_tab, fg_color="transparent")
        log_split.grid(row=1, column=0, sticky="nsew")
        log_split.columnconfigure(0, weight=1)
        log_split.columnconfigure(1, weight=1)
        log_split.rowconfigure(0, weight=1)
        
        self.search_scroll = ctk.CTkScrollableFrame(log_split, fg_color="#2B2B2B", label_text="Results")
        self.search_scroll.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        
        log_preview = ctk.CTkFrame(log_split, fg_color="#2B2B2B")
        log_preview.grid(row=0, column=1, sticky="nsew")
        log_preview.columnconfigure(0, weight=1)
        log_preview.rowconfigure(1, weight=1)
        
        self.log_poster = ctk.CTkLabel(log_preview, text="", width=150, height=225, fg_color="#1E1E1E")
        self.log_poster.grid(row=0, column=0, pady=10)
        self.log_text = ctk.CTkTextbox(log_preview, wrap=tk.WORD, height=10, state='disabled', fg_color="#1E1E1E")
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        ctk.CTkButton(log_preview, text="Add to History", command=self._on_add_to_watched, fg_color="#28A745").grid(row=2, column=0, pady=10)

        # CONSOLE TAB
        log_con = notebook.tab('Log')
        log_con.columnconfigure(0, weight=1)
        log_con.rowconfigure(0, weight=1)
        self.console_output = ctk.CTkTextbox(log_con, wrap=tk.WORD, font=('Consolas', 10), state='disabled', fg_color="#1E1E1E")
        self.console_output.grid(row=0, column=0, sticky="nsew", pady=5)
        
        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)

    def _save_config(self, path):
        with open(CONFIG_FILE, 'w') as f: json.dump({'watched_path': path}, f)

    def _on_browse_click(self):
        path = filedialog.askopenfilename(title="Select 'watched.csv'", filetypes=(("CSV", "*.csv"),))
        if path:
            self.file_path_var.set(path)
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

            print("Analyzing with AI...")
            # CALL ANALYZE
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
                    
                    text = f"{m['title']} ({year})"
                    color = "#4A4A4A"
                    if score > 0:
                        text += f" | ★ {score:.1f}"
                        if score > 4.2: color = "#006633"
                        elif score < 3.0: color = "#993333"
                    
                    btn = ctk.CTkButton(self.results_scroll, text=text, anchor="w", fg_color=color, hover_color="#5A5A5A",
                                        command=lambda x=m: self._on_result_click(x, "res"))
                    btn.pack(fill='x', padx=5, pady=2)
                    self.current_results[btn] = m # Map Button -> Movie
            
            self.console_output.configure(state='disabled')
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def _on_tmdb_search(self):
        q = self.search_entry.get()
        if not q: return
        for w in self.search_scroll.winfo_children(): w.destroy()
        self.current_search_results.clear()
        self._clear_preview(self.log_poster, self.log_text)
        
        try:
            print(f"Searching: {q}...")
            res = requests.get(f"{baseUrl}/search/movie", params={'api_key':key,'query':q}).json().get('results',[])
            for m in res[:15]:
                if m['id'] not in self.watchedSet_ids:
                    year = m['release_date'].split('-')[0] if m.get('release_date') else "N/A"
                    btn = ctk.CTkButton(self.search_scroll, text=f"{m['title']} ({year})", anchor="w", fg_color="#4A4A4A",
                                        command=lambda x=m: self._on_result_click(x, "log"))
                    btn.pack(fill='x', padx=5, pady=2)
                    self.current_search_results[btn] = m
        except Exception as e: print(e)

    def _on_result_click(self, movie, mode):
        # Identify the clicked button from the dictionary
        target_btn = None
        source_dict = self.current_results if mode == "res" else self.current_search_results
        
        # In a cleaner impl, we'd bind the button directly, but this works for now
        # We need to find which button in our dict maps to this movie object
        for btn, m in source_dict.items():
            if m == movie:
                target_btn = btn
                break
        
        if mode == "res":
            self.selected_result_btn = target_btn
            poster, text_w, score_w = self.res_poster, self.res_text, self.res_score
            score = movie.get('ai_score', 0)
            if score_w: score_w.configure(text=f"AI Prediction: {score:.1f} / 5.0")
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
            img.thumbnail((200, 300))
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
        if score_label: score_label.configure(text="")

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
        except: pass


# --- 5. Main Execution ---

if __name__ == "__main__":
    if key is None:
        ctk.set_appearance_mode("dark")
        messagebox.showerror("Error", "TMDB_key missing in .env")
        sys.exit()

    ctk.set_appearance_mode("dark")
    w_path = None
    
    # Check config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: w_path = json.load(f).get('watched_path')
        except: pass
    
    # Startup Sequence: Ask for file, but allow empty start
    if not w_path or not os.path.exists(w_path):
        root_temp = ctk.CTk()
        root_temp.withdraw()
        
        # Ask for file
        msg = "Select your 'watched.csv' (Letterboxd export).\n\nIf you don't have one, press Cancel to start fresh."
        messagebox.showinfo("Welcome", msg)
        
        w_path = filedialog.askopenfilename(title="Select watched.csv", filetypes=(("CSV","*.csv"),))
        
        root_temp.destroy()
        
        # If cancelled, create dummy
        if not w_path:
            w_path = get_path('my_watched_movies.csv')
            if not os.path.exists(w_path):
                with open(w_path, 'w') as f: f.write("Name,Year,Rating\n")
            print("Using fresh/empty history.")
            
        with open(CONFIG_FILE,'w') as f: json.dump({'watched_path':w_path}, f)

    t, i, h = watchedMovies(w_path, APP_MEMORY_FILE)
    
    app = App(w_path, t, i, h)
    app.mainloop()