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
import joblib  # ADDED: To load the AI model
import numpy as np # ADDED: For data handling

def resource_path(relative_path):
    """ Get absolute path to read-only resources, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def app_data_path(relative_path):
    """ Get absolute path to read/write user files, works for dev and for PyInstaller """
    try:
        base_path = os.path.dirname(sys.executable)
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_base_path():
    """ Returns the folder where this script is located. """
    try:
        # If packaged as an .exe
        base_path = sys._MEIPASS
    except AttributeError:
        # If running as a .py script
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

load_dotenv(dotenv_path=resource_path('.env'))
key = os.getenv('TMDB_key')

baseUrl = "https://api.themoviedb.org/3"

CONFIG_FILE = app_data_path('config.json')
APP_MEMORY_FILE = app_data_path('app_memory_ids.csv')
# ADDED: Paths to your model files
MODEL_PATH = os.path.join(get_base_path(), 'models', 'personal_ai_model.pkl')
COLUMNS_PATH = os.path.join(get_base_path(), 'models', 'model_columns.pkl')


def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def watchedMovies(letterboxd_path, app_memory_path):
    watchedSet_titles = set()
    watchedSet_ids = set()
    
    try:
        preLogged = pd.read_csv(letterboxd_path)
        watchedSet_titles.update({titleNormalize(name) for name in preLogged['Name'].str.strip()})
        print(f"Loaded {len(watchedSet_titles)} movies from {letterboxd_path}")
    except FileNotFoundError:
        print(f'Error: Could not find "{letterboxd_path}".')
        print("Please restart and select the correct 'watched.csv' file.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading {letterboxd_path}: {e}")
        return None, None

    try:
        if os.path.exists(app_memory_path) and os.path.getsize(app_memory_path) > 0:
            memLogged = pd.read_csv(app_memory_path)
            if 'movie_id' in memLogged.columns:
                memory_set_ids = set(memLogged['movie_id'].astype(int))
                print(f"Loaded {len(memory_set_ids)} manually logged movies from app memory.")
                watchedSet_ids.update(memory_set_ids)
            else:
                print("Warning: 'app_memory_ids.csv' is missing 'movie_id' column.")
        else:
            print("No app memory file found. Creating one.")
            with open(app_memory_path, 'w', newline='', encoding='utf-8') as f:
                f.write('movie_id,title\n')
                
    except Exception as e:
        print(f"An error occurred while reading {app_memory_path}: {e}")
    
    return watchedSet_titles, watchedSet_ids


def askForMood():
    moodGenreMap = {
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
    return moodGenreMap

# ADDED: Helper to load AI
def load_ai_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
            model = joblib.load(MODEL_PATH)
            columns = joblib.load(COLUMNS_PATH)
            print("✅ AI Model Loaded Successfully.")
            return model, columns
        else:
            print("⚠️ Model files not found. Using standard popularity sort.")
            return None, None
    except Exception as e:
        print(f"⚠️ Error loading AI model: {e}")
        return None, None

# ADDED: Helper to predict rating for a single movie
def predict_movie_score(model, model_columns, movie_genres, context):
    input_data = {}
    
    # Encode PG Rating (Default to 'Unknown' -> 2 to save API calls)
    input_data['rating_encoded'] = 2 

    # Encode Context
    input_data[f'context_{context}'] = 1

    # Encode Genres
    for genre in movie_genres:
        input_data[f'genre_{genre}'] = 1

    # Create DataFrame and align columns
    df_input = pd.DataFrame([input_data])
    df_final = df_input.reindex(columns=model_columns, fill_value=0)

    # Predict
    return model.predict(df_final)[0]


def analyze(watchedSet_titles, watchedSet_ids, desiredGenre, ai_model, ai_columns, user_context):
    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }
    
    # Invert the dictionary to lookup names by ID later
    genre_id_to_name = {v: k for k, v in genreDict.items()}

    if desiredGenre:
        targetGenreIds = []
        for name in desiredGenre:
            genreId = genreDict.get(name)
            if genreId:
                targetGenreIds.append(str(genreId))

        if not targetGenreIds:
            print("Could not find a valid genre ID for that mood.")
            return []

        genreIdString = "|".join(targetGenreIds)
        print(f"Searching TMDB for genres: {genreIdString}")

        discoverUrl = f"{baseUrl}/discover/movie"
        discoverParams = {
            'api_key': key, 'with_genres': genreIdString,
            'vote_average.gte': 6.0, 'vote_count.gte': 100, # Lowered slightly to let AI filter
            'sort_by': 'popularity.desc', 'language': 'en-US',
            'page': 1 
        }

        # Fetch 2 pages to give the AI more options to sort
        all_results = []
        for page in range(1, 3):
            discoverParams['page'] = page
            response = requests.get(discoverUrl, params=discoverParams)
            if response.status_code == 200:
                all_results.extend(response.json().get('results', []))
            else:
                break

        if all_results:
            candidates = []
            print(f"Found {len(all_results)} candidates. Filtering and Ranking...")
            
            for movie in all_results:
                movieTitleNorm = titleNormalize(movie['title'].strip())
                movieId = movie['id']
                
                # Filter Watched
                if (movieTitleNorm not in watchedSet_titles) and (movieId not in watchedSet_ids):
                    
                    # AI PREDICTION STEP
                    if ai_model and ai_columns:
                        # Convert genre IDs back to names for the AI
                        movie_genre_names = [genre_id_to_name.get(gid) for gid in movie.get('genre_ids', []) if gid in genre_id_to_name]
                        
                        # Predict
                        predicted_score = predict_movie_score(ai_model, ai_columns, movie_genre_names, user_context)
                        movie['ai_score'] = predicted_score
                    else:
                        movie['ai_score'] = 0 # Default if no AI
                    
                    candidates.append(movie)

            # SORT by AI Score (Descending)
            if ai_model:
                candidates.sort(key=lambda x: x['ai_score'], reverse=True)
                print("Movies sorted by AI prediction.")
            
            if not candidates:
                print("Found some movies, but it looks like you've seen them all!")
            
            return candidates
        else:
            print("Error fetching from TMDB.")
    
    return []


class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
    
    def flush(self):
        pass


class App(ctk.CTk):
    def __init__(self, watched_path, initialWatchedSet_titles, initialWatchedSet_ids):
        super().__init__()
        
        self.watched_path = watched_path 
        self.watchedSet_titles = initialWatchedSet_titles
        self.watchedSet_ids = initialWatchedSet_ids
        
        # Load AI
        self.ai_model, self.ai_columns = load_ai_model()
        
        self.title("Mood Movie Recommender v2.0 (AI Powered)")
        appWidth = 950 
        appHeight = 800
        screenWidth = self.winfo_screenwidth()
        screenHeight = self.winfo_screenheight()
        x = int((screenWidth / 2) - (appWidth / 2))
        y = int((screenHeight / 2) - (appHeight / 2))
        self.geometry(f"{appWidth}x{appHeight}+{x}+{y}")
        self.configure(fg_color='#2E2E2E')

        self.mood_map = askForMood()
        self.current_results = {}
        self.current_search_results = {}
        self.poster_base_url = "https://image.tmdb.org/t/p/w200"
        
        # Layout Config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) # Tab view expands

        # --- 1. File & Context Frame ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        
        # File Section
        label_file = ctk.CTkLabel(top_frame, text="Watched File:", font=('Segoe UI', 11, 'bold'))
        label_file.pack(side="left", padx=(0, 5))
        
        self.file_path_var = tk.StringVar(value=self.watched_path if self.watched_path else "")
        file_entry = ctk.CTkEntry(top_frame, textvariable=self.file_path_var, state='readonly', width=200)
        file_entry.pack(side="left", padx=5)
        
        browse_button = ctk.CTkButton(top_frame, text="Change", command=self._on_browse_click, width=60)
        browse_button.pack(side="left", padx=5)

        # ADDED: Context Dropdown
        label_ctx = ctk.CTkLabel(top_frame, text="  Who are you with?", font=('Segoe UI', 11, 'bold'))
        label_ctx.pack(side="left", padx=(15, 5))
        
        self.context_var = ctk.StringVar(value="Alone")
        context_dropdown = ctk.CTkOptionMenu(top_frame, variable=self.context_var, values=["Alone", "Friends", "Family", "Partner", "Other"], width=100)
        context_dropdown.pack(side="left")

        # --- 2. Mood Selection Frame ---
        mood_frame = ctk.CTkFrame(self, fg_color="transparent")
        mood_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        
        label_mood = ctk.CTkLabel(mood_frame, text="Select your mood:", font=('Segoe UI', 11, 'bold'))
        label_mood.pack(anchor='w')
        
        self.mood_frame_scrollable = ctk.CTkScrollableFrame(mood_frame, height=100, orientation="horizontal", fg_color="#3C3C3C")
        self.mood_frame_scrollable.pack(fill='x', expand=True, pady=5)
        
        self.selected_mood_var = tk.StringVar()
        
        for mood in self.mood_map.keys():
            rb = ctk.CTkRadioButton(self.mood_frame_scrollable, text=mood.title(), variable=self.selected_mood_var, value=mood)
            rb.pack(side="left", padx=10, pady=5)

        # --- 3. 'Run' Button ---
        run_button = ctk.CTkButton(self, text="Generate Recommendations", command=self._on_analyze_click, height=40, font=('Segoe UI', 12, 'bold'), fg_color="#007ACC", hover_color="#005A9E")
        run_button.grid(row=2, column=0, pady=10, padx=15)

        # --- 4. Tab View ---
        notebook = ctk.CTkTabview(self, fg_color="#3C3C3C")
        notebook.grid(row=3, column=0, sticky="nsew", padx=15, pady=(5, 15))
        notebook.add('Results')
        notebook.add('Log a Movie')
        notebook.add('Log')
        
        # === TAB 1: RESULTS ===
        results_tab = notebook.tab('Results')
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_columnconfigure(1, weight=1)
        results_tab.grid_rowconfigure(0, weight=1)

        # Left: Scrollable List of Buttons
        self.results_scrollable_frame = ctk.CTkScrollableFrame(results_tab, fg_color="#2B2B2B", label_text="Recommendations")
        self.results_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right: Preview Pane
        preview_pane = ctk.CTkFrame(results_tab, fg_color="#2B2B2B")
        preview_pane.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        preview_pane.grid_columnconfigure(0, weight=1)
        preview_pane.grid_rowconfigure(2, weight=1)

        self.results_poster_label = ctk.CTkLabel(preview_pane, text="Select a movie...", width=200, height=300, fg_color="#1E1E1E", corner_radius=5)
        self.results_poster_label.grid(row=0, column=0, pady=10)
        
        self.results_score_label = ctk.CTkLabel(preview_pane, text="", font=('Segoe UI', 12, 'bold'), text_color="#00CC66")
        self.results_score_label.grid(row=1, column=0, pady=0)

        self.results_overview_text = ctk.CTkTextbox(preview_pane, wrap=tk.WORD, height=10, font=('Segoe UI', 11), state='disabled', fg_color="#1E1E1E", border_width=0)
        self.results_overview_text.grid(row=2, column=0, sticky="nsew", pady=5, padx=10)
        
        btn_frame = ctk.CTkFrame(preview_pane, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=10)
        
        ctk.CTkButton(btn_frame, text="Mark Seen", command=self._on_mark_as_seen, width=90, fg_color="#D9534F", hover_color="#C9302C").pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="View Details", command=self._on_view_details, width=90).pack(side="left", padx=5)

        # === TAB 2: LOG MOVIE ===
        log_tab = notebook.tab('Log a Movie')
        log_tab.grid_columnconfigure(0, weight=1)
        log_tab.grid_rowconfigure(1, weight=1)

        search_frame = ctk.CTkFrame(log_tab, fg_color="transparent")
        search_frame.grid(row=0, column=0, sticky="ew", pady=10)
        
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Enter movie title...", width=300)
        self.search_entry.pack(side="left", padx=10)
        ctk.CTkButton(search_frame, text="Search TMDB", command=self._on_tmdb_search).pack(side="left")

        # Split pane for Log tab too
        log_split = ctk.CTkFrame(log_tab, fg_color="transparent")
        log_split.grid(row=1, column=0, sticky="nsew")
        log_split.grid_columnconfigure(0, weight=1)
        log_split.grid_columnconfigure(1, weight=1)
        log_split.grid_rowconfigure(0, weight=1)

        self.search_results_frame = ctk.CTkScrollableFrame(log_split, fg_color="#2B2B2B", label_text="Search Results")
        self.search_results_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        log_preview = ctk.CTkFrame(log_split, fg_color="#2B2B2B")
        log_preview.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        log_preview.grid_columnconfigure(0, weight=1)
        log_preview.grid_rowconfigure(1, weight=1)

        self.log_poster_label = ctk.CTkLabel(log_preview, text="", width=150, height=225, fg_color="#1E1E1E")
        self.log_poster_label.grid(row=0, column=0, pady=10)
        
        self.log_overview_text = ctk.CTkTextbox(log_preview, wrap=tk.WORD, height=10, state='disabled', fg_color="#1E1E1E")
        self.log_overview_text.grid(row=1, column=0, sticky="nsew", pady=5, padx=10)
        
        ctk.CTkButton(log_preview, text="Add to History", command=self._on_add_to_watched, fg_color="#28A745", hover_color="#218838").grid(row=2, column=0, pady=10)

        # === TAB 3: LOG ===
        log_console_tab = notebook.tab('Log')
        log_console_tab.grid_columnconfigure(0, weight=1)
        log_console_tab.grid_rowconfigure(0, weight=1)
        
        self.console_output = ctk.CTkTextbox(log_console_tab, wrap=tk.WORD, font=('Consolas', 10), state='disabled', fg_color="#1E1E1E")
        self.console_output.grid(row=0, column=0, sticky="nsew", pady=5)

        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)

    def _save_config(self, path):
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'watched_path': path}, f)
        print(f"Saved config to {CONFIG_FILE}")

    def _on_browse_click(self):
        path = filedialog.askopenfilename(title="Select 'watched.csv'", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if path:
            self.file_path_var.set(path)
            self.watched_path = path
            self._save_config(path)
            print(f"File selected: {path}")
            self.watchedSet_titles, self.watchedSet_ids = watchedMovies(self.watched_path, APP_MEMORY_FILE)
            if self.watchedSet_titles is not None:
                print(f"Reloaded: {len(self.watchedSet_titles) + len(self.watchedSet_ids)} movies.")

    def _on_mark_as_seen(self):
        try:
            selected_button = self.selected_result_button
            movie_obj = self.current_results.get(selected_button)
        except AttributeError:
            messagebox.showwarning("No Selection", "Please select a movie from the list first.")
            return

        if movie_obj:
            self._add_movie_to_memory(movie_obj)
            selected_button.destroy()
            self.selected_result_button = None
            self._clear_preview_pane(self.results_poster_label, self.results_overview_text, score_label=self.results_score_label)
            print(f"Marked '{movie_obj['title']}' as seen.")

    def _on_view_details(self):
        try:
            selected_button = self.selected_result_button
            movie_obj = self.current_results.get(selected_button)
        except AttributeError:
            messagebox.showwarning("No Selection", "Select a movie first.")
            return

        if movie_obj:
            webbrowser.open_new_tab(f"https://www.themoviedb.org/movie/{movie_obj['id']}")

    def _on_analyze_click(self):
        try:
            self.console_output.configure(state='normal')
            self.console_output.delete('1.0', tk.END)
            
            for widget in self.results_scrollable_frame.winfo_children(): widget.destroy()
            self.current_results.clear()
            self._clear_preview_pane(self.results_poster_label, self.results_overview_text, score_label=self.results_score_label)
            
            selected_mood_name = self.selected_mood_var.get()
            user_context = self.context_var.get() # Get context
            
            if self.watchedSet_titles is None:
                 print("Error: Watched list is not loaded.")
                 return
            
            if not selected_mood_name:
                print("Error: Please select a mood.")
                return

            desiredGenre = self.mood_map.get(selected_mood_name)
            
            # Pass AI model and context to analyze
            finalPicks = analyze(self.watchedSet_titles, self.watchedSet_ids, desiredGenre, self.ai_model, self.ai_columns, user_context)
            
            if finalPicks:
                print(f"\nFound {len(finalPicks)} recommendations.")
                for movie in finalPicks[:30]: # Show top 30
                    year = movie['release_date'].split('-')[0] if movie.get('release_date') else "N/A"
                    ai_score = movie.get('ai_score', 0)
                    
                    # Display Text
                    if ai_score > 0:
                        display_title = f"{movie['title']} ({year}) | AI Rating: {ai_score:.1f}/5"
                        color = "#006633" if ai_score > 4.0 else "#4A4A4A" # Highlight top picks
                    else:
                        display_title = f"{movie['title']} ({year})"
                        color = "#4A4A4A"

                    btn = ctk.CTkButton(self.results_scrollable_frame, text=display_title, anchor="w", fg_color=color, hover_color="#5A5A5A",
                                        command=lambda m=movie: self._on_result_select(m, "result"))
                    btn.pack(fill='x', padx=5, pady=2)
                    self.current_results[btn] = movie
            
            self.console_output.configure(state='disabled')
        except Exception as e:
            print(f"Critical Error: {e}")
            print(traceback.format_exc())

    def _on_tmdb_search(self):
        try:
            query = self.search_entry.get()
            if not query: return

            for widget in self.search_results_frame.winfo_children(): widget.destroy()
            self.current_search_results.clear()
            self._clear_preview_pane(self.log_poster_label, self.log_overview_text)
            
            print(f"Searching: {query}...")
            response = requests.get(f"{baseUrl}/search/movie", params={'api_key': key, 'query': query})
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                for movie in results[:15]:
                    norm_title = titleNormalize(movie['title'])
                    if (norm_title not in self.watchedSet_titles) and (movie['id'] not in self.watchedSet_ids):
                        year = movie['release_date'].split('-')[0] if movie.get('release_date') else "N/A"
                        btn = ctk.CTkButton(self.search_results_frame, text=f"{movie['title']} ({year})", anchor="w", fg_color="#4A4A4A", hover_color="#5A5A5A",
                                            command=lambda m=movie: self._on_result_select(m, "search"))
                        btn.pack(fill='x', padx=5, pady=2)
                        self.current_search_results[btn] = movie
            else:
                print(f"TMDB Error: {response.status_code}")
                
        except Exception as e:
            print(f"Search Error: {e}")

    def _on_result_select(self, movie_obj, list_type):
        if list_type == "result":
            poster_label = self.results_poster_label
            overview_widget = self.results_overview_text
            score_label = self.results_score_label
            # Find which button was clicked
            for btn, movie in self.current_results.items():
                if movie == movie_obj:
                    self.selected_result_button = btn
                    break
            
            # Show AI Score
            score = movie_obj.get('ai_score', 0)
            if score > 0:
                score_label.configure(text=f"AI Predicts: {score:.2f} / 5.0")
            else:
                score_label.configure(text="")

        else:
            poster_label = self.log_poster_label
            overview_widget = self.log_overview_text
            for btn, movie in self.current_search_results.items():
                if movie == movie_obj:
                    self.selected_search_button = btn
                    break

        if movie_obj:
            self._update_overview_text(overview_widget, movie_obj.get('overview', 'No overview.'))
            if movie_obj.get('poster_path'):
                attr = "poster_image" if list_type == "result" else "poster_image_search"
                self._load_poster(movie_obj['poster_path'], poster_label, attr)
            else:
                self._clear_preview_pane(poster_label, None, clear_overview=False)

    def _load_poster(self, poster_path, poster_label, attr):
        try:
            resp = requests.get(f"{self.poster_base_url}{poster_path}", timeout=5)
            img = Image.open(io.BytesIO(resp.content))
            img.thumbnail((200, 300))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            setattr(self, attr, ctk_img)
            poster_label.configure(image=ctk_img, text="")
        except:
            poster_label.configure(image=None, text="No Image")

    def _update_overview_text(self, widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert(tk.END, text)
        widget.configure(state='disabled')

    def _clear_preview_pane(self, poster_label, overview_widget, score_label=None, clear_overview=True):
        poster_label.configure(image=None, text="")
        if score_label: score_label.configure(text="")
        if clear_overview: self._update_overview_text(overview_widget, "")

    def _on_add_to_watched(self):
        try:
            btn = self.selected_search_button
            movie = self.current_search_results.get(btn)
            if movie:
                self._add_movie_to_memory(movie)
                btn.destroy()
                self._clear_preview_pane(self.log_poster_label, self.log_overview_text)
        except AttributeError:
            messagebox.showwarning("Selection", "Select a movie first.")

    def _add_movie_to_memory(self, movie_obj):
        movie_id = movie_obj['id']
        if self.watchedSet_ids and movie_id in self.watchedSet_ids:
            print("Already watched.")
            return
        
        if self.watchedSet_ids: self.watchedSet_ids.add(movie_id)
        else: self.watchedSet_ids = {movie_id}
        
        try:
            with open(APP_MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{movie_id},"{movie_obj["title"]}"\n')
            print(f"Logged '{movie_obj['title']}'.")
        except: pass

if __name__ == "__main__":
    if key is None:
        ctk.set_appearance_mode("dark")
        messagebox.showerror("Error", "TMDB_key not found.")
        sys.exit()

    ctk.set_appearance_mode("dark")
    watched_path = None
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                path = config.get('watched_path')
                if path and os.path.exists(path): watched_path = path
        except: pass

    if not watched_path:
        # Ask for file
        tr = ctk.CTk()
        tr.withdraw()
        watched_path = filedialog.askopenfilename(title="Select 'watched.csv'", filetypes=(("CSV", "*.csv"),))
        tr.destroy()
        if not watched_path: sys.exit()
        with open(CONFIG_FILE, 'w') as f: json.dump({'watched_path': watched_path}, f)

    t, i = watchedMovies(watched_path, APP_MEMORY_FILE)
    if t is None: sys.exit()

    app = App(watched_path, t, i)
    app.mainloop()