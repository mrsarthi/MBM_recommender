import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import traceback
import re
import json
import webbrowser

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


load_dotenv(dotenv_path=resource_path('.env'))
key = os.getenv('TMDB_key')

baseUrl = "https://api.themoviedb.org/3"

CONFIG_FILE = app_data_path('config.json')
# CHANGED: Our new memory file will now store IDs, so let's rename it
APP_MEMORY_FILE = app_data_path('app_memory_ids.csv')


def titleNormalize(title):
    # This function is still needed for the initial Letterboxd file
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

# CHANGED: This function now returns TWO sets: one for titles, one for IDs
def watchedMovies(letterboxd_path, app_memory_path):
    watchedSet_titles = set()
    watchedSet_ids = set()
    
    # 1. Load the main Letterboxd file (TITLES)
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

    # 2. Load the app's internal memory file (IDs)
    try:
        if os.path.exists(app_memory_path) and os.path.getsize(app_memory_path) > 0:
            memLogged = pd.read_csv(app_memory_path)
            if 'movie_id' in memLogged.columns:
                # CHANGED: We now load the 'movie_id' column into a set of integers
                memory_set_ids = set(memLogged['movie_id'].astype(int))
                print(f"Loaded {len(memory_set_ids)} manually logged movies from app memory.")
                watchedSet_ids.update(memory_set_ids)
            else:
                print("Warning: 'app_memory_ids.csv' is missing 'movie_id' column.")
        else:
            print("No app memory file found. Creating one.")
            with open(app_memory_path, 'w', newline='', encoding='utf-8') as f:
                # CHANGED: The new memory file has an 'movie_id' and 'title' header
                f.write('movie_id,title\n')
                
    except Exception as e:
        print(f"An error occurred while reading {app_memory_path}: {e}")
    
    # CHANGED: Return both sets
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


# CHANGED: 'analyze' now accepts both watched sets
def analyze(watchedSet_titles, watchedSet_ids, desiredGenre):
    genreDict = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
        'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
        'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
        'TV Movie': 10770, 'Thriller': 53, 'War': 10752, 'Western': 37
    }

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
        print(f"Searching for genres: {genreIdString}")

        discoverUrl = f"{baseUrl}/discover/movie"
        discoverParams = {
            'api_key': key, 'with_genres': genreIdString,
            'vote_average.gte': 7.0, 'vote_count.gte': 500,
            'sort_by': 'popularity.desc', 'language': 'en-US'
        }

        response = requests.get(discoverUrl, params=discoverParams)

        if response.status_code == 200:
            data = response.json()
            results = data['results']

            finalPicks = []
            for movie in results:
                # CHANGED: We now check BOTH sets
                movieTitleNorm = titleNormalize(movie['title'].strip())
                movieId = movie['id']
                
                if (movieTitleNorm not in watchedSet_titles) and (movieId not in watchedSet_ids):
                    finalPicks.append(movie)

            if not finalPicks:
                print("Found some movies, but it looks like you've seen them all!")
            
            return finalPicks
        else:
            print(f"Error fetching from TMDB: {response.status_code}")
            print(f"Message: {response.json().get('status_message')}")
    
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


class App:
    # CHANGED: 'initialWatchedSet' is now TWO variables
    def __init__(self, root, watched_path, initialWatchedSet_titles, initialWatchedSet_ids):
        self.root = root
        self.watched_path = watched_path 
        self.watchedSet_titles = initialWatchedSet_titles # CHANGED: Store title set
        self.watchedSet_ids = initialWatchedSet_ids     # CHANGED: Store ID set
        
        root.title("Mood Movie Recommender v1.0")
        appWidth = 750
        appHeight = 850
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()
        x = int((screenWidth / 2) - (appWidth / 2))
        y = int((screenHeight / 2) - (appHeight / 2))
        root.geometry(f"{appWidth}x{appHeight}+{x}+{y}")
        root.configure(bg='#2E2E2E')

        self.style = ttk.Style(root)
        self.style.theme_use('clam')

        BG_COLOR = '#2E2E2E'
        TEXT_COLOR = '#EAEAEA'
        BUTTON_COLOR = '#4A4A4A'
        BUTTON_HOVER = '#5A5A5A'
        ACCENT_COLOR = '#007ACC'

        self.style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 11, 'bold'))
        self.style.configure('TButton', background=BUTTON_COLOR, font=('Segoe UI', 10, 'bold'), borderwidth=0)
        self.style.map('TButton',
            background=[('active', BUTTON_HOVER)],
            foreground=[('active', TEXT_COLOR)]
        )
        self.style.configure('TEntry', fieldbackground='#3C3C3C', foreground=TEXT_COLOR, borderwidth=0)
        self.style.configure('TNotebook', background=BG_COLOR, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=BUTTON_COLOR, foreground=TEXT_COLOR, padding=[10, 5], font=('Segoe UI', 10, 'bold'))
        self.style.map('TNotebook.Tab',
            background=[('selected', ACCENT_COLOR), ('active', BUTTON_HOVER)],
        )
        
        self.mood_map = askForMood()
        
        self.current_results = {}
        self.current_search_results = {}
        
        file_frame = ttk.Frame(root, padding="15 10 15 5")
        file_frame.pack(fill='x')

        ttk.Label(file_frame, text="1. Watched File:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        if self.watched_path:
            self.file_path_var.set(self.watched_path)
            
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state='readonly', width=40)
        file_entry.pack(side=tk.LEFT, fill='x', expand=True, ipady=4)
        
        browse_button = ttk.Button(file_frame, text="Change...", command=self._on_browse_click, style='TButton')
        browse_button.pack(side=tk.LEFT, padx=(10, 0))

        mood_frame = ttk.Frame(root, padding="15 10 15 5")
        mood_frame.pack(fill='x')
        
        ttk.Label(mood_frame, text="2. Select your mood:").pack(anchor='w')
        
        self.mood_listbox = tk.Listbox(mood_frame, height=10, exportselection=False,
                                       background='#3C3C3C', foreground=TEXT_COLOR,
                                       borderwidth=0, relief='flat',
                                       highlightthickness=1, highlightbackground=BUTTON_COLOR,
                                       selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                                       font=('Segoe UI', 10))
        self.mood_listbox.pack(fill='x', expand=True, pady=5)
        
        for mood in self.mood_map.keys():
            self.mood_listbox.insert(tk.END, mood.title())

        run_button = ttk.Button(root, text="Get Recommendations", command=self._on_analyze_click, style='TButton')
        run_button.pack(pady=5, ipady=5, ipadx=10)

        notebook = ttk.Notebook(root, padding="15 10")
        notebook.pack(fill='both', expand=True)
        
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text='Results')
        
        self.results_listbox = tk.Listbox(results_frame, height=15, exportselection=False,
                                          background='#3C3C3C', foreground=TEXT_COLOR,
                                          borderwidth=0, relief='flat',
                                          highlightthickness=1, highlightbackground=BUTTON_COLOR,
                                          selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                                          font=('Segoe UI', 10))
        self.results_listbox.pack(fill='both', expand=True, pady=5)
        
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill='x', pady=5)
        
        mark_seen_button = ttk.Button(button_frame, text="Mark as Seen", command=self._on_mark_as_seen, style='TButton')
        mark_seen_button.pack(side=tk.LEFT, fill='x', expand=True, padx=5, ipady=5)
        
        view_details_button = ttk.Button(button_frame, text="View Details", command=self._on_view_details, style='TButton')
        view_details_button.pack(side=tk.LEFT, fill='x', expand=True, padx=5, ipady=5)
        
        log_movie_frame = ttk.Frame(notebook)
        notebook.add(log_movie_frame, text='Log a Movie')
        
        search_bar_frame = ttk.Frame(log_movie_frame, padding="5 10")
        search_bar_frame.pack(fill='x')

        ttk.Label(search_bar_frame, text="Search for a movie:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_entry = ttk.Entry(search_bar_frame, width=30, font=('Segoe UI', 10))
        self.search_entry.pack(side=tk.LEFT, fill='x', expand=True, ipady=4)
        
        search_button = ttk.Button(search_bar_frame, text="Search TMDB", command=self._on_tmdb_search, style='TButton')
        search_button.pack(side=tk.LEFT, padx=(10, 0))
        
        search_results_frame = ttk.Frame(log_movie_frame, padding="10 5")
        search_results_frame.pack(fill='both', expand=True)
        
        ttk.Label(search_results_frame, text="Search Results:").pack(anchor='w')
        
        self.search_results_listbox = tk.Listbox(search_results_frame, height=15, exportselection=False,
                                                 background='#3C3C3C', foreground=TEXT_COLOR,
                                                 borderwidth=0, relief='flat',
                                                 highlightthickness=1, highlightbackground=BUTTON_COLOR,
                                                 selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                                                 font=('Segoe UI', 10))
        self.search_results_listbox.pack(fill='both', expand=True, pady=5)
        
        add_to_watched_button = ttk.Button(search_results_frame, text="Add to Watched History", command=self._on_add_to_watched, style='TButton')
        add_to_watched_button.pack(fill='x', pady=5, ipady=5)
        
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text='Log')
        
        self.console_output = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15,
                                                        background='#1E1E1E', foreground=TEXT_COLOR,
                                                        borderwidth=0, relief='flat',
                                                        highlightthickness=0, font=('Consolas', 10))
        self.console_output.pack(fill='both', expand=True, pady=5)
        self.console_output.configure(state='disabled')

        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)

    def _save_config(self, path):
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'watched_path': path}, f)
        print(f"Saved config to {CONFIG_FILE}")

    def _on_browse_click(self):
        path = filedialog.askopenfilename(
            title="Please select your Letterboxd 'watched.csv' file",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if path:
            self.file_path_var.set(path)
            self.watched_path = path
            self._save_config(path)
            print(f"File selected: {path}")
            print("Reloading watched list...")
            # CHANGED: Reload both sets
            self.watchedSet_titles, self.watchedSet_ids = watchedMovies(self.watched_path, APP_MEMORY_FILE)
            if self.watchedSet_titles is not None:
                print(f"Loaded {len(self.watchedSet_titles) + len(self.watchedSet_ids)} total watched movies.")

    def _on_mark_as_seen(self):
        selected_indices = self.results_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a movie from the list first.")
            return

        selected_title_display = self.results_listbox.get(selected_indices[0])
        movie_obj = self.current_results.get(selected_title_display)
        
        if movie_obj:
            self._add_movie_to_memory(movie_obj)
            self.results_listbox.delete(selected_indices[0])
            print(f"Marked '{selected_title_display}' as seen.")

    def _on_view_details(self):
        selected_indices = self.results_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a movie from the list first.")
            return

        selected_title_display = self.results_listbox.get(selected_indices[0])
        movie_obj = self.current_results.get(selected_title_display)
        
        if movie_obj:
            movie_id = movie_obj['id']
            url = f"https://www.themoviedb.org/movie/{movie_id}"
            print(f"Opening {url} in browser...")
            webbrowser.open_new_tab(url)
        else:
            print(f"Error: Could not find details for {selected_title_display}")


    def _on_analyze_click(self):
        try:
            self.console_output.configure(state='normal')
            self.console_output.delete('1.0', tk.END)
            
            self.results_listbox.delete(0, tk.END)
            self.current_results.clear()
            
            selected_indices = self.mood_listbox.curselection()
            
            # CHANGED: Check if the title set is loaded
            if self.watchedSet_titles is None:
                 print("Error: Watched list is not loaded.")
                 print("Please click 'Change...' to select your 'watched.csv' file.")
                 self.console_output.configure(state='disabled')
                 return
            
            if not selected_indices:
                print("Error: Please select a mood from the list.")
                self.console_output.configure(state='disabled')
                return

            selected_mood_name = self.mood_listbox.get(selected_indices[0])
            desiredGenre = self.mood_map.get(selected_mood_name.lower())
            
            # CHANGED: Pass both sets to the analyze function
            finalPicks = analyze(self.watchedSet_titles, self.watchedSet_ids, desiredGenre)
            
            if finalPicks:
                print(f"\nFound {len(finalPicks)} recommendations. Populating results list...")
                for movie in finalPicks[:20]:
                    year = movie['release_date'].split('-')[0] if movie['release_date'] else "N/A"
                    rating = movie['vote_average']
                    display_title = f"{movie['title']} ({year}) - Rated: {rating}/10"
                    
                    self.results_listbox.insert(tk.END, display_title)
                    self.current_results[display_title] = movie
            
            self.console_output.configure(state='disabled')
                
        except Exception as e:
            print("--- A CRITICAL ERROR OCCURRED ---")
            print(traceback.format_exc())
            self.console_output.configure(state='disabled')

    def _on_tmdb_search(self):
        try:
            self.console_output.configure(state='normal')
            self.console_output.delete('1.0', tk.END)
            
            query = self.search_entry.get()
            if not query:
                print("Error: Please enter a movie title to search.")
                self.console_output.configure(state='disabled')
                return

            self.search_results_listbox.delete(0, tk.END)
            self.current_search_results.clear()
            
            print(f"Searching TMDB for '{query}'...")
            
            search_url = f"{baseUrl}/search/movie"
            search_params = {
                'api_key': key, 'query': query, 'language': 'en-US'
            }
            
            response = requests.get(search_url, params=search_params)
            
            if response.status_code == 200:
                data = response.json()
                results = data['results']
                
                if results:
                    movies_added = 0
                    print(f"Found {len(results)} matches. Filtering against your watched history...")

                    for movie in results[:20]:
                        # CHANGED: Now we filter by BOTH sets
                        normalized_title = titleNormalize(movie['title'])
                        movie_id = movie['id']
                        
                        if (normalized_title not in self.watchedSet_titles) and (movie_id not in self.watchedSet_ids):
                            year = movie['release_date'].split('-')[0] if movie['release_date'] else "N/A"
                            display_title = f"{movie['title']} ({year})"
                            
                            self.search_results_listbox.insert(tk.END, display_title)
                            self.current_search_results[display_title] = movie
                            movies_added += 1
                    
                    if movies_added == 0:
                        print("Found matches, but you've already logged all of them.")
                else:
                    print(f"No results found for '{query}'.")
            else:
                print(f"Error fetching from TMDB: {response.status_code}")
                print(f"Message: {response.json().get('status_message')}")
                
            self.console_output.configure(state='disabled')
        except Exception as e:
            print("--- A CRITICAL ERROR OCCURRED ---")
            print(traceback.format_exc())
            self.console_output.configure(state='disabled')

    def _on_add_to_watched(self):
        selected_indices = self.search_results_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a movie from the search results first.")
            return
            
        selected_title_display = self.search_results_listbox.get(selected_indices[0])
        movie_obj = self.current_search_results.get(selected_title_display)
        
        if movie_obj:
            self._add_movie_to_memory(movie_obj)
            self.search_results_listbox.delete(selected_indices[0])
        else:
            print(f"Error: Could not find data for {selected_title_display}")

    # CHANGED: This function is now the single source for adding to memory
    def _add_movie_to_memory(self, movie_obj):
        """Helper to add a movie's ID to the in-memory set and the CSV file."""
        
        movie_id = movie_obj['id'] # CHANGED: We use the ID
        
        # CHANGED: Check the ID set
        if self.watchedSet_ids and movie_id in self.watchedSet_ids:
            print(f"'{movie_obj['title']}' is already in your watched history.")
            return
        
        # Also check the title set, just in case
        normalized_title = titleNormalize(movie_obj['title'])
        if self.watchedSet_titles and normalized_title in self.watchedSet_titles:
            print(f"'{movie_obj['title']}' is already in your Letterboxd file.")
            return

        # 1. Add to in-memory ID set
        if self.watchedSet_ids is not None:
            self.watchedSet_ids.add(movie_id)
        else:
            self.watchedSet_ids = {movie_id}
        
        # 2. Add to app_memory_ids.csv
        try:
            with open(APP_MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                # CHANGED: We now write the ID and the Title
                f.write(f'{movie_id},"{movie_obj["title"]}"\n')
            print(f"Successfully logged '{movie_obj['title']}' as seen.")
        except Exception as e:
            print(f"Error saving to app_memory.csv: {e}")


if __name__ == "__main__":
    if key is None:
        def show_key_error():
            error_root = tk.Tk()
            error_root.withdraw()
            messagebox.showerror("Fatal Error", "ERROR: TMDB_key not found. Please check your .env file.")
            error_root.destroy()
        show_key_error()
        sys.exit()

    watched_path = None
    initialWatchedSet_titles = None # CHANGED: We need two sets
    initialWatchedSet_ids = None   # CHANGED: We need two sets
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                path_from_config = config.get('watched_path')
                
                if path_from_config and os.path.exists(path_from_config):
                    watched_path = path_from_config
                else:
                    if path_from_config: 
                        print(f"Saved path not valid: {path_from_config}")
        except Exception as e:
            print(f"Error reading config.json: {e}")

    if watched_path:
        print(f"Config loaded. Loading 'watched' list from: {watched_path}")
        # CHANGED: Load both sets
        initialWatchedSet_titles, initialWatchedSet_ids = watchedMovies(watched_path, APP_MEMORY_FILE)
    else:
        print("No config file found. App will start in an unconfigured state.")
        print("Please select your 'watched.csv' file using the 'Change...' button.")

    root = tk.Tk()
    
    # CHANGED: Pass both sets to the App
    app = App(root, watched_path, initialWatchedSet_titles, initialWatchedSet_ids)
    root.mainloop()