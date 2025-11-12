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
APP_MEMORY_FILE = app_data_path('app_memory.csv')


def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def watchedMovies(letterboxd_path, app_memory_path):
    watchedSet = set()
    
    try:
        preLogged = pd.read_csv(letterboxd_path)
        watchedSet.update({titleNormalize(name) for name in preLogged['Name'].str.strip()})
        print(f"Loaded {len(watchedSet)} movies from {letterboxd_path}")
    except FileNotFoundError:
        print(f'Error: Could not find "{letterboxd_path}".')
        print("Please restart and select the correct 'watched.csv' file.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {letterboxd_path}: {e}")
        return None

    try:
        if os.path.exists(app_memory_path) and os.path.getsize(app_memory_path) > 0:
            memLogged = pd.read_csv(app_memory_path)
            if 'title' in memLogged.columns:
                memory_set = {titleNormalize(name) for name in memLogged['title'].str.strip()}
                original_size = len(watchedSet)
                watchedSet.update(memory_set)
                print(f"Loaded {len(watchedSet) - original_size} more movies from app memory.")
            else:
                print("Warning: 'app_memory.csv' is missing 'title' column.")
        else:
            print("No app memory file found. Creating one.")
            with open(app_memory_path, 'w', newline='', encoding='utf-8') as f:
                f.write('title\n')
                
    except Exception as e:
        print(f"An error occurred while reading {app_memory_path}: {e}")
    
    return watchedSet


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


def analyze(watchedSet, desiredGenre):
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
                movieTitle = titleNormalize(movie['title'].strip())
                if movieTitle not in watchedSet:
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
    def __init__(self, root, watched_path, initialWatchedSet):
        self.root = root
        self.watched_path = watched_path 
        self.watchedSet = initialWatchedSet
        
        root.title("Mood Movie Recommender v1.4")
        # root.geometry("750x850")
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
        self.current_search_results = {} # ADDED: A dictionary to hold movies from the "Log a Movie" tab
        
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
        
        # --- ADDED: This is the new "Log a Movie" tab ---
        log_movie_frame = ttk.Frame(notebook)
        notebook.add(log_movie_frame, text='Log a Movie')
        
        # ADDED: A frame for the search bar
        search_bar_frame = ttk.Frame(log_movie_frame, padding="5 10")
        search_bar_frame.pack(fill='x')

        ttk.Label(search_bar_frame, text="Search for a movie:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_entry = ttk.Entry(search_bar_frame, width=30, font=('Segoe UI', 10))
        self.search_entry.pack(side=tk.LEFT, fill='x', expand=True, ipady=4)
        
        search_button = ttk.Button(search_bar_frame, text="Search TMDB", command=self._on_tmdb_search, style='TButton')
        search_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # ADDED: A frame for the search results
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
        # --- End of new "Log a Movie" tab ---

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
            self.watchedSet = watchedMovies(self.watched_path, APP_MEMORY_FILE)
            if self.watchedSet is not None:
                print(f"Loaded {len(self.watchedSet)} total watched movies.")

    def _on_mark_as_seen(self):
        selected_indices = self.results_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a movie from the list first.")
            return

        selected_title_display = self.results_listbox.get(selected_indices[0])
        movie_obj = self.current_results.get(selected_title_display)
        
        if movie_obj:
            # CHANGED: We now call a helper function to do the "add to memory" logic
            self._add_movie_to_memory(movie_obj)
            
            # Remove from the GUI list
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
            
            if self.watchedSet is None:
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
            
            finalPicks = analyze(self.watchedSet, desiredGenre)
            
            if finalPicks:
                print(f"\nFound {len(finalPicks)} recommendations. Populating results list...")
                for movie in finalPicks[:20]:
                    year = movie['release_date'].split('-')[0]
                    rating = movie['vote_average']
                    display_title = f"{movie['title']} ({year}) - Rated: {rating}/10"
                    
                    self.results_listbox.insert(tk.END, display_title)
                    self.current_results[display_title] = movie
            
            self.console_output.configure(state='disabled')
                
        except Exception as e:
            print("--- A CRITICAL ERROR OCCURRED ---")
            print(traceback.format_exc())
            self.console_output.configure(state='disabled')

    # --- ADDED: New function to handle TMDB Search ---
    # --- This is the MODIFIED function. Paste over your old one. ---
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
                'api_key': key,
                'query': query,
                'language': 'en-US'
            }
            
            response = requests.get(search_url, params=search_params)
            
            if response.status_code == 200:
                data = response.json()
                results = data['results']
                
                if results:
                    # --- ADDED: We now keep track of how many movies we add ---
                    movies_added = 0
                    print(f"Found {len(results)} matches. Filtering against your watched history...")

                    for movie in results[:20]:
                        
                        # --- ADDED: This is the new filter logic ---
                        normalized_title = titleNormalize(movie['title'])
                        
                        # We only add the movie if it's NOT in your watchedSet
                        if normalized_title not in self.watchedSet:
                            year = movie['release_date'].split('-')[0] if movie['release_date'] else "N/A"
                            display_title = f"{movie['title']} ({year})"
                            
                            self.search_results_listbox.insert(tk.END, display_title)
                            self.current_search_results[display_title] = movie
                            movies_added += 1 # Count it
                    
                    # --- ADDED: A message if all movies were filtered out ---
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

    # --- ADDED: New function to add a movie from the search tab to memory ---
    def _on_add_to_watched(self):
        selected_indices = self.search_results_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a movie from the search results first.")
            return
            
        selected_title_display = self.search_results_listbox.get(selected_indices[0])
        movie_obj = self.current_search_results.get(selected_title_display)
        
        if movie_obj:
            # We call our new helper function
            self._add_movie_to_memory(movie_obj)
            
            # Remove it from the search list so you can't add it twice
            self.search_results_listbox.delete(selected_indices[0])
        else:
            print(f"Error: Could not find data for {selected_title_display}")

    # --- ADDED: A new helper function to avoid duplicating code ---
    def _add_movie_to_memory(self, movie_obj):
        """Helper to add a movie to the in-memory set and the CSV file."""
        
        normalized_title = titleNormalize(movie_obj['title'])
        
        # Check if we already know about it
        if normalized_title in self.watchedSet:
            print(f"'{movie_obj['title']}' is already in your watched history.")
            return

        # 1. Add to in-memory set
        self.watchedSet.add(normalized_title)
        
        # 2. Add to app_memory.csv
        try:
            # 'a' (append mode) adds to the end of the file
            with open(APP_MEMORY_FILE, 'a', newline='', encoding='utf-8') as f:
                # We save the *real* title, not the normalized one
                f.write(f'"{movie_obj["title"]}"\n')
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
    initialWatchedSet = None
    
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
        initialWatchedSet = watchedMovies(watched_path, APP_MEMORY_FILE)
    else:
        print("No config file found. App will start in an unconfigured state.")
        print("Please select your 'watched.csv' file using the 'Change...' button.")

    root = tk.Tk()
    
    app = App(root, watched_path, initialWatchedSet)
    root.mainloop()