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


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


load_dotenv(dotenv_path=resource_path('.env'))
key = os.getenv('TMDB_key')

baseUrl = "https://api.themoviedb.org/3"


def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title


def watchedMovies(watched_csv_path):
    try:
        preLogged = pd.read_csv(watched_csv_path)
        watchedSet = {titleNormalize(name) for name in preLogged['Name'].str.strip()}
        return watchedSet
    except FileNotFoundError:
        print(f'Could not find "{watched_csv_path}".')
        print("Please restart and select the correct 'watched.csv' file.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


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
        'Action': 28,
        'Adventure': 12,
        'Animation': 16,
        'Comedy': 35,
        'Crime': 80,
        'Documentary': 99,
        'Drama': 18,
        'Family': 10751,
        'Fantasy': 14,
        'History': 36,
        'Horror': 27,
        'Music': 10402,
        'Mystery': 9648,
        'Romance': 10749,
        'Science Fiction': 878,
        'TV Movie': 10770,
        'Thriller': 53,
        'War': 10752,
        'Western': 37
    }

    if desiredGenre:
        targetGenreIds = []
        for name in desiredGenre:
            genreId = genreDict.get(name)
            if genreId:
                targetGenreIds.append(str(genreId))

        if not targetGenreIds:
            print("Could not find a valid genre ID for that mood.")
            return

        genreIdString = "|".join(targetGenreIds)
        print(f"Searching for genres: {genreIdString}")

        discoverUrl = f"{baseUrl}/discover/movie"

        discoverParams = {
            'api_key': key,
            'with_genres': genreIdString,
            'vote_average.gte': 7.0,
            'vote_count.gte': 500,
            'sort_by': 'popularity.desc',
            'language': 'en-US'
        }

        time.sleep(1)

        response = requests.get(discoverUrl, params=discoverParams)

        if response.status_code == 200:
            data = response.json()
            results = data['results']

            finalPicks = []
            for movie in results:
                movieTitle = titleNormalize(movie['title'].strip())
                if movieTitle not in watchedSet:
                    finalPicks.append(movie)

            if finalPicks:
                print("\nHere are some movies you might like:")
                for pick in finalPicks[:10]:
                    year = pick['release_date'].split('-')[0]
                    rating = pick['vote_average']
                    print(f"- {pick['title']} ({year}) - Rated: {rating}/10")
            else:
                print("Found some movies, but it looks like you've seen them all!")
        else:
            print(f"Error fetching from TMDB: {response.status_code}")
            print(f"Message: {response.json().get('status_message')}")


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
    def __init__(self, root):
        self.root = root
        root.title("Mood Movie Recommender v1.0")
        root.geometry("650x700")
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
        
        self.mood_map = askForMood()
        
        file_frame = ttk.Frame(root, padding="15 10")
        file_frame.pack(fill='x')

        ttk.Label(file_frame, text="1. Select 'watched.csv':").pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state='readonly', width=40)
        file_entry.pack(side=tk.LEFT, fill='x', expand=True, ipady=4)
        
        browse_button = ttk.Button(file_frame, text="Browse...", command=self._on_browse_click, style='TButton')
        browse_button.pack(side=tk.LEFT, padx=(10, 0))

        mood_frame = ttk.Frame(root, padding="15 10")
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
        run_button.pack(pady=10, ipady=5, ipadx=10)

        console_frame = ttk.Frame(root, padding="15 10")
        console_frame.pack(fill='both', expand=True)
        
        ttk.Label(console_frame, text="--- Results ---").pack(anchor='w')
        
        self.console_output = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=15,
                                                        background='#1E1E1E', foreground=TEXT_COLOR,
                                                        borderwidth=0, relief='flat',
                                                        highlightthickness=0,
                                                        font=('Consolas', 10))
        self.console_output.pack(fill='both', expand=True, pady=5)
        self.console_output.configure(state='disabled')

        sys.stdout = ConsoleRedirector(self.console_output)
        sys.stderr = ConsoleRedirector(self.console_output)

    def _on_browse_click(self):
        path = filedialog.askopenfilename(
            title="Please select your Letterboxd 'watched.csv' file",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if path:
            self.file_path_var.set(path)
            print(f"File selected: {path}")

    def _on_analyze_click(self):
        try:
            self.console_output.configure(state='normal')
            self.console_output.delete('1.0', tk.END)
            
            file_path = self.file_path_var.get()
            selected_indices = self.mood_listbox.curselection()

            if not file_path:
                print("Error: Please select your 'watched.csv' file first.")
                self.console_output.configure(state='disabled')
                return
            
            if not selected_indices:
                print("Error: Please select a mood from the list.")
                self.console_output.configure(state='disabled')
                return

            print("Loading 'watched' list...")
            watchedSet = watchedMovies(file_path)
            
            if watchedSet is not None:
                selected_mood_name = self.mood_listbox.get(selected_indices[0])
                
                desiredGenre = self.mood_map.get(selected_mood_name.lower())
                
                analyze(watchedSet, desiredGenre)
            
            self.console_output.configure(state='disabled')
                
        except Exception as e:
            print("--- A CRITICAL ERROR OCCURRED ---")
            print(traceback.format_exc())
            self.console_output.configure(state='disabled')


if __name__ == "__main__":
    if key is None:
        def show_key_error():
            error_root = tk.Tk()
            error_root.withdraw()
            messagebox.showerror("Fatal Error", "ERROR: TMDB_key not found. Please check your .env file.")
            error_root.destroy()
        
        show_key_error()
        sys.exit()

    root = tk.Tk()
    app = App(root)
    root.mainloop()