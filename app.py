import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
# import time
import re
import json

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def app_data_path(relative_path):
    try:
        base_path = os.path.dirname(sys.executable)
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

load_dotenv(dotenv_path=resource_path('.env'))
key = os.getenv('TMDB_key')

baseUrl = "https://api.themoviedb.org/3"

CONFIG_FILE = app_data_path('config.json')
APP_MEMORY_FILE = app_data_path('app_memory_ids.csv')

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

    moods = list(moodGenreMap.keys())
    print("\nWhat are we in the mood for today? Choose a number or type a mood name:")
    for i, m in enumerate(moods, start=1):
        print(f"  {i}. {m.title()}")
    print("\nType 'exit' or 'quit' to close the app.")

    choice = input("Enter choice (number or name): ").strip().lower()

    if choice == 'exit' or choice == 'quit':
        return "exit"

    target = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(moods):
            target = moodGenreMap[moods[idx]]
    else:
        for key in moods:
            if choice == key.lower() or key.lower().startswith(choice):
                target = moodGenreMap[key]
                break

    if target:
        print(f"Got it. Looking for {'| '.join(target)} movies.")
        return target
    else:
        print(f"Sorry, I don't have a 'mood setting' for '{choice}'.")
        return None

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

        response = requests.get(discoverUrl, params=discoverParams)

        if response.status_code == 200:
            data = response.json()
            results = data['results']

            finalPicks = []
            for movie in results:
                movieTitle = titleNormalize(movie['title'].strip())
                movieId = movie['id']
                if (movieTitle not in watchedSet_titles) and (movieId not in watchedSet_ids):
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

if __name__ == "__main__":
    if key is None:
        print("ERROR: TMDB_key not found. Please check your .env file.")
        sys.exit()

    watched_path = None
    initialWatchedSet_titles = None
    initialWatchedSet_ids = None
    
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

    if not watched_path:
        print("--- First-Time Setup ---")
        print("No 'watched.csv' file path found in config.")
        
        while True:
            path_input = input("Please paste the FULL path to your Letterboxd 'watched.csv' file: ").strip()
            if path_input.startswith('"') and path_input.endswith('"'):
                path_input = path_input[1:-1]
                
            if os.path.exists(path_input):
                watched_path = path_input
                try:
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump({'watched_path': watched_path}, f)
                    print(f"Saved path to {CONFIG_FILE}")
                except Exception as e:
                    print(f"Error saving config: {e}")
                break
            else:
                print("\nError: File not found at that path. Please try again.")
    
    print(f"\nConfig loaded. Loading 'watched' list from: {watched_path}")
    initialWatchedSet_titles, initialWatchedSet_ids = watchedMovies(watched_path, APP_MEMORY_FILE)
    
    if initialWatchedSet_titles is None:
        print("Could not load the watched list. Please check the file and restart.")
        sys.exit()

    print("\nWelcome to the Mood Movie Recommender!")
    print(f"Loaded {len(initialWatchedSet_titles) + len(initialWatchedSet_ids)} total watched movies.")

    while True:
        desiredGenre = askForMood()
        if desiredGenre == "exit":
            print("Goodbye!")
            break
        if desiredGenre:
            analyze(initialWatchedSet_titles, initialWatchedSet_ids, desiredGenre)
        print("\n" + "="*40 + "\n")