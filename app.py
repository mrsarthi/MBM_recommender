import os
import pandas as pd
import requests
from dotenv import load_dotenv
import time

load_dotenv()
key = os.getenv('TMDB_key')

baseUrl = "https://api.themoviedb.org/3"


def watchedMovies():
    try:
        preLogged = pd.read_csv('dataset/watched.csv')
        watchedSet = set(preLogged['Name'].str.strip())
        return watchedSet
    except FileNotFoundError:
        print('Could not find "dataset/watched.csv".')
        print("Please add your Letterboxd 'watched.csv' file to the 'dataset' folder.")
        exit()


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
    print("What are we in the mood for today? Choose a number or type a mood name:")
    for i, m in enumerate(moods, start=1):
        print(f"  {i}. {m.title()}")

    choice = input("Enter choice (number or name): ").strip().lower()

    target = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(moods):
            target = moodGenreMap[moods[idx]]
    else:
        # exact match or prefix match
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


def analyze(watchedSet):
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
    desiredGenre = askForMood()

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
                movieTitle = movie['title'].strip()
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


if __name__ == "__main__":
    if key is None:
        print("ERROR: TMDB_key not found. Please check your .env file.")
        exit()

    print("Loading your 'watched' list...")
    watchedSet = watchedMovies()

    analyze(watchedSet)
