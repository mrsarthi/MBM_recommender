import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('TMDB_key')


def analyze():
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
    x
    desiredGenre = askForMood()
    # print(desiredGenre)


def askForMood():
    mood_genreMap = {
        'happy': ['Comedy', 'Musical', 'Animation', 'Family'],
        'sad': ['Drama', 'Romance'],
        'tense': ['Horror', 'Thriller', 'Mystery', 'Crime'],
        'adventurous': ['Adventure', 'Sci-Fi', 'Fantasy', 'Action'],
        'calm': ['Documentary', 'Drama', 'History']
    }

    mood = input('What are we in mood for today? ').lower()
    target = mood_genreMap.get(mood)

    if target:
        return target
    else:
        return None


analyze()

# if key is None:
#     print("Error!")
#     exit()
# else:
#     print("Successful")

# call_url = 'https://api.themoviedb.org/3'
# urlTocall = f'{call_url}/search/movie'

# dict = {
#     'api_key': key,
#     'query': searchMovie
# }

# print(f"Searching the database for {searchMovie}...")
# response = requests.get(urlTocall, params=dict)

# if response.status_code == 200:
#     data = response.json()
#     rList = data['results']

#     if rList:
#         year = rList[0]['release_date'].split('-')[0]
#         print(f'The movie was released in: {year}')
#     else:
#         print("Something went wrong")

# else:
#     print(f"Oops! Something went wrong. Status code: {response.status_code}")
