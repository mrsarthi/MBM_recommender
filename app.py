# import os
# import pandas as pd
# import requests
# from dotenv import load_dotenv

# dataFrame = pd.read_csv('dataset/ratings.csv')
# load_dotenv()
# key = os.getenv('TMDB_key')

# searchMovie = input("Enter the movie name: ")


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
        print(f"Got it! Looking for {', '.join(target)} movies")
        return target
    else:
        return None


print(askForMood())


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
