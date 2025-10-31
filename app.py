import os
import pandas as pd
import requests
from dotenv import load_dotenv

dataFrame = pd.read_csv('dataset/ratings.csv')
# load_dotenv()

searchMovie = input("Enter the movie name: ")

yearFinder = dataFrame[dataFrame['Name'] == searchMovie]
yearFinder = yearFinder.iloc[0]['Year']
print(yearFinder)

key = os.getenv('TMDB_key')
if key is None:
    print("Error!")
    exit()
else:
    print("Successful")

call_url = 'https://api.themoviedb.org/3'
movie_id = 551
urlTocall = f'{call_url}/movie/{movie_id}'

params = {
    'api_key': key
}

print(f"Calling TMDB for movie {movie_id}...")
response = requests.get(urlTocall, params=params)

if response.status_code == 200:
    print("...Success! We got a response.")

    data = response.json()

    print(f"Movie Title: {data['title']}")

else:
    print(f"Oops! Something went wrong. Status code: {response.status_code}")

# def askForMood():
#     mood = input('What are we in mood for today? ')

#     if mood == 'horror':
#         print('The Conjuring would be a great choice right about now')


# askForMood()
