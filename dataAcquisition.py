import pandas as pd
import requests
from dotenv import load_dotenv
import os
import re

load_dotenv()

key = os.getenv('TMDB_key')
baseUrl = "https://api.themoviedb.org/3"

user_data = pd.read_csv('dataset/ratings.csv')

# def tmdbDataCollection():
    

def migrate():
    newData = pd.read_csv('dataset/V2ModelTrain.csv')
    newData['title'] = user_data['Name']
    newData['year'] = user_data['Year']
    newData['user_rating'] = user_data['Rating']

    newData.to_csv('dataset/V2ModelTrain.csv', index=False)

def createCSV():
    smartHeaders = {
        "movie_id": None, "title": None, "year": None, "summary": None, "user_rating": None, "tag": None, "with_whom": None, "after_feel": None
    }

    headers = list(smartHeaders.keys())
    df = pd.DataFrame(columns=headers)
    df.to_csv('dataset/V2ModelTrain.csv', index=False)

    print(f"Created '{'dataset/V2ModelTrain.csv'}' with your headers.")


def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title


if __name__ == "__main__":
    createCSV()
    migrate()
    # tmdbDataCollection()
    