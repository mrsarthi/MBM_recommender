import pandas as pd
import requests
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm   

load_dotenv()

key = os.getenv('TMDB_key')
baseUrl = "https://api.themoviedb.org/3"

user_data = pd.read_csv('dataset/ratings.csv')

def tmdbDataCollection(newData):
    for index, row in tqdm(newData.iterrows(), total=newData.shape[0], desc="Fetching TMDB data"):
        title = row['title']
        year = row['year']

        search_url = f"{baseUrl}/search/movie"
        params = {"api_key": key, "query": title, "year": year}
        response = requests.get(search_url, params=params).json()

        if response.get("results"):
            movie = response["results"][0]  
            movie_id = movie["id"]

            details_url = f"{baseUrl}/movie/{movie_id}"
            details_params = {"api_key": key}
            details = requests.get(details_url, params=details_params).json()

            summary = details.get("overview", "")
            genres = [g["name"] for g in details.get("genres", [])]

            newData.at[index, "movie_id"] = movie_id
            newData.at[index, "summary"] = summary
            newData.at[index, "tag"] = ", ".join(genres)

    newData.to_csv('dataset/V2ModelTrain.csv', index=False)

def migrate():
    newData = pd.read_csv('dataset/V2ModelTrain.csv')
    newData['title'] = user_data['Name']
    newData['year'] = user_data['Year']
    newData['user_rating'] = user_data['Rating']

    newData.to_csv('dataset/V2ModelTrain.csv', index=False)
    tmdbDataCollection(newData)

def createCSV():
    smartHeaders = {
        "movie_id": None, "title": None, "year": None, "summary": None,
        "user_rating": None, "tag": None, "with_whom": None, "after_feel": None
    }

    headers = list(smartHeaders.keys())
    df = pd.DataFrame(columns=headers)
    df.to_csv('dataset/V2ModelTrain.csv', index=False)

    print(f"Created 'dataset/V2ModelTrain.csv' with your headers.")
    migrate()

def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

if __name__ == "__main__":
    createCSV()
