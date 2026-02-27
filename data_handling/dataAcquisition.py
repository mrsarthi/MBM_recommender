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
    if 'pg_rating' not in newData.columns:
        newData['pg_rating'] = "NR"
    for index, row in tqdm(newData.iterrows(), total=newData.shape[0], desc="Fetching Movie Data"):
        title = row['title']
        year = row['year']
        try:
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
                cert_url = f"{baseUrl}/movie/{movie_id}/release_dates"
                cert_response = requests.get(cert_url, params={"api_key": key}).json()
                rating = "NR"
                cert_results = cert_response.get('results', [])
                for country in cert_results:
                    if country['iso_3166_1'] == 'US':
                        for release in country['release_dates']:
                            if release['certification']:
                                rating = release['certification']
                                break
                        if rating != "NR":
                            break
                newData.at[index, "movie_id"] = movie_id
                newData.at[index, "summary"] = summary
                newData.at[index, "tag"] = ", ".join(genres)
                newData.at[index, "pg_rating"] = rating
        except Exception as e:
            pass
    newData.to_csv('dataset/V2ModelTrain1.0.csv', index=False)
    print("Data collection complete.")

def migrate():
    newData = pd.read_csv('dataset/V2ModelTrain1.0.csv')
    newData['title'] = user_data['Name']
    newData['year'] = user_data['Year']
    newData['user_rating'] = user_data['Rating']
    newData.to_csv('dataset/V2ModelTrain1.0.csv', index=False)
    tmdbDataCollection(newData)

def createCSV():
    smartHeaders = {
        "movie_id": None, "title": None, "year": None, "summary": None,
        "user_rating": None, "tag": None, "pg_rating": None, "with_whom": None, "after_feel": None
    }
    headers = list(smartHeaders.keys())
    df = pd.DataFrame(columns=headers)
    df.to_csv('dataset/V2ModelTrain1.0.csv', index=False)
    print(f"Created 'dataset/V2ModelTrain1.0.csv' with your headers.")
    migrate()

def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def get_us_certification(movie_id):
    if pd.isna(movie_id):
        return "NR"
    url = f"{baseUrl}/movie/{int(movie_id)}/release_dates"
    params = {'api_key': key}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            for country in results:
                if country['iso_3166_1'] == 'US':
                    for release in country['release_dates']:
                        if release['certification']:
                            return release['certification']
            if results and results[0]['release_dates']:
                 first_cert = results[0]['release_dates'][0].get('certification')
                 if first_cert:
                     return first_cert
    except Exception as e:
        pass
    return "NR"

def update_csv():
    if not os.path.exists('dataset/V2ModelTrain.csv'):
        print(f"Error: Could not find file at {'dataset/V2ModelTrain.csv'}")
        return
    print(f"Reading {'dataset/V2ModelTrain.csv'}...")
    df = pd.read_csv('dataset/V2ModelTrain.csv')
    df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce')
    print("Fetching PG ratings from TMDB...")
    tqdm.pandas(desc="Updating Ratings")
    df['pg_rating'] = df['movie_id'].progress_apply(get_us_certification)
    print("Saving updated CSV...")
    df.to_csv('dataset/V2ModelTrain.csv', index=False)
    print("✅ Success! Your V2ModelTrain.csv now has PG ratings.")

if __name__ == "__main__":
    update_csv()