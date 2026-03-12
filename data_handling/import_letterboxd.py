import os
import zipfile
import pandas as pd
import requests
from dotenv import load_dotenv
import re
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

TMDB_KEY = os.getenv('TMDB_key')
BASE_URL = "https://api.themoviedb.org/3"

def titleNormalize(title):
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def extract_letterboxd_zip(zip_path, extract_to_dir="dataset/temp_letterboxd"):
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            target_files = ['ratings.csv', 'watched.csv']
            extracted = []
            for file_info in zip_ref.infolist():
                if any(file_info.filename.endswith(t) for t in target_files):
                    zip_ref.extract(file_info, extract_to_dir)
                    extracted.append(os.path.join(extract_to_dir, file_info.filename))
            print(f"Successfully extracted: {[os.path.basename(f) for f in extracted]}")
            return extracted
    except zipfile.BadZipFile:
        print("Error: The provided file is not a valid zip archive.")
        return []
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return []

def hydrate_with_tmdb(df, progress_callback=None):
    print("Hydrating dataset with TMDB metadata (this may take a few minutes)...")
    if 'movie_id' not in df.columns:
        df['movie_id'] = pd.NA
    if 'genres' not in df.columns:
        df['genres'] = ""
    if 'overview' not in df.columns:
        df['overview'] = ""
    if 'Rating' in df.columns:
        df = df.sort_values(by='Rating', ascending=False).reset_index(drop=True)
    total_movies = df.shape[0]
    for index, row in tqdm(df.iterrows(), total=total_movies, desc="Fetching TMDB Data"):
        if progress_callback:
            progress_callback(index + 1, total_movies)
        title = row.get('Name') or row.get('Title')
        year = row.get('Year')
        if pd.isna(title):
            continue
        try:
            search_url = f"{BASE_URL}/search/movie"
            params = {"api_key": TMDB_KEY, "query": title}
            if not pd.isna(year):
                params["year"] = int(year)
            response = requests.get(search_url, params=params).json()
            if response.get("results"):
                movie = response["results"][0]  
                movie_id = movie["id"]
                details_url = f"{BASE_URL}/movie/{movie_id}"
                details_params = {"api_key": TMDB_KEY}
                details = requests.get(details_url, params=details_params).json()
                overview = details.get("overview", "")
                genres = [g["name"] for g in details.get("genres", [])]
                df.at[index, "movie_id"] = movie_id
                df.at[index, "overview"] = overview
                df.at[index, "genres"] = ", ".join(genres)
        except Exception as e:
            pass
    return df

def process_letterboxd_import(zip_path, output_csv_path="dataset/user_profile.csv", progress_callback=None):
    if not TMDB_KEY:
        print("Error: TMDB_key not found in .env. Cannot hydrate data.")
        return False
    extracted_files = extract_letterboxd_zip(zip_path)
    ratings_file = next((f for f in extracted_files if f.endswith('ratings.csv')), None)
    if not ratings_file:
         print("Error: Could not find ratings.csv in the zip file. The ML model requires ratings to train.")
         return False
    print(f"Loading {ratings_file}...")
    df = pd.read_csv(ratings_file)
    df.columns = [c.strip() for c in df.columns]
    hydrated_df = hydrate_with_tmdb(df, progress_callback=progress_callback)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    hydrated_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Success! Fully hydrated user profile saved to {output_csv_path}")
    print(f"Total Movies Processed: {len(hydrated_df)}")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        zip_file = sys.argv[1]
        process_letterboxd_import(zip_file)
    else:
        print("Usage: python import_letterboxd.py path/to/letterboxd-export.zip")
