import pandas as pd
from dotenv import load_dotenv
import os
import re

load_dotenv()

key = os.getenv('TMDB_key')
baseUrl = "https://api.themoviedb.org/3"

smartHeaders = {
    "movie_id": None, "title": None, "summary": None, "user_rating": None, "tag": None, "with_whom": None, "after_feel": None
}

headers = list(smartHeaders.keys())

df = pd.DataFrame(columns=headers)

df.to_csv('dataset/V2ModelTrain.csv', index=False)

print(f"Created '{'dataset/V2ModelTrain.csv'}' with your headers.")

params = {
    "api_key": key
}

def titleNormalize(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9]', '', title)
    return title