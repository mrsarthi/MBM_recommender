import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILE = '../dataset/V2ModelTrain.csv'
OUTPUT_FILE = '../dataset/V2ModelTrain_Cleaned.csv'

def clean_data():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 1. Drop rows with missing movie_id or title
    # These are useless for training
    initial_count = len(df)
    df = df.dropna(subset=['movie_id', 'title'])
    print(f"Dropped {initial_count - len(df)} rows with missing IDs or titles.")

    # 2. Fix Data Types (Float -> Integer)
    # 'Int64' is a nullable integer type (handles missing years gracefully)
    df['movie_id'] = df['movie_id'].astype('Int64')
    df['year'] = df['year'].astype('Int64')

    # 3. Fill Missing Text Values
    df['summary'] = df['summary'].fillna("No summary available")
    df['tag'] = df['tag'].fillna("Unknown")
    df['pg_rating'] = df['pg_rating'].fillna("NR")

    # 4. Standardize 'with_whom'
    # We map specific descriptions to broad categories
    def categorize_context(text):
        if pd.isna(text):
            return "Unknown"
        text = str(text).lower()
        
        if 'alone' in text:
            return 'Alone'
        elif 'friend' in text:
            return 'Friends'
        elif 'family' in text or 'parent' in text or 'sibling' in text:
            return 'Family'
        elif 'partner' in text or 'date' in text or 'wife' in text or 'husband' in text:
            return 'Partner'
        else:
            return 'Other' # Or keep original text

    df['with_whom_original'] = df['with_whom'] # Keep original just in case
    df['with_whom'] = df['with_whom'].apply(categorize_context)
    print("Standardized 'with_whom' column.")

    # 5. Impute 'NR' PG Ratings based on Genres (Heuristic)
    # This is a "best guess" to fill gaps. 
    def impute_rating(row):
        rating = row['pg_rating']
        tags = str(row['tag']).lower()
        
        # Only change if it's currently Not Rated
        if rating == "NR" or pd.isna(rating):
            if 'family' in tags or 'animation' in tags:
                return 'PG'
            elif 'horror' in tags or 'crime' in tags or 'thriller' in tags:
                return 'R'
            elif 'action' in tags or 'adventure' in tags:
                return 'PG-13'
            else:
                return 'NR' # Still unknown
        return rating

    df['pg_rating'] = df.apply(impute_rating, axis=1)
    print("Imputed missing PG ratings based on genres.")

    # 6. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Cleaned data saved to '{OUTPUT_FILE}'")
    
    # Show a preview
    print("\n--- Preview of Cleaned Data ---")
    print(df[['title', 'year', 'with_whom', 'pg_rating']].head())

if __name__ == "__main__":
    clean_data()