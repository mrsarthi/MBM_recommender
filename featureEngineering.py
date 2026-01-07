import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# --- Configuration ---
INPUT_FILE = 'dataset/V2ModelTrain_Cleaned.csv'
OUTPUT_FILE = 'dataset/V2ModelTrain_ReadyForAI.csv'
VECTORIZER_PATH = 'models/summary_vectorizer.pkl' # NEW: Save the "Translator"

def feature_engineering():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found. Please run dataClean.py first.")
        return

    # 1. Encoding PG Ratings
    print("Encoding PG Ratings...")
    rating_map = {
        'G': 0, 'TV-G': 0, 'PG': 1, 'TV-PG': 1,
        'PG-13': 2, 'TV-14': 2, 'R': 3, 'TV-MA': 3,
        'NC-17': 4, 'NR': 2, 'Unknown': 2
    }
    df['rating_encoded'] = df['pg_rating'].map(rating_map).fillna(2)

    # 2. Encoding Context (With Whom)
    print("Encoding Context (With Whom)...")
    context_dummies = pd.get_dummies(df['with_whom'], prefix='context')
    df = pd.concat([df, context_dummies], axis=1)

    # 3. Encoding Genres
    print("Encoding Genres...")
    df['tag_list'] = df['tag'].fillna("").apply(lambda x: [t.strip() for t in str(x).split(',')])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['tag_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_])
    df = pd.concat([df, genre_df], axis=1)

    # 4. NEW: Encoding Summaries (TF-IDF)
    # This teaches the AI to read the plot
    print("Encoding Summaries (Reading the Plots)...")
    
    # Fill missing summaries
    df['summary'] = df['summary'].fillna("")
    
    # Create the Vectorizer
    # max_features=100 means it will learn the top 100 most important words from your library
    # stop_words='english' removes boring words like "the", "and", "is"
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    
    # Learn the vocabulary and transform the summaries
    tfidf_matrix = tfidf.fit_transform(df['summary'])
    
    # Create a DataFrame for these new features
    # Columns will look like: summary_space, summary_heist, summary_love
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"summary_{w}" for w in tfidf.get_feature_names_out()])
    
    # Attach to main dataframe
    df = pd.concat([df, tfidf_df], axis=1)
    
    # SAVE THE VECTORIZER (Crucial for the App!)
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"✅ Saved Summary Vectorizer to '{VECTORIZER_PATH}'")

    # 5. Cleanup
    # We drop the raw text columns
    drop_cols = ['summary', 'tag', 'with_whom', 'after_feel', 'pg_rating', 'tag_list', 'with_whom_original', 'year']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    final_df = df.drop(columns=drop_cols)

    # Move target to the end
    cols = [c for c in final_df.columns if c != 'user_rating'] + ['user_rating']
    final_df = final_df[cols]

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Success! Feature Engineering Complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"New Matrix Shape: {final_df.shape} (Rows, Features)")
    print(f"Your AI now looks at {final_df.shape[1]} different features per movie!")

if __name__ == "__main__":
    feature_engineering()