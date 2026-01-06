import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob

INPUT_FILE = 'dataset/V2ModelTrain_Cleaned.csv'
OUTPUT_FILE = 'dataset/V2ModelTrain_ReadyForAI.csv'

def feature_engineering():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found. Please run dataClean.py first.")
        return

    print("Encoding PG Ratings...")

    rating_map = {
        'G': 0,
        'TV-G': 0,
        'PG': 1,
        'TV-PG': 1,
        'PG-13': 2,
        'TV-14': 2,
        'R': 3,
        'TV-MA': 3,
        'NC-17': 4,
        'NR': 2,
        'Unknown': 2
    }

    df['rating_encoded'] = df['pg_rating'].map(rating_map).fillna(2)

    print("Encoding Context (With Whom)...")

    context_dummies = pd.get_dummies(df['with_whom'], prefix='context')
    df = pd.concat([df, context_dummies], axis=1)

    print("Encoding Genres...")

    df['tag_list'] = df['tag'].fillna("").apply(lambda x: [t.strip() for t in str(x).split(',')])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['tag_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_])
    df = pd.concat([df, genre_df], axis=1)

    print("Analyzing Sentiment...")

    def get_sentiment(text):
        if pd.isna(text):
            return 0.0
        return TextBlob(str(text)).sentiment.polarity

    df['sentiment_score'] = df['after_feel'].apply(get_sentiment)

    drop_cols = ['summary', 'tag', 'with_whom', 'after_feel', 'pg_rating', 'tag_list', 'with_whom_original', 'year']
    drop_cols = [c for c in drop_cols if c in df.columns]

    final_df = df.drop(columns=drop_cols)

    cols = [c for c in final_df.columns if c != 'user_rating'] + ['user_rating']
    final_df = final_df[cols]

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Success! Feature Engineering Complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"New Matrix Shape: {final_df.shape} (Rows, Features)")
    print("\n--- First 5 Rows of Training Data ---")
    print(final_df.head())

if __name__ == "__main__":
    feature_engineering()