import pandas as pd
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def feature_engineering(input_file='dataset/user_profile.csv', 
                        output_file='dataset/user_profile_features.csv', 
                        vectorizer_path='models/summary_vectorizer.pkl'):
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File not found: {input_file}. Please import your Letterboxd data first.")
        return False
    if 'Rating' not in df.columns:
         print("Error: 'Rating' column missing. Models need user ratings to train.")
         return False
    df = df.rename(columns={'Rating': 'user_rating'})
    print("Encoding Genres...")
    df['genres'] = df['genres'].fillna("")
    df['tag_list'] = df['genres'].apply(lambda x: [t.strip() for t in str(x).split(',') if t.strip()])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['tag_list'])
    if len(mlb.classes_) > 0:
        genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_])
        df = pd.concat([df, genre_df], axis=1)
    print("Encoding Summaries (Reading the Plots)...")
    df['overview'] = df['overview'].fillna("")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = tfidf.fit_transform(df['overview'])
        if len(tfidf.get_feature_names_out()) > 0:
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"summary_{w}" for w in tfidf.get_feature_names_out()])
            df = pd.concat([df, tfidf_df], axis=1)
            os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
            joblib.dump(tfidf, vectorizer_path)
            print(f"✅ Saved Summary Vectorizer to '{vectorizer_path}'")
    except ValueError:
        print("Warning: Overview data empty or insufficient to build TF-IDF vocabulary.")
    drop_cols = ['Name', 'Title', 'Date', 'Letterboxd URI', 'genres', 'overview', 'tag_list', 'Year']
    drop_cols = [c for c in drop_cols if c in df.columns]
    final_df = df.drop(columns=drop_cols)
    cols = [c for c in final_df.columns if c != 'user_rating'] + ['user_rating']
    final_df = final_df[cols]
    final_df = final_df.fillna(0)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print("\n✅ Success! Feature Engineering Complete.")
    print(f"Saved to: {output_file}")
    print(f"New Matrix Shape: {final_df.shape} (Rows, Features)")
    return True

if __name__ == "__main__":
    feature_engineering()