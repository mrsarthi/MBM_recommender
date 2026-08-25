import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.config import FEATURES_PATH, PROFILE_PATH, VECTORIZER_PATH, ENCODERS_PATH

def feature_engineering(input_file=PROFILE_PATH, output_file=FEATURES_PATH, vectorizer_path=VECTORIZER_PATH, encoders_path=ENCODERS_PATH):
    """
    Transforms user movie profile CSV into vectorized feature matrix.
    Encodes Genres, Directors, Plot Keywords, and NLP summaries.
    """
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return False
        
    print(f"Reading {input_file} for feature engineering...")
    df = pd.read_csv(input_file)
    df.columns = [c.strip() for c in df.columns]
    
    if 'Rating' not in df.columns:
        print("No 'Rating' column found in profile.")
        return False
        
    encoded_dfs = []
    
    # 1. Multi-Hot Encode Genres
    if 'genres' in df.columns:
        genres_series = df['genres'].fillna('').astype(str)
        genres_split = genres_series.apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
        all_genres = sorted(list(set(g for sublist in genres_split for g in sublist)))
        for g in all_genres:
            df[f'genre_{g}'] = genres_split.apply(lambda x: 1 if g in x else 0)
        encoded_dfs.append(df[[f'genre_{g}' for g in all_genres]])
        
    # 2. Directors
    if 'director' in df.columns:
        dir_series = df['director'].fillna('').astype(str).str.strip()
        top_dirs = dir_series.value_counts()
        valid_dirs = top_dirs[top_dirs >= 2].index.tolist()
        if '' in valid_dirs: valid_dirs.remove('')
        for d in valid_dirs:
            df[f'dir_{d}'] = (dir_series == d).astype(int)
        if valid_dirs:
            encoded_dfs.append(df[[f'dir_{d}' for d in valid_dirs]])
            
    # 3. Plot Keywords TF-IDF
    if 'keywords' in df.columns:
        kw_series = df['keywords'].fillna('').astype(str)
        kw_vec = TfidfVectorizer(max_features=25, stop_words='english')
        try:
            kw_matrix = kw_vec.fit_transform(kw_series)
            kw_df = pd.DataFrame(kw_matrix.toarray(), columns=[f'kw_{w}' for w in kw_vec.get_feature_names_out()])
            encoded_dfs.append(kw_df)
        except Exception: pass
        
    # 4. Plot Summaries TF-IDF
    if 'overview' in df.columns:
        ov_series = df['overview'].fillna('').astype(str)
        ov_vec = TfidfVectorizer(max_features=50, stop_words='english')
        try:
            ov_matrix = ov_vec.fit_transform(ov_series)
            ov_df = pd.DataFrame(ov_matrix.toarray(), columns=[f'ov_{w}' for w in ov_vec.get_feature_names_out()])
            encoded_dfs.append(ov_df)
            os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
            joblib.dump(ov_vec, vectorizer_path)
        except Exception: pass

    # Combine Base + Encoded Features
    base_cols = ['Rating']
    if 'Date' in df.columns: base_cols.append('Date')
    if 'Year' in df.columns: base_cols.append('Year')
    
    final_df = pd.concat([df[base_cols]] + encoded_dfs, axis=1)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"✅ Feature Engineering Complete! Matrix Shape: {final_df.shape} -> Saved to {output_file}")
    return True
