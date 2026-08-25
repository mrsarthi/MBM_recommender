import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from backend.config import FEATURES_PATH, MODEL_PATH, COLUMNS_PATH

def train_personal_model(input_file=FEATURES_PATH, model_path=MODEL_PATH, columns_path=COLUMNS_PATH):
    """
    Trains a recency-weighted Random Forest regressor on user rating history.
    """
    if not os.path.exists(input_file):
        print(f"Feature file not found: {input_file}")
        return False
        
    print(f"Loading personalized feature data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Filter rows with valid ratings
    df = df[df['Rating'].notna()].copy()
    if len(df) < 5:
        print("Not enough rated movies to train a personalized model (minimum 5).")
        return False
        
    y = df['Rating']
    
    # Drop non-feature columns
    drop_cols = ['Rating', 'Date', 'Name', 'Letterboxd URI', 'movie_id', 'overview', 'director', 'cast', 'keywords', 'genres']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    feature_cols = X.columns.tolist()
    
    # Compute recency sample weights (newer watched movies get higher weight)
    sample_weights = np.ones(len(df))
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'], errors='coerce')
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            days_diff = (dates - min_date).dt.days.fillna(0)
            max_days = max(days_diff.max(), 1)
            sample_weights = 0.5 + 0.5 * (days_diff / max_days)
            
    print(f"Training Personal AI on {len(df)} movies with {len(feature_cols)} feature dimensions...")
    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, min_samples_leaf=2)
    model.fit(X, y, sample_weight=sample_weights)
    
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"✅ Model evaluation - Average AI Prediction Error: ±{mae:.2f} stars")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, columns_path)
    print(f"Saved Personal Model -> '{model_path}'")
    return True

if __name__ == "__main__":
    train_personal_model()
