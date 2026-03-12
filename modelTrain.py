import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_personal_model(input_file='dataset/user_profile_features.csv', 
                         model_path='models/personal_ai_model.pkl', 
                         columns_path='models/model_columns.pkl'):
    print("Loading personalized data...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run featureEngineering.py first.")
        return False
    df = pd.read_csv(input_file)
    if 'user_rating' not in df.columns:
        print("Error: No 'user_rating' target column found to train the AI.")
        return False
    y = df['user_rating']
    drop_cols = ['user_rating', 'movie_id', 'title', 'Title', 'Name']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    X = X.fillna(0)
    if len(X) < 15:
        print(f"Insufficient data (only {len(X)} movies). The AI needs at least 15 to train properly.")
        return False
    print(f"Features: {X.shape[1]} columns (Genres, Context, Plot Keywords, etc.)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Personal AI on {len(X_train)} movies...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\n--- Results ---")
    print(f"Average AI Prediction Error: ±{mae:.2f} stars")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(columns_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), columns_path)
    print(f"\n✅ Personal Model saved to '{model_path}'")
    print(f"✅ Feature columns saved to '{columns_path}'")
    return True

if __name__ == "__main__":
    train_personal_model()