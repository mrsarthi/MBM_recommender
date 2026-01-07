import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- Configuration ---
INPUT_FILE = 'dataset/V2ModelTrain_ReadyForAI.csv'
MODEL_PATH = 'models/personal_ai_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

def train():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run featureEngineering.py first.")
        return

    df = pd.read_csv(INPUT_FILE)

    # --- 1. Separate Features (X) and Target (y) ---
    y = df['user_rating']
    
    # DROP columns that are not features.
    # We removed 'sentiment_score' from this list because it no longer exists.
    drop_cols = ['user_rating', 'movie_id', 'title']
    
    # Safe drop: Only drop columns that actually exist
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    
    # Fill any remaining NaNs with 0
    X = X.fillna(0)

    # --- 2. Split Data ---
    print(f"Features: {X.shape[1]} columns (Genres, Context, Plot Keywords, etc.)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Train the Model ---
    print(f"Training Random Forest on {len(X_train)} movies...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. Evaluate ---
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n--- Results ---")
    print(f"Average Error: {mae:.2f} stars")
    
    # --- 5. Save ---
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)
    
    print(f"\n✅ Model saved to '{MODEL_PATH}'")
    print(f"✅ Feature columns saved to '{COLUMNS_PATH}'")

if __name__ == "__main__":
    train()