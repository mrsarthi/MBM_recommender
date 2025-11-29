import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
INPUT_FILE = 'dataset/V2ModelTrain_ReadyForAI.csv'
MODEL_PATH = 'models/personal_ai_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

def train():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Training data not found. Run engineer_features.py first.")
        return

    # --- 1. Separate Features (X) and Target (y) ---
    
    # The Target: What we want to predict
    y = df['user_rating']
    
    # The Features: What the AI gets to see to make its prediction
    # We DROP columns that are IDs, Targets, or Leakage (sentiment)
    X = df.drop(columns=[
        'user_rating',      # The answer (don't let the AI see it!)
        'movie_id',         # Random number, implies no pattern
        'title',            # Text name, not useful for this numeric model
        'sentiment_score'   # LEAKAGE: You don't know this before watching
    ])
    
    # Handle any leftover NaNs (just in case)
    X = X.fillna(0)

    # --- 2. Split Data ---
    # We hide 20% of your data to "test" the AI later. 
    # It's like saving some questions for the final exam.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Train the Model ---
    print(f"Training Random Forest on {len(X_train)} movies...")
    
    # n_estimators=100 means "create 100 different decision trees and average them"
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. Evaluate ---
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    # Mean Absolute Error: On average, how many stars is the AI off by?
    mae = mean_absolute_error(y_test, predictions)
    print(f"\n--- Results ---")
    print(f"Average Error: {mae:.2f} stars")
    print(f"(If the AI predicts 4.0, the real rating is usually between {4.0-mae:.2f} and {4.0+mae:.2f})")

    # --- 5. Save ---
    # We save the model AND the list of columns. 
    # We need the column list later to ensure we feed new data in the exact same order.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)
    
    print(f"\n✅ Model saved to '{MODEL_PATH}'")
    print(f"✅ Feature columns saved to '{COLUMNS_PATH}'")

if __name__ == "__main__":
    train()