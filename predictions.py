import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
MODEL_PATH = 'models/personal_ai_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

def load_ai():
    """Loads the model and the column definition."""
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        return model, model_columns
    except FileNotFoundError:
        print("Error: Model files not found. Train the model first!")
        return None, None

def predict_rating(model, model_columns, movie_genres, pg_rating, context):
    """
    Prepares the input data and predicts a rating (1-5).
    
    Args:
        model: The trained AI.
        model_columns: List of columns the AI expects.
        movie_genres: List of strings (e.g., ['Action', 'Sci-Fi'])
        pg_rating: String (e.g., 'PG-13')
        context: String (e.g., 'Alone')
    """
    
    # 1. Create a dictionary to hold our features
    input_data = {}

    # --- A. Encode PG Rating (Ordinal) ---
    # We must use the EXACT same map as training
    rating_map = {
        'G': 0, 'TV-G': 0, 'PG': 1, 'TV-PG': 1, 
        'PG-13': 2, 'TV-14': 2, 'R': 3, 'TV-MA': 3, 
        'NC-17': 4, 'NR': 2, 'Unknown': 2
    }
    input_data['rating_encoded'] = rating_map.get(pg_rating, 2) # Default to 2 if unknown

    # --- B. Encode Context (One-Hot) ---
    # Example: If context is 'Alone', we set 'context_Alone' = 1
    input_data[f'context_{context}'] = 1

    # --- C. Encode Genres (Multi-Hot) ---
    # Example: If genres are ['Action', 'Comedy'], set both cols to 1
    for genre in movie_genres:
        input_data[f'genre_{genre}'] = 1

    # 2. Convert to DataFrame
    # This creates a tiny dataframe with just the columns we set above
    df_input = pd.DataFrame([input_data])

    # 3. ALIGN COLUMNS (The Magic Step)
    # The model expects 50+ columns (genre_Western, context_Family, etc.)
    # Our df_input currently only has the 3-4 columns we just set.
    # .reindex() adds all missing columns and fills them with 0.
    df_final = df_input.reindex(columns=model_columns, fill_value=0)

    # 4. Predict
    prediction = model.predict(df_final)
    
    return prediction[0]

# --- Run a Test ---
if __name__ == "__main__":
    print("Loading AI...")
    ai_model, ai_columns = load_ai()

    if ai_model:
        # Let's pretend we are considering watching a new movie
        # Scenario: A dark R-rated Thriller, watching Alone.
        
        test_movie_title = "Bad Boys"
        test_genres = ["Action, Comedy, Crime, Thriller"]
        test_rating = "R"
        test_context = "With friends"

        print(f"\n--- Scenario ---")
        print(f"Movie: {test_movie_title}")
        print(f"Genres: {test_genres}")
        print(f"Rating: {test_rating}")
        print(f"Context: {test_context}")

        print("\nAsking AI for prediction...")
        predicted_score = predict_rating(ai_model, ai_columns, test_genres, test_rating, test_context)

        print(f"🤖 The AI predicts you will rate this: {predicted_score:.2f} / 5.0")