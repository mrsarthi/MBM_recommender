import pandas as pd
import joblib

MODEL_PATH = 'models/personal_ai_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

def load_ai():
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        return model, model_columns
    except FileNotFoundError:
        print("Error: Model files not found. Train the model first!")
        return None, None

def predict_rating(model, model_columns, movie_genres, pg_rating, context):
    input_data = {}
    rating_map = {
        'G': 0, 'TV-G': 0, 'PG': 1, 'TV-PG': 1, 
        'PG-13': 2, 'TV-14': 2, 'R': 3, 'TV-MA': 3, 
        'NC-17': 4, 'NR': 2, 'Unknown': 2
    }
    input_data['rating_encoded'] = rating_map.get(pg_rating, 2)
    input_data[f'context_{context}'] = 1
    for genre in movie_genres:
        input_data[f'genre_{genre}'] = 1
    df_input = pd.DataFrame([input_data])
    df_final = df_input.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df_final)
    return prediction[0]

if __name__ == "__main__":
    print("Loading AI...")
    ai_model, ai_columns = load_ai()
    if ai_model:
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