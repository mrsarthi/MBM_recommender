# ✦ CineAI — Personalized AI Movie Recommender & Letterboxd Companion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)]()

**CineAI** is a personalized AI-powered cinema recommendation engine and modern Letterboxd companion. It learns your unique film taste using a machine learning model trained on your Letterboxd diary, interprets natural language vibes with Google Gemini AI, clusters your watchlist, and predicts how much you will love any movie.

---

## ✨ Features

- **🧠 Personal AI Taste Model**: Custom **Random Forest** regression model trained on your logged Letterboxd ratings, genres, directors, cast, and keyword affinities to predict your exact star rating (±0.6★ error).
- **🎯 Hybrid Mood & Title Search**: Search for movie titles, directors, or natural language vibes (*"gritty 90s cyber thriller with neon aesthetics"*). Powered by Google Gemini AI with instant fallback heuristics.
- **🔖 Intelligent Watchlist with Mood Clusters**: Organize your Letterboxd watchlist into smart clusters (*🛋️ Comfort, 🧠 Mind-Bending, 🍿 Popcorn & Adrenaline, ⏳ Quick <105m*) with streaming platform filters.
- **🎲 "Pick For Me Tonight" Matchmaker**: Tell CineAI your available time and current mood, and let the AI pick the single best movie from your watchlist with a personalized pitch.
- **📖 Visual Film Journal (Diary)**: Browse your entire watch history with high-resolution TMDB posters, dual view modes (**List View ≡** and **Poster Grid ⊞**), and rating filters.
- **🎬 Cinema Spotlight Drawer**: Click any film to inspect its 4K backdrop, synopsis, streaming providers (Netflix, Prime, Apple TV, etc.), and **Post-Watch Ripple Recommendations**.
- **⚡ 1-Click Letterboxd Sync**: Sync your public Letterboxd diary and watchlist with one click, or retrain your AI model anytime directly from the UI.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mrsarthi/MBM_recommender.git
cd MBM_recommender
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Using standard Python venv
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Or using uv (ultra fast)
uv sync
```

### 3. Configure API Keys
Copy the example environment template:
```bash
# On Linux/macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```

Open `.env` and add your API keys:
```env
# TMDB API Key (Free: https://www.themoviedb.org/settings/api)
TMDB_key=YOUR_TMDB_API_KEY_HERE

# Google Gemini API Key (Free: https://aistudio.google.com/)
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# Your Letterboxd Username (Optional)
LETTERBOXD_USERNAME=your_username
```

### 4. Run CineAI
```bash
python run.py
```
Open **[http://127.0.0.1:8899](http://127.0.0.1:8899)** in your web browser.

---

## 🔑 Obtaining Free API Keys

1. **TMDB API Key (The Movie Database)**:
   - Create a free account at [themoviedb.org](https://www.themoviedb.org/).
   - Go to **Settings → API** and generate a Developer API key.
   - Set `TMDB_key=your_key` in `.env`.

2. **Google Gemini API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/).
   - Click **Get API Key** and create a free key.
   - Set `GEMINI_API_KEY=your_key` in `.env`.

---

## 🔄 Personalizing with Your Own Letterboxd Account

1. Launch CineAI (`python run.py`).
2. Click the **Profile / Settings (`⚡`)** icon in the bottom-left dock or header.
3. Enter your **Letterboxd Username** (`@your_username`).
4. Click **"1-Click Diary Sync"** to pull your watched history, or **"Sync Letterboxd Watchlist"** to import your watchlist.
5. Click **"Retrain Personal AI Model"** — CineAI will perform TF-IDF feature engineering and fit your personalized Random Forest model in seconds!

---

## 🧪 Running Automated Tests

CineAI comes with a comprehensive automated test suite covering API endpoints, recommendation filters, Gemini query expansion, title search, watchlist clustering, and diary poster hydration:

```bash
python -m unittest discover tests
```

---

## 📁 Project Architecture

```text
MBM_recommender/
├── backend/
│   ├── api.py                  # Threaded HTTP server & REST endpoints
│   ├── config.py               # Environment configuration & API key validation
│   ├── feature_engineering.py  # TF-IDF, director/cast encoding, and taste feature extraction
│   ├── gemini_client.py        # Gemini AI prompt interpreter & query expander
│   ├── model_train.py          # Random Forest personal regression model training
│   ├── predictions.py          # Real-time star rating predictions & post-watch ripples
│   ├── recommender.py          # Multi-vector movie discovery & direct title search
│   ├── sync_letterboxd.py      # Letterboxd RSS diary scraper & profile merger
│   └── watchlist.py            # Watchlist scraper, mood clustering & matchmaker logic
├── frontend/
│   ├── index.html              # Modern glassmorphism UI & modal dialogs
│   ├── styles.css              # Dark-mode design system & animations
│   └── app.js                  # Frontend state, spotlight drawer, and API bindings
├── tests/                      # Automated test suite (35 tests)
├── user_data/                  # Starter profile, watchlist, and AI model weights
├── .env.example                # Sample environment template
├── requirements.txt            # Python dependencies
└── run.py                      # Application launcher
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
