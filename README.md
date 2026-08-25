# ✦ MBMR — Mood-Based Movie Recommender & Letterboxd Companion

[![Live App](https://img.shields.io/badge/Live%20App-mbmr.onrender.com-success?style=for-the-badge&logo=render)](https://mbmr.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)]()

> **🌐 Try the Live Web App**: **[https://mbmr.onrender.com](https://mbmr.onrender.com)**

**MBMR (Mood-Based Movie Recommender)** is a personalized AI-powered cinema recommendation engine and modern Letterboxd companion. It learns your unique film taste using a machine learning model trained on your Letterboxd diary, interprets natural language vibes and moods with Google Gemini AI, clusters your watchlist, and predicts how much you will love any movie.

---

## ✨ Features

- **🧠 Personal AI Taste Model**: Custom **Random Forest** regression model trained on your logged Letterboxd ratings, genres, directors, cast, and keyword affinities to predict your exact star rating (±0.6★ error).
- **🎭 Mood-Based & Title Search**: Search for movie titles, directors, or natural language vibes (*"gritty 90s cyber thriller with neon aesthetics"*, *"melancholic coming-of-age on a rainy night"*). Powered by Google Gemini AI with instant fallback heuristics.
- **🔖 Intelligent Watchlist with Mood Clusters**: Organize your Letterboxd watchlist into smart clusters (*🛋️ Comfort, 🧠 Mind-Bending, 🍿 Popcorn & Adrenaline, ⏳ Quick <105m*) with streaming platform filters.
- **🎲 "Pick For Me Tonight" Matchmaker**: Tell MBMR your available time and current mood, and let the AI pick the single best movie from your watchlist with a personalized pitch.
- **📖 Visual Film Journal (Diary)**: Browse your entire watch history with high-resolution TMDB posters, dual view modes (**List View ≡** and **Poster Grid ⊞**), and rating filters.
- **🎬 Cinema Spotlight Drawer**: Click any film to inspect its 4K backdrop, synopsis, streaming providers (Netflix, Prime, Apple TV, etc.), and **Post-Watch Ripple Recommendations**.
- **⚡ 1-Click Letterboxd Sync**: Sync your public Letterboxd diary and watchlist with one click, or retrain your AI model anytime directly from the UI.
- **📱 Fully Responsive Mobile App & PWA**: Seamless experience across mobile phones, tablets, and desktops with bottom navigation dock, safe-area padding, and IndexedDB local client storage.

---

## 🌐 Live Web App

MBMR is deployed and ready to use at:
👉 **[https://mbmr.onrender.com](https://mbmr.onrender.com)**

When you visit for the first time:
1. The **MBMR Onboarding Wizard** will prompt you to enter your **Letterboxd Username** (`@handle`).
2. Optionally enter your own free **TMDB** and **Google Gemini** API keys.
3. All credentials and preferences are **stored 100% locally in your browser (IndexedDB)** — your data remains private and is never stored on the server.

---

## 🚀 Quick Start (Run Locally)

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

### 3. Configure API Keys (Optional for Local Development)
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

### 4. Run MBMR
```bash
python run.py
```
Open **[http://localhost:8899](http://localhost:8899)** in your web browser.

---

## ☁️ Deploying to Render (Free Tier)

You can host your own instance of MBMR on **Render**:

1. **Push your code to GitHub**:
   ```bash
   git push origin main
   ```

2. **Create a Free Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com/) and click **New + → Web Service**.
   - Connect your GitHub repository (`MBM_recommender`).

3. **Configure the Service Settings**:
   - **Name**: `mbmr`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
   - **Plan**: `Free`

4. **Environment Variables**:
   - *Leave empty!* MBMR uses client-side IndexedDB onboarding so each visitor provides and securely stores their own keys locally in their browser.

> [!TIP]
> **Preventing Render Free-Tier Sleeping**: Render's free tier spins down after 15 minutes of inactivity. You can use a free 10-minute ping monitor like [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) pointing to `https://mbmr.onrender.com/api/status` to keep your instance warm 24/7.

---

## 🔑 Obtaining Free API Keys

1. **TMDB API Key (The Movie Database)**:
   - Create a free account at [themoviedb.org](https://www.themoviedb.org/).
   - Go to **Settings → API** and generate a Developer API key.

2. **Google Gemini API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/).
   - Click **Get API Key** and generate a free API key.

---

## 🧪 Running Automated Tests

MBMR includes a comprehensive test suite covering API endpoints, recommendation filters, Gemini query expansion, title search, watchlist clustering, and diary poster hydration:

```bash
python -m unittest discover tests
```

---

## 📁 Project Architecture

```text
MBM_recommender/
├── backend/
│   ├── api.py                  # Threaded HTTP server, CORS, & REST endpoints
│   ├── config.py               # Environment configuration & API key validation
│   ├── feature_engineering.py  # TF-IDF, director/cast encoding, and taste feature extraction
│   ├── gemini_client.py        # Gemini AI prompt interpreter & query expander
│   ├── model_train.py          # Random Forest personal regression model training
│   ├── predictions.py          # Real-time star rating predictions & post-watch ripples
│   ├── recommender.py          # Multi-vector movie discovery & direct title search
│   ├── sync_letterboxd.py      # Letterboxd RSS diary scraper & profile merger
│   └── watchlist.py            # Watchlist scraper, mood clustering & matchmaker logic
├── frontend/
│   ├── index.html              # Responsive mobile/desktop shell & onboarding modals
│   ├── styles.css              # Dark-mode design system, mobile media queries, & animations
│   └── app.js                  # Frontend state, IndexedDB local storage, & wake-up handler
├── tests/                      # Automated test suite (35 tests)
├── user_data/                  # Starter profile, watchlist, and AI model weights
├── .env.example                # Sample environment template
├── requirements.txt            # Python dependencies
└── run.py                      # Application launcher (dynamic PORT for Render)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
