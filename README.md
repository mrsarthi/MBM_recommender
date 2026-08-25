# ✦ MBMR — Mind-Bending Movie Recommender & Letterboxd Companion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)]()

**MBMR** is a personalized AI-powered cinema recommendation engine and modern Letterboxd companion. It learns your unique film taste using a machine learning model trained on your Letterboxd diary, interprets natural language vibes with Google Gemini AI, clusters your watchlist, and predicts how much you will love any movie.

---

## ✨ Features

- **🧠 Personal AI Taste Model**: Custom **Random Forest** regression model trained on your logged Letterboxd ratings, genres, directors, cast, and keyword affinities to predict your exact star rating (±0.6★ error).
- **🎯 Hybrid Mood & Title Search**: Search for movie titles, directors, or natural language vibes (*"gritty 90s cyber thriller with neon aesthetics"*). Powered by Google Gemini AI with instant fallback heuristics.
- **🔖 Intelligent Watchlist with Mood Clusters**: Organize your Letterboxd watchlist into smart clusters (*🛋️ Comfort, 🧠 Mind-Bending, 🍿 Popcorn & Adrenaline, ⏳ Quick <105m*) with streaming platform filters.
- **🎲 "Pick For Me Tonight" Matchmaker**: Tell MBMR your available time and current mood, and let the AI pick the single best movie from your watchlist with a personalized pitch.
- **📖 Visual Film Journal (Diary)**: Browse your entire watch history with high-resolution TMDB posters, dual view modes (**List View ≡** and **Poster Grid ⊞**), and rating filters.
- **🎬 Cinema Spotlight Drawer**: Click any film to inspect its 4K backdrop, synopsis, streaming providers (Netflix, Prime, Apple TV, etc.), and **Post-Watch Ripple Recommendations**.
- **⚡ 1-Click Letterboxd Sync**: Sync your public Letterboxd diary and watchlist with one click, or retrain your AI model anytime directly from the UI.
- **📱 Fully Responsive Mobile App & PWA**: Seamless experience across mobile phones, tablets, and desktops with bottom navigation bar, safe-area padding, and IndexedDB local client storage.

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

### 4. Run MBMR
```bash
python run.py
```
Open **[http://127.0.0.1:8899](http://127.0.0.1:8899)** in your web browser.

---

## ☁️ Deploying to Render (Free Tier Step-by-Step)

You can host MBMR for free on **Render** so you and your friends can access it from any phone or browser 24/7:

1. **Push your code to GitHub**:
   Make sure your repo is pushed to GitHub (`git push origin main` or `tester`).

2. **Create a Free Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com/) and click **New + → Web Service**.
   - Connect your GitHub repository (`MBM_recommender`).

3. **Configure the Service Settings**:
   - **Name**: `mbmr` (or your preferred name)
   - **Language / Runtime**: `Python 3`
   - **Branch**: `main` (or your active branch)
   - **Region**: Closest to you (e.g. `Frankfurt`, `Oregon`, `Singapore`)
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     python run.py
     ```
   - **Plan**: `Free`

4. **Add Environment Variables** (under *Advanced / Environment Variables*):
   - `TMDB_key` = *(Your TMDB API Key)*
   - `GEMINI_API_KEY` = *(Your Gemini API Key)*
   - `LETTERBOXD_USERNAME` = `sarthi_watcher` (or your username)
   - `PYTHONUNBUFFERED` = `1`

5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build and deploy your app in ~1-2 minutes.
   - Your permanent mobile-friendly URL will be ready at: `https://mbmr-xxxx.onrender.com`!

> [!TIP]
> **Preventing Render Free-Tier Sleeping**: Render's free tier spins down after 15 minutes of inactivity. You can use a free 10-minute ping monitor like [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) pointing to `https://your-app.onrender.com/api/status` to keep your instance warm.

---

## 🔑 Obtaining Free API Keys

1. **TMDB API Key (The Movie Database)**:
   - Create a free account at [themoviedb.org](https://www.themoviedb.org/).
   - Go to **Settings → API** and generate a Developer API key.
   - Set `TMDB_key=your_key` in `.env` or during in-app onboarding.

2. **Google Gemini API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/).
   - Click **Get API Key** and create a free key.
   - Set `GEMINI_API_KEY=your_key` in `.env` or during in-app onboarding.

---

## 🔄 Personalizing with Your Own Letterboxd Account

1. Launch MBMR in your browser.
2. If visiting for the first time, the **MBMR Onboarding Wizard** will automatically guide you.
3. Enter your **Letterboxd Username** (`@your_username`).
4. Click **"1-Click Diary Sync"** to pull your watched history, or **"Sync Letterboxd Watchlist"** to import your watchlist.
5. Click **"Retrain Personal AI Model"** — MBMR will perform TF-IDF feature engineering and fit your personalized Random Forest model in seconds!

---

## 🧪 Running Automated Tests

MBMR comes with a comprehensive automated test suite covering API endpoints, recommendation filters, Gemini query expansion, title search, watchlist clustering, and diary poster hydration:

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
