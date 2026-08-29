<div align="center">
  <img src="frontend/assets/logo.svg" alt="MBMR Logo" width="100" height="100" />
  <h1>MBMR — Mood-Based Movie Recommender</h1>
  <p><em>Your Personalized AI Cinema Intelligence & Letterboxd Companion</em></p>
</div>

[![Live App](https://img.shields.io/badge/Live%20App-mbmr.onrender.com-success?style=for-the-badge&logo=render)](https://mbmr.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> **🌐 Try the Live Web App**: **[https://mbm-recommender-nine.vercel.app/](https://mbm-recommender-nine.vercel.app/)**

**MBMR (Mood-Based Movie Recommender)** is a personalized AI-powered cinema recommendation engine and modern Letterboxd companion. It learns your unique film taste using a machine learning model trained on your Letterboxd diary, interprets natural language vibes and moods with Google Gemini AI, clusters your watchlist, and predicts how much you will love any movie.

---

## ✨ Features

- **🧠 Two-Stage Hybrid AI Recommendation Engine**: Combines **Google Gemini AI** semantic vibe discovery with your **Local Random Forest** taste model. Gemini interprets nuanced natural language vibes grounded by your Letterboxd taste anchors (top directors, 5★ favorites, high-affinity genres), purges all movies you've already seen, and ranks candidates from highest to lowest predicted likeness.
- **🎭 Taste-Aware Mood & Vibe Search**: Search for movie titles or natural language vibes (*"gritty 90s cyber thriller with neon aesthetics"*, *"melancholic coming-of-age on a rainy night"*). Gemini generates curated candidate films with 1-sentence **AI Vibe Pitches** explaining why each film fits your mood.
- **🚫 100% Unseen Discovery Guarantee**: Automated deduplication against your synced Letterboxd diary and ratings ensures you never waste time seeing recommendations for films you already logged.
- **🎲 "Pick For Me Tonight" AI Matchmaker**: Tell MBMR your available time, current mood, and streaming platform. The AI picks the single best movie from your watchlist with a personalized, witty matchmaker pitch (*"Since you loved Drive and have 90 mins..."*).
- **🔖 Intelligent Watchlist with Mood Clusters**: Organize your Letterboxd watchlist into smart clusters (*🛋️ Comfort, 🧠 Mind-Bending, 🍿 Popcorn & Adrenaline, ⏳ Quick <105m*) with streaming platform filters.
- **📖 Visual Film Journal (Diary)**: Browse your entire watch history with high-resolution TMDB posters, dual view modes (**List View ≡** and **Poster Grid ⊞**), and rating filters.
- **🎬 Cinema Spotlight Drawer**: Click any film to inspect its 4K backdrop, synopsis, streaming providers (Netflix, Prime, Apple TV, etc.), AI Vibe Match reason, and **Post-Watch Ripple Recommendations**.
- **⚡ 1-Click Letterboxd Sync**: Sync your public Letterboxd watchlist (all of it) and your 50 most recent diary entries with one click, or retrain your AI model anytime directly from the UI.
- **📥 Full History via CSV Import**: Export your complete watch history from Letterboxd (`ratings.csv`, `diary.csv`, `watchlist.csv`) and drop it onto MBMR for automated ingestion, TMDB poster hydration, and model recalibration.
- **📱 Fully Responsive Mobile App & PWA**: Seamless experience across mobile phones, tablets, and desktops with bottom navigation dock, safe-area padding, and IndexedDB local client storage.

---

## 🌐 Live Web App

MBMR is deployed and ready to use at:
👉 **[https://mbm-recommender-nine.vercel.app/](https://mbm-recommender-nine.vercel.app/)**

When you visit for the first time:
1. The **MBMR Onboarding Wizard** will prompt you to enter your **Letterboxd Username** (`@handle`) and a 4-6 digit PIN.
2. Enter your free **TMDB API Key** (required for posters, metadata, and cast).
3. Optionally enter your free **Google Gemini API Key** for deep semantic mood reasoning, bespoke 1-sentence vibe pitches, and dynamic AI watchlist ranking (or leave blank to use the **In-Built Machine Learning Model Fallback**).
4. All credentials and preferences are **stored 100% locally in your browser (IndexedDB)** — your data remains private and secure.

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
