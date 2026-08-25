import os
import re
import html
import subprocess
import requests
import json
import xml.etree.ElementTree as ET
import pandas as pd
from backend.config import PROFILE_PATH, get_base_dir
from backend.feature_engineering import feature_engineering
from backend.model_train import train_personal_model

def sync_rss(username="sarthi_watcher", profile_path=PROFILE_PATH):
    """
    Syncs the latest activity from Letterboxd RSS feed.
    """
    clean_user = username.strip().lstrip('@')
    rss_url = f"https://letterboxd.com/{clean_user}/rss/"
    print(f"Fetching RSS from {rss_url}...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(rss_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return False, f"HTTP Error {resp.status_code}"
            
        root = ET.fromstring(resp.content)
        ns = {
            'letterboxd': 'https://letterboxd.com',
            'tmdb': 'https://www.themoviedb.org',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }
        
        rss_movies = []
        for item in root.findall('./channel/item'):
            title_elem = item.find('letterboxd:filmTitle', ns)
            year_elem = item.find('letterboxd:filmYear', ns)
            rating_elem = item.find('letterboxd:memberRating', ns)
            date_elem = item.find('letterboxd:watchedDate', ns)
            tmdb_elem = item.find('tmdb:movieId', ns)
            link_elem = item.find('link')
            
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
            if not title:
                raw_t = item.find('title')
                if raw_t is not None and raw_t.text:
                    m = re.match(r'^(.*?),\s*(\d{4})?\s*-\s*([★½]+)?', raw_t.text)
                    if m: title = m.group(1).strip()
            if not title: continue
            
            year = year_elem.text.strip() if year_elem is not None and year_elem.text else ''
            rating = float(rating_elem.text.strip()) if rating_elem is not None and rating_elem.text else None
            date = date_elem.text.strip() if date_elem is not None and date_elem.text else ''
            tmdb_id = tmdb_elem.text.strip() if tmdb_elem is not None and tmdb_elem.text else None
            link = link_elem.text.strip() if link_elem is not None and link_elem.text else ''
            
            rss_movies.append({
                'Date': date, 'Name': title, 'Year': year,
                'Letterboxd URI': link, 'Rating': rating, 'movie_id': tmdb_id
            })
            
        if not rss_movies:
            return True, "No new activity found in RSS."
            
        return merge_records_into_profile(rss_movies, profile_path)
    except Exception as e:
        return False, str(e)

def merge_records_into_profile(records, profile_path=PROFILE_PATH):
    """
    Merges list of dict movie records into user_profile.csv and triggers auto-retraining.
    """
    if os.path.exists(profile_path):
        df = pd.read_csv(profile_path)
        df.columns = [c.strip() for c in df.columns]
    else:
        df = pd.DataFrame()
        
    def norm(t):
        clean = re.sub(r'^Poster for\s+', '', str(t), flags=re.IGNORECASE).strip()
        clean = re.sub(r'\s*\(\d{4}\)$', '', clean).strip()
        return re.sub(r'[^a-z0-9]', '', clean.lower())
        
    existing_map = {}
    for idx, r in df.iterrows():
        k = norm(r['Name'])
        if k: existing_map[k] = idx
        
    new_rows = []
    updated_count = 0
    
    for item in records:
        title = item.get('Name', '').strip()
        k = norm(title)
        if not k: continue
        
        rating = item.get('Rating')
        date = item.get('Date', '')
        year = item.get('Year', '')
        uri = item.get('Letterboxd URI', '')
        tmdb_id = item.get('movie_id')
        
        if k in existing_map:
            idx = existing_map[k]
            if rating is not None and pd.notna(rating):
                df.at[idx, 'Rating'] = float(rating)
                updated_count += 1
            if date and pd.isna(df.at[idx, 'Date']):
                df.at[idx, 'Date'] = date
            if tmdb_id and pd.isna(df.at[idx, 'movie_id']):
                df.at[idx, 'movie_id'] = tmdb_id
        else:
            new_rows.append({
                'Date': date, 'Name': title, 'Year': year,
                'Letterboxd URI': uri, 'Rating': rating or 3.5,
                'movie_id': tmdb_id, 'genres': '', 'overview': '',
                'director': '', 'cast': '', 'keywords': ''
            })
            existing_map[k] = len(df) + len(new_rows) - 1
            
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
    # Sort dates
    df['dt_sort'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['dt_sort'], ascending=[False], na_position='last').reset_index(drop=True)
    df = df.drop(columns=['dt_sort'])
    
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    df.to_csv(profile_path, index=False)
    
    # Trigger feature engineering and model training
    try:
        feature_engineering(input_file=profile_path)
        train_personal_model()
    except Exception as e:
        print(f"Retraining error: {e}")
        
    msg = f"Synced {len(records)} films ({len(new_rows)} new, {updated_count} updated). Total profile: {len(df)} films."
    return True, msg
