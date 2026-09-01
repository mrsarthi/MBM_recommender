import os
import sys
import unittest
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

def apply_diary_filters_and_sort(df, year_filter="All Time", rating_filter="All Ratings", sort_mode="Newest Log First", search_query=""):
    if df.empty:
        return df

    filtered_df = df.copy()

    # 1. Standardize and clean columns
    filtered_df['Rating_num'] = pd.to_numeric(filtered_df['Rating'], errors='coerce').fillna(0.0)
    filtered_df['Year_num'] = pd.to_numeric(filtered_df['Year'], errors='coerce')
    filtered_df['dt_parsed'] = pd.to_datetime(filtered_df['Date'], errors='coerce')

    # 2. Year Filter (matches watched year from Date OR release Year)
    if year_filter != "All Time":
        if year_filter == "Earlier":
            # Watched before 2022 or released before 2022
            mask = (filtered_df['dt_parsed'].dt.year < 2022) | (filtered_df['Year_num'] < 2022)
            filtered_df = filtered_df[mask]
        else:
            try:
                y_target = int(year_filter)
                # match Date year or release Year
                mask = (filtered_df['dt_parsed'].dt.year == y_target) | (filtered_df['Year_num'] == y_target)
                filtered_df = filtered_df[mask]
            except ValueError:
                pass

    # 3. Rating Filter
    if rating_filter == "5★ Only":
        filtered_df = filtered_df[filtered_df['Rating_num'] >= 4.9]
    elif rating_filter == "4★ & Above":
        filtered_df = filtered_df[filtered_df['Rating_num'] >= 3.9]
    elif rating_filter == "3★ - 3.5★":
        filtered_df = filtered_df[(filtered_df['Rating_num'] >= 2.9) & (filtered_df['Rating_num'] <= 3.6)]
    elif rating_filter == "< 3★":
        filtered_df = filtered_df[filtered_df['Rating_num'] < 2.9]

    # 4. Search Filter (matches title, director, genres, cast)
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = filtered_df['Name'].astype(str).str.lower().str.contains(q, regex=False)
        if 'director' in filtered_df.columns:
            mask |= filtered_df['director'].astype(str).str.lower().str.contains(q, regex=False)
        if 'genres' in filtered_df.columns:
            mask |= filtered_df['genres'].astype(str).str.lower().str.contains(q, regex=False)
        if 'cast' in filtered_df.columns:
            mask |= filtered_df['cast'].astype(str).str.lower().str.contains(q, regex=False)
        filtered_df = filtered_df[mask]

    # 5. Sorting Engine
    if sort_mode == "Newest Log First":
        filtered_df = filtered_df.sort_values(by=['dt_parsed', 'Year_num'], ascending=[False, False], na_position='last')
    elif sort_mode == "Oldest Log First":
        filtered_df = filtered_df.sort_values(by=['dt_parsed', 'Year_num'], ascending=[True, True], na_position='last')
    elif sort_mode == "Highest Rating":
        filtered_df = filtered_df.sort_values(by=['Rating_num', 'dt_parsed'], ascending=[False, False], na_position='last')
    elif sort_mode == "Lowest Rating":
        filtered_df = filtered_df.sort_values(by=['Rating_num', 'dt_parsed'], ascending=[True, False], na_position='last')
    elif sort_mode == "Title (A-Z)":
        filtered_df = filtered_df.sort_values(by='Name', key=lambda col: col.str.lower(), ascending=True)
    elif sort_mode == "Release Year":
        filtered_df = filtered_df.sort_values(by=['Year_num', 'dt_parsed'], ascending=[False, False], na_position='last')

    return filtered_df

SAMPLE_DIARY_RECORDS = [
    {"Name": "Pinocchio: Unstrung", "Date": "2026-08-23", "Year": 2024, "Rating": 5.0, "director": "Guillermo del Toro", "genres": "Fantasy|Adventure", "cast": "Benjamin Alfonzo"},
    {"Name": "Weapons", "Date": "2026-08-21", "Year": 2024, "Rating": 4.5, "director": "Zach Cregger", "genres": "Action|Horror", "cast": "Josh Brolin"},
    {"Name": "Spider-Man: Across the Spider-Verse", "Date": "2026-08-20", "Year": 2023, "Rating": 5.0, "director": "Phil Lord", "genres": "Animation|Action", "cast": "Shameik Moore"},
    {"Name": "A Tale of Two Sisters", "Date": "2026-08-19", "Year": 2003, "Rating": 4.0, "director": "Kim Jee-woon", "genres": "Drama|Horror|Mystery", "cast": "Lim Soo-jung"},
    {"Name": "Blade Runner 2049", "Date": "2026-08-18", "Year": 2017, "Rating": 4.5, "director": "Denis Villeneuve", "genres": "Science Fiction|Mystery", "cast": "Ryan Gosling"},
    {"Name": "Chinatown", "Date": "2026-08-17", "Year": 1974, "Rating": 4.5, "director": "Roman Polanski", "genres": "Crime|Drama|Mystery", "cast": "Jack Nicholson"},
    {"Name": "Dune: Part Two", "Date": "2026-08-16", "Year": 2024, "Rating": 4.5, "director": "Denis Villeneuve", "genres": "Science Fiction|Adventure", "cast": "Timothée Chalamet"},
    {"Name": "Everything Everywhere All at Once", "Date": "2026-08-15", "Year": 2022, "Rating": 4.0, "director": "Daniel Kwan", "genres": "Action|Adventure|Science Fiction", "cast": "Michelle Yeoh"},
    {"Name": "Fight Club", "Date": "2026-08-14", "Year": 1999, "Rating": 4.5, "director": "David Fincher", "genres": "Drama", "cast": "Brad Pitt"},
    {"Name": "Get Out", "Date": "2026-08-13", "Year": 2017, "Rating": 4.0, "director": "Jordan Peele", "genres": "Mystery|Thriller|Horror", "cast": "Daniel Kaluuya"},
    {"Name": "Hereditary", "Date": "2026-08-12", "Year": 2018, "Rating": 4.0, "director": "Ari Aster", "genres": "Horror|Mystery", "cast": "Toni Collette"},
    {"Name": "Inception", "Date": "2026-08-11", "Year": 2010, "Rating": 5.0, "director": "Christopher Nolan", "genres": "Action|Science Fiction|Adventure", "cast": "Leonardo DiCaprio"},
    {"Name": "Interstellar", "Date": "2026-08-10", "Year": 2014, "Rating": 4.5, "director": "Christopher Nolan", "genres": "Adventure|Drama|Science Fiction", "cast": "Matthew McConaughey"},
    {"Name": "Joker", "Date": "2026-08-09", "Year": 2019, "Rating": 3.5, "director": "Todd Phillips", "genres": "Crime|Thriller|Drama", "cast": "Joaquin Phoenix"},
    {"Name": "Kill Bill: Vol. 1", "Date": "2026-08-08", "Year": 2003, "Rating": 4.0, "director": "Quentin Tarantino", "genres": "Action|Crime", "cast": "Uma Thurman"},
    {"Name": "La La Land", "Date": "2026-08-07", "Year": 2016, "Rating": 4.5, "director": "Damien Chazelle", "genres": "Comedy|Drama|Music|Romance", "cast": "Ryan Gosling"},
    {"Name": "Memento", "Date": "2026-08-06", "Year": 2000, "Rating": 4.5, "director": "Christopher Nolan", "genres": "Mystery|Thriller", "cast": "Guy Pearce"},
    {"Name": "Nightcrawler", "Date": "2026-08-05", "Year": 2014, "Rating": 4.0, "director": "Dan Gilroy", "genres": "Crime|Drama|Thriller", "cast": "Jake Gyllenhaal"},
    {"Name": "Oppenheimer", "Date": "2026-08-04", "Year": 2023, "Rating": 4.5, "director": "Christopher Nolan", "genres": "Drama|History", "cast": "Cillian Murphy"},
    {"Name": "Parasite", "Date": "2026-08-03", "Year": 2019, "Rating": 5.0, "director": "Bong Joon-ho", "genres": "Comedy|Thriller|Drama", "cast": "Song Kang-ho"},
    {"Name": "Spirited Away", "Date": "2026-08-02", "Year": 2001, "Rating": 5.0, "director": "Hayao Miyazaki", "genres": "Animation|Family|Fantasy", "cast": "Rumi Hiiragi"},
    {"Name": "The Batman", "Date": "2026-08-01", "Year": 2022, "Rating": 4.0, "director": "Matt Reeves", "genres": "Crime|Mystery|Thriller", "cast": "Robert Pattinson"},
    {"Name": "The Matrix", "Date": "2026-07-31", "Year": 1999, "Rating": 5.0, "director": "Lana Wachowski", "genres": "Action|Science Fiction", "cast": "Keanu Reeves"},
    {"Name": "The Room", "Date": "2026-07-30", "Year": 2003, "Rating": 1.0, "director": "Tommy Wiseau", "genres": "Drama|Romance", "cast": "Tommy Wiseau"},
    {"Name": "Troll 2", "Date": "2026-07-29", "Year": 1990, "Rating": 0.5, "director": "Claudio Fragasso", "genres": "Fantasy|Horror", "cast": "Michael Stephenson"},
    {"Name": "Zoolander 2", "Date": "2026-07-28", "Year": 2016, "Rating": 1.5, "director": "Ben Stiller", "genres": "Comedy", "cast": "Ben Stiller"}
]

class TestDiaryFilters(unittest.TestCase):
    def setUp(self):
        self.profile_path = 'user_data/user_profile.csv'
        if os.path.exists(self.profile_path):
            self.df = pd.read_csv(self.profile_path)
        else:
            os.makedirs('user_data', exist_ok=True)
            self.df = pd.DataFrame(SAMPLE_DIARY_RECORDS)
            self.df.to_csv(self.profile_path, index=False)

    def test_sort_newest_first(self):
        sorted_df = apply_diary_filters_and_sort(self.df, sort_mode="Newest Log First")
        first_row = sorted_df.iloc[0]
        # Should be Pinocchio: Unstrung on 2026-08-23 or Weapons on 2026-08-21
        self.assertEqual(first_row['Date'], '2026-08-23')
        self.assertEqual(first_row['Name'], 'Pinocchio: Unstrung')
        print(f"  -> PASSED: Newest Log First placed newest watch at top: '{first_row['Name']}' ({first_row['Date']})")

    def test_sort_oldest_first(self):
        sorted_df = apply_diary_filters_and_sort(self.df, sort_mode="Oldest Log First")
        # Find first valid date
        valid_dates = sorted_df['Date'].dropna().astype(str)
        if len(valid_dates) >= 2:
            self.assertLessEqual(valid_dates.iloc[0], valid_dates.iloc[-1])
        print(f"  -> PASSED: Oldest Log First placed oldest watch at top.")

    def test_sort_highest_rating(self):
        sorted_df = apply_diary_filters_and_sort(self.df, sort_mode="Highest Rating")
        self.assertEqual(sorted_df.iloc[0]['Rating'], 5.0)
        self.assertLessEqual(sorted_df.iloc[-1]['Rating'], 2.0)
        print(f"  -> PASSED: Highest Rating correctly ordered 5.0★ down to {sorted_df.iloc[-1]['Rating']}★")

    def test_sort_lowest_rating(self):
        sorted_df = apply_diary_filters_and_sort(self.df, sort_mode="Lowest Rating")
        self.assertLessEqual(sorted_df.iloc[0]['Rating'], 2.0)
        self.assertEqual(sorted_df.iloc[-1]['Rating'], 5.0)
        print(f"  -> PASSED: Lowest Rating correctly ordered {sorted_df.iloc[0]['Rating']}★ up to 5.0★")

    def test_sort_title_az(self):
        sorted_df = apply_diary_filters_and_sort(self.df, sort_mode="Title (A-Z)")
        titles = sorted_df['Name'].tolist()
        self.assertTrue(titles[0].lower() <= titles[10].lower() <= titles[-1].lower())
        print(f"  -> PASSED: Title A-Z correctly sorted ('{titles[0]}' -> '{titles[-1]}')")

    def test_filter_year_2026(self):
        filtered = apply_diary_filters_and_sort(self.df, year_filter="2026")
        self.assertGreater(len(filtered), 0)
        for _, r in filtered.iterrows():
            matches = str(r['Date']).startswith('2026') or str(r['Year']).startswith('2026')
            self.assertTrue(matches)
        print(f"  -> PASSED: 2026 Filter returned {len(filtered)} films matching 2026")

    def test_filter_rating_5_stars(self):
        filtered = apply_diary_filters_and_sort(self.df, rating_filter="5★ Only")
        self.assertGreater(len(filtered), 0)
        for _, r in filtered.iterrows():
            self.assertEqual(r['Rating'], 5.0)
        print(f"  -> PASSED: 5★ Only filter returned {len(filtered)} films")

    def test_search_filter(self):
        filtered = apply_diary_filters_and_sort(self.df, search_query="Spider")
        self.assertGreater(len(filtered), 0)
        for _, r in filtered.iterrows():
            in_name = 'spider' in str(r['Name']).lower()
            in_dir = 'spider' in str(r.get('director', '')).lower()
            in_gen = 'spider' in str(r.get('genres', '')).lower()
            self.assertTrue(in_name or in_dir or in_gen)
        print(f"  -> PASSED: Search query 'Spider' returned {len(filtered)} matching films")

if __name__ == '__main__':
    unittest.main(verbosity=2)
