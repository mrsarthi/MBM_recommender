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

class TestDiaryFilters(unittest.TestCase):
    def setUp(self):
        self.profile_path = 'user_data/user_profile.csv'
        self.assertTrue(os.path.exists(self.profile_path))
        self.df = pd.read_csv(self.profile_path)

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
