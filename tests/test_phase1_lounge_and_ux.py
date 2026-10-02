#!/usr/bin/env python3
"""
Phase 1 Verification Tests
Validates: HTML structure, arcade.js engine, CSS, app.js modifications
Run: python tests/test_phase1_lounge_and_ux.py
"""

import os
import subprocess
import unittest

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')


def read_file(name):
    path = os.path.join(FRONTEND, name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class TestHTMLStructure(unittest.TestCase):
    """Verify index.html contains all new Phase 1 elements."""

    @classmethod
    def setUpClass(cls):
        cls.html = read_file('index.html')

    def test_cinema_lounge_overlay_mounted(self):
        self.assertIn('id="cinema-lounge-overlay"', self.html)

    def test_lounge_status_bar(self):
        self.assertIn('class="lounge-status-bar"', self.html)

    def test_lounge_pulse_indicator(self):
        self.assertIn('class="lounge-pulse-dot"', self.html)

    def test_lounge_countdown(self):
        self.assertIn('class="lounge-countdown"', self.html)

    def test_lounge_tabs_present(self):
        for tab in ['trivia-blitz', 'title-scramble']:
            self.assertIn("showLoungeGame('{}')".format(tab), self.html)

    def test_removed_lounge_tabs_absent(self):
        for tab in ['cinematch', 'trivia-live']:
            self.assertNotIn("showLoungeGame('{}')".format(tab), self.html)

    def test_arcade_game_container(self):
        self.assertIn('id="arcade-game-container"', self.html)

    def test_lounge_ready_toast(self):
        self.assertIn('id="lounge-ready-toast"', self.html)

    def test_arcade_js_loaded(self):
        self.assertIn('src="arcade.js', self.html)

    def test_runtime_quick_filter_discover(self):
        self.assertIn('id="quick-filter-btn"', self.html)
        self.assertIn('toggleQuickWatch()', self.html)

    def test_runtime_select_watchlist(self):
        self.assertIn('id="wl-runtime-select"', self.html)
        self.assertIn('applyWatchlistFilters()', self.html)

    def test_spotlight_director_row(self):
        self.assertIn('id="spotlight-director-row"', self.html)
        self.assertIn('id="spotlight-director-val"', self.html)

    def test_spotlight_cast_row(self):
        self.assertIn('id="spotlight-cast-row"', self.html)
        self.assertIn('id="spotlight-cast-val"', self.html)

    def test_reroll_button_in_modal(self):
        self.assertIn('id="pick-reroll-btn"', self.html)
        self.assertIn('rerollPickTonight()', self.html)

    def test_all_production_views_exist(self):
        for view in ['view-discover', 'view-watchlist', 'view-journal', 'view-taste']:
            self.assertIn('id="{}"'.format(view), self.html)

    def test_all_modals_exist(self):
        for modal in ['pick-tonight-modal', 'onboarding-modal', 'sync-modal', 'log-modal']:
            self.assertIn('id="{}"'.format(modal), self.html)

    def test_lounge_button_in_sidebar(self):
        self.assertIn('id="lounge-btn"', self.html)
        self.assertIn('showCinemaLounge()', self.html)

    def test_lounge_button_icon(self):
        self.assertIn('Cinema Lounge', self.html)

    def test_login_form_in_onboarding(self):
        self.assertIn('onboard-form-login', self.html)


class TestArcadeJS(unittest.TestCase):
    """Verify arcade.js implements CinemaArcadeEngine with Trivia Blitz and Title Scramble."""

    @classmethod
    def setUpClass(cls):
        cls.js = read_file('arcade.js')

    def test_cinema_arcade_engine_exists(self):
        self.assertIn('CinemaArcadeEngine', self.js)

    def test_show_lounge_function(self):
        self.assertIn('showLounge', self.js)

    def test_show_lounge_game_function(self):
        self.assertIn('showLoungeGame', self.js)

    def test_hide_lounge_function(self):
        self.assertIn('hideLounge', self.js)

    # Trivia Blitz (OpenTDB)
    def test_trivia_blitz_init(self):
        self.assertIn('initTriviaBlitz', self.js)

    def test_removed_games_absent(self):
        self.assertNotIn('initCineMatch', self.js)
        self.assertNotIn('initLiveTrivia', self.js)

    def test_opentdb_api_url(self):
        self.assertIn('opentdb.com/api.php', self.js)

    def test_opentdb_category_11(self):
        self.assertIn('category=11', self.js)

    def test_opentdb_session_token(self):
        self.assertIn('api_token.php', self.js)
        self.assertIn('opentdb_session', self.js)
        # Token must be sent as the `token` parameter for OpenTDB to avoid repeats
        self.assertIn('&token=', self.js)

    def test_seen_questions_tracked(self):
        self.assertIn('mbmr_trivia_seen', self.js)
        self.assertIn('markSeen', self.js)

    def test_html_entity_decoding(self):
        self.assertIn('decodeHtml', self.js)
        self.assertIn('DOMParser', self.js)

    def test_option_shuffling(self):
        self.assertIn('shuffle', self.js)

    def test_offline_fallback(self):
        self.assertIn('fallback', self.js.lower())
        self.assertIn('triviaBank', self.js)

    def test_trivia_streak_tracking(self):
        self.assertIn('score', self.js)

    def test_trivia_timer(self):
        self.assertIn('startCountdown', self.js)
        self.assertIn('stopAll', self.js)

    # Title Scramble
    def test_title_scramble_init(self):
        self.assertIn('initScrambleGame', self.js)

    def test_scramble_bank_exists(self):
        self.assertIn('scrambleBank', self.js)

    def test_scramble_clues(self):
        self.assertIn('clue', self.js)
        self.assertNotIn('Director:', self.js)

    def test_response_code_check(self):
        self.assertIn('response_code', self.js)

    def test_js_syntax_valid(self):
        path = os.path.join(FRONTEND, 'arcade.js')
        result = subprocess.run(
            ['node', '--check', path],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, "Syntax error:\n{}".format(result.stderr))


class TestAppJSFeatures(unittest.TestCase):
    """Verify app.js Phase 1 modifications."""

    @classmethod
    def setUpClass(cls):
        cls.js = read_file('app.js')

    # Auto-wake poller
    def test_session_picked_movie_ids(self):
        self.assertIn('sessionPickedMovieIds', self.js)

    def test_reroll_pick_tonight_function(self):
        self.assertIn('async function rerollPickTonight', self.js)
        self.assertIn('sessionPickedMovieIds.add', self.js)
        self.assertIn('executePickTonight', self.js)

    def test_excluded_ids_in_pick_tonight(self):
        self.assertIn('excluded_ids', self.js)

    def test_toggle_quick_watch(self):
        self.assertIn('function toggleQuickWatch', self.js)
        self.assertIn('quickWatchActive', self.js)

    def test_apply_watchlist_filters(self):
        self.assertIn('function applyWatchlistFilters', self.js)
        self.assertIn('wl-runtime-select', self.js)

    def test_runtime_max_in_recommendations(self):
        self.assertIn('runtime_max', self.js)
        self.assertIn('quickWatchActive ? 90 : null', self.js)

    def test_runtime_filter_in_watchlist_url(self):
        self.assertIn('runtime_max=90', self.js)

    def test_spotlight_director_fetch(self):
        self.assertIn('spotlight-director-val', self.js)
        self.assertIn('movie_credits', self.js)

    def test_spotlight_cast_fetch(self):
        self.assertIn('spotlight-cast-val', self.js)

    def test_lounge_overlay_integration(self):
        self.assertIn('showCinemaLounge', self.js)
        self.assertIn('hideCinemaLounge', self.js)

    def test_ready_toast_integration(self):
        self.assertIn('showLoungeReadyToast', self.js)

    def test_hide_cinema_lounge_function(self):
        self.assertIn('function hideCinemaLounge', self.js)

    def test_show_cinema_lounge_function(self):
        self.assertIn('function showCinemaLounge', self.js)

    def test_switch_view_hides_lounge(self):
        self.assertIn('hideCinemaLounge()', self.js)

    def test_watchlist_page_size_49(self):
        self.assertIn('WATCHLIST_PAGE_SIZE = 49', self.js)

    def test_watchlist_total_pages_ceil(self):
        self.assertIn('Math.ceil(totalItems / WATCHLIST_PAGE_SIZE)', self.js)

    def test_js_syntax_valid(self):
        path = os.path.join(FRONTEND, 'app.js')
        result = subprocess.run(
            ['node', '--check', path],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, "Syntax error:\n{}".format(result.stderr))


class TestCSSFeatures(unittest.TestCase):
    """Verify CSS includes lounge styles."""

    @classmethod
    def setUpClass(cls):
        raw = open(os.path.join(FRONTEND, 'styles.css'), 'rb').read()
        cls.css = raw.decode('utf-8')
        cls.raw = raw

    def test_no_bom(self):
        self.assertNotEqual(self.raw[:3], b'\xef\xbb\xbf')

    def test_no_mojibake(self):
        mojibake_patterns = ['\u00D4\u00C7\u00F6', '\u00D4\u00F6\u00C7']
        for pat in mojibake_patterns:
            self.assertNotIn(pat, self.css)

    def test_braces_balanced(self):
        self.assertEqual(self.css.count('{'), self.css.count('}'))

    def test_lounge_overlay_css(self):
        self.assertIn('cinema-lounge-overlay', self.css)

    def test_lounge_pulse_css(self):
        self.assertIn('loungePulse', self.css)
        self.assertIn('lounge-pulse-dot', self.css)

    def test_lounge_tabs_css(self):
        self.assertIn('lounge-tab', self.css)

    def test_trivia_css(self):
        self.assertIn('trivia-opt-btn', self.css)

    def test_correct_pulse_css(self):
        self.assertIn('correctPulse', self.css)

    def test_spotlight_meta_row_css(self):
        self.assertIn('sp-meta-row', self.css)

    def test_cast_badge_css(self):
        self.assertIn('sp-cast-badge', self.css)

    def test_reroll_dice_css(self):
        self.assertIn('pick-reroll-btn', self.css)
        self.assertIn('diceRoll', self.css)

    def test_ready_toast_css(self):
        self.assertIn('lounge-ready-toast', self.css)

    def test_mobile_responsive(self):
        self.assertIn('@media (max-width: 768px)', self.css)


class TestBackendCompatibility(unittest.TestCase):
    """Verify backend endpoints referenced by frontend exist."""

    @classmethod
    def setUpClass(cls):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.app_path = os.path.join(base, 'backend', 'api.py')
        cls.rec_path = os.path.join(base, 'backend', 'recommender.py')
        with open(cls.app_path, 'r', encoding='utf-8') as f:
            cls.api_py = f.read()
        with open(cls.rec_path, 'r', encoding='utf-8') as f:
            cls.rec_py = f.read()

    def test_movie_credits_endpoint(self):
        self.assertIn('movie_credits', self.api_py)

    def test_pick_tonight_excluded_ids(self):
        self.assertIn('excluded_ids', self.api_py)


if __name__ == '__main__':
    unittest.main(verbosity=2)
