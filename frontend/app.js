// ================================================================
// MBMR — RAYCAST-STYLE CONTROLLER WITH WATCHLIST & MATCHMAKER
// ================================================================

const API = window.location.origin;
const IMG500 = 'https://image.tmdb.org/t/p/w500';
const IMG1280 = 'https://image.tmdb.org/t/p/w1280';
const IMG200 = 'https://image.tmdb.org/t/p/w200';

let currentPicks = [];
let currentSpotlight = null;
let selectedLogMovie = null;
let watchlistIds = new Set();
let currentMatchmakerWinner = null;
let currentWatchlistCluster = 'All';
let discoverMode = 'search';
let currentView = 'watchlist';
let renderedMoviesMap = new Map();

// ── In-Memory Fast Caches (Instant 0ms Tab Switching) ──
const watchlistCache = new Map();
const diaryCache = new Map();

// ── IndexedDB Storage for Private Credentials & Offline State ──
const MBMRStorage = {
    dbName: 'mbmr_local_db',
    dbVersion: 1,
    db: null,

    async init() {
        return new Promise((resolve) => {
            if (!window.indexedDB) { resolve(null); return; }
            const req = indexedDB.open(this.dbName, this.dbVersion);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings', { keyPath: 'key' });
                }
            };
            req.onsuccess = (e) => {
                this.db = e.target.result;
                resolve(this.db);
            };
            req.onerror = () => resolve(null);
        });
    },

    async get(key) {
        if (!this.db) await this.init();
        if (!this.db) return localStorage.getItem(key);
        return new Promise((resolve) => {
            try {
                const tx = this.db.transaction('settings', 'readonly');
                const store = tx.objectStore('settings');
                const req = store.get(key);
                req.onsuccess = () => resolve(req.result ? req.result.value : localStorage.getItem(key));
                req.onerror = () => resolve(localStorage.getItem(key));
            } catch { resolve(localStorage.getItem(key)); }
        });
    },

    async set(key, value) {
        try { localStorage.setItem(key, value); } catch(e) {}
        if (!this.db) await this.init();
        if (!this.db) return;
        return new Promise((resolve) => {
            try {
                const tx = this.db.transaction('settings', 'readwrite');
                const store = tx.objectStore('settings');
                store.put({ key, value });
                tx.oncomplete = () => resolve();
                tx.onerror = () => resolve();
            } catch { resolve(); }
        });
    }
};

const _nativeFetch = window.fetch.bind(window);
async function mbmrFetch(url, options = {}) {
    const tmdbKey = await MBMRStorage.get('tmdb_key');
    const geminiKey = await MBMRStorage.get('gemini_key');
    const username = await MBMRStorage.get('letterboxd_username');

    options.headers = options.headers || {};
    if (tmdbKey) options.headers['X-TMDB-Key'] = tmdbKey;
    if (geminiKey) options.headers['X-Gemini-Key'] = geminiKey;
    if (username) options.headers['X-Letterboxd-User'] = username;

    return _nativeFetch(url, options);
}

window.fetch = function(url, options = {}) {
    if (typeof url === 'string' && (url.startsWith('/api') || url.includes('/api/'))) {
        return mbmrFetch(url, options);
    }
    return _nativeFetch(url, options);
};

// Instant Hydration from persistent cache (0.0ms initial screen render)
function hydrateFromStorage() {
    try {
        const cachedWl = localStorage.getItem('mbmr_cached_watchlist');
        if (cachedWl) {
            const data = JSON.parse(cachedWl);
            if (data && data.watchlist && data.watchlist.length > 0) {
                renderWatchlistGrid(data.watchlist);
                updateWatchlistBadge(data.total !== undefined ? data.total : data.watchlist.length);
                const avg = (data.watchlist.reduce((acc, m) => acc + (m.ai_score || 3.8), 0) / data.watchlist.length).toFixed(1);
                const avgEl = document.getElementById('wl-avg-score');
                if (avgEl) avgEl.textContent = `${avg}★`;
            }
        }
        const cachedDiary = localStorage.getItem('mbmr_cached_diary');
        if (cachedDiary) {
            const d = JSON.parse(cachedDiary);
            if (d && d.films && d.films.length > 0) {
                currentDiaryFilms = d.films;
                renderJournal(d.films);
                if (d.total) {
                    const jCount = document.getElementById('journal-total-count');
                    const nCount = document.getElementById('nav-count');
                    if (jCount) jCount.textContent = d.total;
                    if (nCount) nCount.textContent = d.total;
                }
            }
        }
    } catch(e) {}
}

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    hydrateFromStorage();
    await MBMRStorage.init();
    checkOnboarding();
    loadStatus();
    loadWatchlistIds();
    switchView('watchlist');
    const promptInput = document.getElementById('prompt-input');
    if (promptInput) {
        if (!promptInput.value.trim()) {
            promptInput.value = 'Mind-Bending';
        }
        promptInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') generateRecommendations();
        });
        promptInput.addEventListener('input', () => {
            document.querySelectorAll('.vibe').forEach(v => v.classList.remove('active'));
        });
    }
});

// ── Onboarding Controller ──
async function checkOnboarding() {
    const onboarded = await MBMRStorage.get('mbmr_onboarded');
    if (!onboarded) {
        setTimeout(() => {
            const modal = document.getElementById('onboarding-modal');
            if (modal) modal.style.display = 'flex';
        }, 600);
    }
}

async function loadStatus() {
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    let wakeTimer = null;
    if (!isLocal) {
        wakeTimer = setTimeout(() => {
            const b = document.getElementById('render-wake-banner');
            if (b) b.style.display = 'flex';
        }, 6000);
    }

    try {
        const d = await (await mbmrFetch(`${API}/api/status`)).json();
        if (wakeTimer) clearTimeout(wakeTimer);
        const b = document.getElementById('render-wake-banner');
        if (b) b.style.display = 'none';

        document.getElementById('nav-count').textContent = d.total_films || 0;
        document.getElementById('journal-total-count').textContent = d.total_films || 0;
        document.getElementById('journal-avg-rating').textContent = d.avg_rating ? `${d.avg_rating}★` : '—';
        
        const localUser = await MBMRStorage.get('letterboxd_username');
        const activeUser = localUser || d.username || 'guest';
        document.getElementById('profile-user').textContent = `@${activeUser}`;
        const userInput = document.getElementById('sync-user-input');
        if (userInput) userInput.value = activeUser === 'guest' ? '' : activeUser;

        updateWatchlistBadge(d.watchlist_count || 0);
    } catch(e) {
        console.warn('Status fetch failed', e);
    }
}

async function loadWatchlistIds() {
    try {
        const d = await (await mbmrFetch(`${API}/api/watchlist`)).json();
        watchlistIds = new Set((d.watchlist || []).map(m => m.id || m.movie_id));
        updateWatchlistBadge(watchlistIds.size);
    } catch(e) {}
}

function updateWatchlistBadge(count) {
    const badge = document.getElementById('rail-watchlist-badge');
    const num = document.getElementById('wl-count-num');
    if (badge) {
        badge.textContent = count;
        badge.style.display = count > 0 ? 'inline-block' : 'none';
    }
    if (num) num.textContent = count;
}

// ── View Switching ──
function switchView(name) {
    currentView = name;
    document.querySelectorAll('.rail-icon').forEach(b => b.classList.toggle('active', b.dataset.view === name));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const v = document.getElementById(`view-${name}`);
    if (v) v.classList.add('active');
    
    const journalCount = document.getElementById('journal-total-count')?.textContent || document.getElementById('nav-count')?.textContent || '0';
    const labels = { 
        discover: 'Discover New Movies', 
        watchlist: 'Your Curated Watchlist',
        journal: `Your Film Journal (${journalCount} Lifetime Films)`, 
        taste: 'Taste Radar & Achievements' 
    };
    document.getElementById('dock-label').textContent = labels[name] || '';
    
    if (name === 'discover') setDiscoverMode(discoverMode);
    if (name === 'watchlist') fetchWatchlist();
    if (name === 'journal') fetchDiary();
    if (name === 'taste') loadTasteRadar();
}

// ── Recommendations (Discover) ──
function renderSkeletonGrid(count = 8) {
    const grid = document.getElementById('films-grid');
    grid.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const card = document.createElement('div');
        card.className = 'poster-card skeleton';
        card.innerHTML = `
            <div class="skeleton-badge"></div>
            <div class="skeleton-info">
                <div class="skeleton-title"></div>
                <div class="skeleton-sub"></div>
            </div>
        `;
        grid.appendChild(card);
    }
}

async function generateRecommendations() {
    const prompt = document.getElementById('prompt-input').value.trim();
    if (!prompt) return;

    const btn = document.getElementById('generate-btn');
    const dock = document.getElementById('dock-label');
    
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>`;
    renderSkeletonGrid(8);
    
    const displayPrompt = prompt.length > 28 ? prompt.slice(0, 28) + '...' : prompt;
    if (dock) {
        dock.innerHTML = `<span class="dock-loading"><span class="dock-spinner"></span> Curating matches for "${displayPrompt}"...</span>`;
    }

    try {
        const sourceSelect = document.getElementById('recommend-source-select');
        const sourceVal = sourceSelect ? sourceSelect.value : 'all';
        const res = await fetch(`${API}/api/recommend`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                context: document.getElementById('context-select').value,
                streaming: document.getElementById('stream-select').value,
                source: sourceVal
            })
        });
        const data = await res.json();
        currentPicks = data.candidates || [];
        renderGrid(currentPicks);
    } catch(e) { 
        console.error('Rec error', e); 
        const grid = document.getElementById('films-grid');
        grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">Failed to load recommendations. Please try again.</div>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7"/><line x1="15" y1="15" x2="19" y2="19"/></svg>`;
        if (dock) dock.textContent = 'Discover New Movies';
    }
}

function setVibe(el, text) {
    document.querySelectorAll('.vibe').forEach(v => v.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('prompt-input').value = text;
    generateRecommendations();
}

function openLogForMovieFromCard(m) {
    selectMovieForLog(m);
    openLogModal();
}

function setDiscoverMode(mode) {
    discoverMode = mode;
    const searchBtn = document.getElementById('discover-mode-search-btn');
    const recommendBtn = document.getElementById('discover-mode-recommend-btn');
    if (searchBtn) searchBtn.classList.toggle('active', mode === 'search');
    if (recommendBtn) recommendBtn.classList.toggle('active', mode === 'recommend');
    
    const searchSubview = document.getElementById('discover-search-subview');
    const recommendSubview = document.getElementById('discover-recommend-subview');
    if (searchSubview) searchSubview.style.display = mode === 'search' ? 'block' : 'none';
    if (recommendSubview) recommendSubview.style.display = mode === 'recommend' ? 'block' : 'none';
    
    const grid = document.getElementById('films-grid');
    if (grid) grid.innerHTML = '';
}

async function executeDirectMovieSearch() {
    const query = document.getElementById('direct-search-input').value.trim();
    if (!query) return;
    
    const btn = document.getElementById('direct-search-btn');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>`;
    renderSkeletonGrid(8);
    
    try {
        const res = await fetch(`${API}/api/search_tmdb?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        const results = data.results || [];
        
        const mapped = results.map(m => {
            const isSaved = watchlistIds.has(m.id);
            return {
                id: m.id,
                movie_id: m.id,
                title: m.title,
                release_date: m.release_date || '',
                year: (m.release_date || '').split('-')[0] || '',
                genres: m.genre_ids ? m.genre_ids.map(gid => {
                    const genreDict = {
                        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
                        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
                        14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
                        9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
                        10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
                    };
                    return genreDict[gid] || '';
                }).filter(Boolean).join(', ') : '',
                overview: m.overview || '',
                poster_path: m.poster_path || '',
                backdrop_path: m.backdrop_path || '',
                ai_score: 3.5
            };
        });
        
        currentPicks = mapped;
        renderGrid(mapped);
    } catch(e) {
        console.error('Search error', e);
        const grid = document.getElementById('films-grid');
        if (grid) grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">Search failed. Please try again.</div>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7"/><line x1="15" y1="15" x2="19" y2="19"/></svg>`;
    }
}

function renderGrid(movies) {
    const grid = document.getElementById('films-grid');
    grid.innerHTML = '';
    if (!movies.length) {
        grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">No unwatched films found. Try a different mood.</div>';
        return;
    }
    movies.forEach(m => {
        renderedMoviesMap.set(m.id, m);
        const card = document.createElement('div');
        card.className = 'poster-card';
        card.onclick = () => openSpotlight(m);

        const poster = m.poster_path ? `${IMG500}${m.poster_path}` : '';
        const year = (m.release_date || '').split('-')[0] || '';
        const pct = Math.min(99, Math.max(60, Math.round((m.ai_score || 3.5) * 20)));
        const isSaved = watchlistIds.has(m.id);

        let badgeClass = 'poster-badge';
        let badgeText = `✦ ${pct}%`;
        if (m.is_direct_match) {
            if (m.is_watched) {
                badgeClass = 'poster-badge in-diary';
                badgeText = `👁️ ${pct}%`;
            } else {
                badgeClass = 'poster-badge direct-match';
                badgeText = `🎯 ${pct}%`;
            }
        }

        card.innerHTML = `
            ${poster ? `<img src="${poster}" alt="${m.title}" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;"></div>'}
            <div class="wl-card-actions" onclick="event.stopPropagation()">
                <button class="wl-act-btn diary-add" title="Rate & Log Film" onclick="openLogForMovieFromCardId(${m.id})">
                    👁️
                </button>
                <button class="wl-act-btn bookmark-add ${isSaved ? 'in-watchlist' : ''}" title="${isSaved ? 'In Watchlist' : 'Add to Watchlist'}" onclick="toggleWatchlistFromCardId(${m.id}, this)">
                    ${isSaved ? '✓' : '🔖'}
                </button>
            </div>
            <span class="${badgeClass}">${badgeText}</span>
            <div class="poster-info">
                <div class="poster-name">${m.title}</div>
                <div class="poster-sub">${m.is_direct_match ? 'Direct Match · ' : 'Match: ' + pct + '%'}${year ? ' · ' + year : ''}</div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function openLogForMovieFromCardId(id) {
    const m = renderedMoviesMap.get(id);
    if (m) {
        selectMovieForLog(m);
        openLogModal();
    }
}

async function toggleWatchlistFromCardId(id, btn) {
    const m = renderedMoviesMap.get(id);
    if (m) {
        await toggleWatchlistFromCard(m, btn);
    }
}

// ── Watchlist Management ──
async function toggleWatchlistFromCard(movie, btn) {
    const mId = movie.id || movie.movie_id;
    if (watchlistIds.has(mId)) {
        // Remove
        await fetch(`${API}/api/watchlist/remove`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_id: mId })
        });
        watchlistIds.delete(mId);
        btn.classList.remove('in-watchlist');
        btn.innerHTML = '🔖';
        btn.title = 'Add to Watchlist';
    } else {
        // Add
        await fetch(`${API}/api/watchlist/add`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(movie)
        });
        watchlistIds.add(mId);
        btn.classList.add('in-watchlist');
        btn.innerHTML = '✓';
        btn.title = 'In Watchlist';
    }
    updateWatchlistBadge(watchlistIds.size);
}

async function toggleWatchlistForCurrentSpotlight() {
    if (!currentSpotlight) return;
    const mId = currentSpotlight.id || currentSpotlight.movie_id;
    const btn = document.getElementById('spotlight-watchlist-toggle-btn');
    
    if (watchlistIds.has(mId)) {
        await fetch(`${API}/api/watchlist/remove`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_id: mId })
        });
        watchlistIds.delete(mId);
        btn.textContent = '🔖 Add to Watchlist';
        btn.classList.remove('primary');
    } else {
        await fetch(`${API}/api/watchlist/add`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentSpotlight)
        });
        watchlistIds.add(mId);
        btn.textContent = '✓ Saved in Watchlist';
        btn.classList.add('primary');
    }
    updateWatchlistBadge(watchlistIds.size);
    // Refresh discover and watchlist cards
    renderGrid(currentPicks);
}

function renderWatchlistSkeleton(count = 6) {
    const grid = document.getElementById('watchlist-grid');
    if (!grid) return;
    grid.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const card = document.createElement('div');
        card.className = 'poster-card skeleton';
        card.innerHTML = `
            <div class="skeleton-badge"></div>
            <div class="skeleton-info">
                <div class="skeleton-title"></div>
                <div class="skeleton-sub"></div>
            </div>
        `;
        grid.appendChild(card);
    }
}

async function fetchWatchlist() {
    const stream = document.getElementById('wl-stream-select')?.value || 'All Platforms';
    const sort = document.getElementById('wl-sort-select')?.value || 'Highest Predicted ★';
    const cacheKey = `${currentWatchlistCluster}_${sort}_${stream}`;
    const url = `${API}/api/watchlist?cluster=${encodeURIComponent(currentWatchlistCluster)}&sort=${encodeURIComponent(sort)}&platform=${encodeURIComponent(stream)}`;
    const dock = document.getElementById('dock-label');

    // 1. Instant Cache Hit (0ms render)
    if (watchlistCache.has(cacheKey)) {
        const cachedData = watchlistCache.get(cacheKey);
        const cachedItems = cachedData.watchlist || [];
        renderWatchlistGrid(cachedItems);
        updateWatchlistBadge(cachedData.total !== undefined ? cachedData.total : cachedItems.length);
        if (cachedItems.length > 0) {
            const avg = (cachedItems.reduce((acc, m) => acc + (m.ai_score || 3.8), 0) / cachedItems.length).toFixed(1);
            const avgEl = document.getElementById('wl-avg-score');
            if (avgEl) avgEl.textContent = `${avg}★`;
        }
    } else {
        renderWatchlistSkeleton(6);
        if (dock) {
            dock.innerHTML = `<span class="dock-loading"><span class="dock-spinner"></span> Curating your watchlist...</span>`;
        }
    }

    try {
        const res = await fetch(url);
        const data = await res.json();
        const items = data.watchlist || [];
        watchlistCache.set(cacheKey, data);
        if (currentWatchlistCluster === 'All' && sort === 'Highest Predicted ★' && stream === 'All Platforms') {
            try { localStorage.setItem('mbmr_cached_watchlist', JSON.stringify(data)); } catch(e) {}
        }
        renderWatchlistGrid(items);
        updateWatchlistBadge(data.total !== undefined ? data.total : items.length);

        if (items.length > 0) {
            const avg = (items.reduce((acc, m) => acc + (m.ai_score || 3.8), 0) / items.length).toFixed(1);
            const avgEl = document.getElementById('wl-avg-score');
            if (avgEl) avgEl.textContent = `${avg}★`;
        } else {
            const avgEl = document.getElementById('wl-avg-score');
            if (avgEl) avgEl.textContent = `—`;
        }
    } catch(e) { 
        console.error('Watchlist fetch error', e); 
        if (!watchlistCache.has(cacheKey)) {
            const grid = document.getElementById('watchlist-grid');
            if (grid) grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">Failed to load watchlist.</div>';
        }
    } finally {
        if (dock) dock.textContent = 'Your Curated Watchlist';
    }
}

function setWatchlistCluster(el, cluster) {
    document.querySelectorAll('.wl-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    currentWatchlistCluster = cluster;
    fetchWatchlist();
}

function renderWatchlistGrid(items) {
    const grid = document.getElementById('watchlist-grid');
    grid.innerHTML = '';

    if (!items.length) {
        grid.innerHTML = `
            <div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">
                <p style="font-size:16px;">No films found in this cluster.</p>
                <p style="font-size:12px;margin-top:6px;">Add movies from Search or click "Sync Letterboxd Watchlist".</p>
            </div>
        `;
        return;
    }

    items.forEach(m => {
        const card = document.createElement('div');
        card.className = 'poster-card';
        card.onclick = () => openSpotlight(m);

        const poster = m.poster_path ? `${IMG500}${m.poster_path}` : '';
        const year = m.year || '';
        const score = (m.ai_score || 3.8).toFixed(1);
        const runtime = m.runtime ? `${Math.floor(m.runtime/60)}h ${m.runtime%60}m` : '';

        card.innerHTML = `
            ${poster ? `<img src="${poster}" alt="${m.title}" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;"></div>'}
            <div class="wl-card-actions" onclick="event.stopPropagation()">
                <button class="wl-act-btn" title="Remove from Watchlist" onclick="removeFromWatchlistDirect(${m.movie_id})">✕</button>
            </div>
            <span class="poster-badge">★ ${score}</span>
            ${runtime ? `<span class="wl-runtime-tag">${runtime}</span>` : ''}
            <div class="poster-info">
                <div class="poster-name">${m.title}</div>
                <div class="poster-sub">${year}${m.clusters?.length ? ' · ' + m.clusters[0] : ''}</div>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function removeFromWatchlistDirect(movieId) {
    try {
        await fetch(`${API}/api/watchlist/remove`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_id: movieId })
        });
        watchlistIds.delete(movieId);
        watchlistCache.clear();
        fetchWatchlist();
        updateWatchlistBadge(watchlistIds.size);
    } catch(e) {}
}

// ── Matchmaker ("Pick For Me Tonight") ──
function openPickTonightModal() {
    document.getElementById('pick-tonight-modal').classList.add('open');
    document.getElementById('pick-result-area').style.display = 'none';
}

function closePickTonightModal() {
    document.getElementById('pick-tonight-modal').classList.remove('open');
}

async function executePickTonight() {
    const duration = document.getElementById('pick-duration').value;
    const mood = document.getElementById('pick-mood').value;
    const resArea = document.getElementById('pick-result-area');
    
    try {
        const res = await fetch(`${API}/api/watchlist/pick_tonight`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ duration, mood })
        });
        const data = await res.json();
        if (!data.success || !data.movie) {
            alert(data.message || 'No match found.');
            return;
        }

        currentMatchmakerWinner = data.movie;
        const m = data.movie;
        
        document.getElementById('winner-title').textContent = m.title;
        const year = m.year || '';
        const runtime = m.runtime ? `${Math.floor(m.runtime/60)}h ${m.runtime%60}m` : '';
        document.getElementById('winner-meta').textContent = `${year} · ${runtime} · ★ ${(m.ai_score||4.5).toFixed(1)} Predicted`;
        document.getElementById('winner-pitch').textContent = m.pitch || '';
        document.getElementById('winner-poster').src = m.poster_path ? `${IMG200}${m.poster_path}` : '';

        resArea.style.display = 'flex';
    } catch(e) {
        alert('Error picking movie.');
    }
}

function openWinnerSpotlight() {
    if (currentMatchmakerWinner) {
        closePickTonightModal();
        openSpotlight(currentMatchmakerWinner);
    }
}

function logWinnerDirectly() {
    if (currentMatchmakerWinner) {
        closePickTonightModal();
        selectedLogMovie = currentMatchmakerWinner;
        openLogForCurrentSpotlight();
    }
}

async function triggerWatchlistSync() {
    const btn = document.querySelector('.wl-sync-btn');
    const originalText = btn ? btn.innerHTML : '🔄 Sync Watchlist';
    if (btn) btn.innerHTML = '<span class="spinner"></span> Syncing Watchlist...';
    const msg = document.getElementById('sync-status-msg');
    if (msg) msg.textContent = 'Scraping Letterboxd watchlist...';

    const localUser = await MBMRStorage.get('letterboxd_username');
    const inputUser = document.getElementById('sync-user-input')?.value.trim();
    const username = inputUser || localUser || '';
    const tmdb = await MBMRStorage.get('tmdb_key');

    if (!username) {
        if (msg) msg.textContent = 'Please enter your username first.';
        if (btn) btn.innerHTML = originalText;
        return;
    }

    renderWatchlistSkeleton(6);
    try {
        const res = await fetch(`${API}/api/watchlist/sync`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, tmdb_key: tmdb })
        });
        const d = await res.json();
        // The sync runs in the background; refreshing before it finishes shows stale data.
        if (d.job_id) {
            const final = await pollImportJob(d.job_id, (s) => {
                if (msg) msg.textContent = `${s.progress || 0}% — ${s.message || 'Syncing...'}`;
            });
            if (msg) msg.textContent = final.message;
        } else {
            if (msg) msg.textContent = d.message || 'Watchlist synced!';
        }
        await loadStatus();
        await loadWatchlistIds();
        await fetchWatchlist();
    } catch(e) {
        if (msg) msg.textContent = 'Sync error.';
        fetchWatchlist();
    } finally {
        if (btn) btn.innerHTML = originalText;
    }
}

// ── Spotlight Drawer ──
function openSpotlight(m, isFromDiary = false) {
    currentSpotlight = m;
    const mId = m.id || m.movie_id;

    document.getElementById('spotlight-title').textContent = m.title || 'Untitled';
    const year = (m.release_date || m.year || '').split('-')[0] || '';
    document.getElementById('spotlight-meta').textContent = `${year} · TMDB ${m.vote_average || 'N/A'}`;
    const pct = Math.min(99, Math.max(60, Math.round((m.ai_score || 3.5) * 20)));
    document.getElementById('spotlight-ai-score').textContent = isFromDiary && m.user_rating ? `Logged: ★ ${m.user_rating}` : `${pct}% Match`;
    document.getElementById('spotlight-overview').textContent = m.overview || 'No synopsis available.';
    document.getElementById('spotlight-backdrop-img').src = m.backdrop_path ? `${IMG1280}${m.backdrop_path}` : (m.poster_path ? `${IMG500}${m.poster_path}` : '');
    document.getElementById('spotlight-tmdb-link').href = `https://www.themoviedb.org/movie/${mId}`;

    const wlBtn = document.getElementById('spotlight-watchlist-toggle-btn');
    const logBtn = document.getElementById('spotlight-log-btn');

    if (isFromDiary) {
        if (wlBtn) wlBtn.style.display = 'none';
        if (logBtn) logBtn.style.display = 'none';
    } else {
        if (wlBtn) {
            wlBtn.style.display = 'inline-flex';
            if (watchlistIds.has(mId)) {
                wlBtn.textContent = '✓ In Watchlist';
                wlBtn.classList.add('primary');
            } else {
                wlBtn.textContent = '🔖 Add to Watchlist';
                wlBtn.classList.remove('primary');
            }
        }
        if (logBtn) logBtn.style.display = 'inline-flex';
    }

    const pc = document.getElementById('spotlight-providers');
    pc.innerHTML = '';
    (m.providers || []).forEach(p => { const s = document.createElement('span'); s.className = 'sp-prov'; s.textContent = p; pc.appendChild(s); });
    if (!m.providers || !m.providers.length) pc.innerHTML = '<span style="font-size:11px;color:var(--text3)">Digital purchase / rent</span>';

    document.getElementById('ripple-section').style.display = 'none';
    document.getElementById('spotlight-overlay').classList.add('open');
}

function closeSpotlight() { document.getElementById('spotlight-overlay').classList.remove('open'); }
function closeSpotlightOnBg(e) { if (e.target.id === 'spotlight-overlay') closeSpotlight(); }

async function fetchRippleRecs() {
    if (!currentSpotlight) return;
    const mId = currentSpotlight.id || currentSpotlight.movie_id;
    const sec = document.getElementById('ripple-section');
    const grid = document.getElementById('ripple-grid');
    sec.style.display = 'block';
    grid.innerHTML = '<span style="color:var(--text3);font-size:12px">Discovering ripples...</span>';
    try {
        const d = await (await fetch(`${API}/api/ripple?movie_id=${mId}`)).json();
        grid.innerHTML = '';
        (d.ripples || []).slice(0, 4).forEach(r => {
            const c = document.createElement('div');
            c.className = 'ripple-card'; c.onclick = () => openSpotlight(r);
            const t = r.poster_path ? `${IMG200}${r.poster_path}` : '';
            c.innerHTML = `${t ? `<img src="${t}" style="width:28px;height:40px;border-radius:3px;object-fit:cover;">` : ''}
                <div style="flex:1;min-width:0"><div style="font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.title}</div><div style="font-size:10px;color:var(--text3)">★ ${(r.ai_score||3.5).toFixed(1)}</div></div>`;
            grid.appendChild(c);
        });
        if (!(d.ripples || []).length) grid.innerHTML = '<span style="color:var(--text3);font-size:12px">No ripples found.</span>';
    } catch(e) { grid.innerHTML = '<span style="color:var(--text3);font-size:12px">Failed.</span>'; }
}

// ── Journal (History) ──
let diaryRating = 'All';
let diaryViewMode = 'list';
let currentDiaryFilms = [];
let diaryCurrentPage = 1;
const DIARY_PAGE_SIZE = 50;

function setDiaryViewMode(mode) {
    diaryViewMode = mode;
    document.getElementById('j-view-list-btn')?.classList.toggle('active', mode === 'list');
    document.getElementById('j-view-grid-btn')?.classList.toggle('active', mode === 'grid');
    
    const listEl = document.getElementById('journal-list');
    const gridEl = document.getElementById('journal-grid');
    if (listEl && gridEl) {
        listEl.style.display = mode === 'list' ? 'flex' : 'none';
        gridEl.style.display = mode === 'grid' ? 'grid' : 'none';
    }
    renderJournal(currentDiaryFilms);
}

async function fetchDiary(resetPage = true) {
    if (resetPage) diaryCurrentPage = 1;
    const s = document.getElementById('diary-search')?.value.trim() || '';
    const sort = document.getElementById('diary-sort')?.value || 'Newest Log First';
    const cacheKey = `${s}_${diaryRating}_${sort}`;
    const url = `${API}/api/diary?search=${encodeURIComponent(s)}&rating=${encodeURIComponent(diaryRating)}&sort=${encodeURIComponent(sort)}`;

    // 1. Instant Cache Hit (0ms render)
    if (diaryCache.has(cacheKey)) {
        const cached = diaryCache.get(cacheKey);
        currentDiaryFilms = cached.films || [];
        renderJournal(currentDiaryFilms);
        if (cached.total) { 
            const jCount = document.getElementById('journal-total-count');
            const nCount = document.getElementById('nav-count');
            if (jCount) jCount.textContent = cached.total; 
            if (nCount) nCount.textContent = cached.total; 
        }
    }

    try {
        const d = await (await fetch(url)).json();
        diaryCache.set(cacheKey, d);
        if (!s && diaryRating === 'All' && sort === 'Newest Log First') {
            try { localStorage.setItem('mbmr_cached_diary', JSON.stringify(d)); } catch(e) {}
        }
        currentDiaryFilms = d.films || [];
        renderJournal(currentDiaryFilms);
        if (d.total) { 
            const jCount = document.getElementById('journal-total-count');
            const nCount = document.getElementById('nav-count');
            if (jCount) jCount.textContent = d.total; 
            if (nCount) nCount.textContent = d.total; 
        }
    } catch(e) { console.error('Diary error', e); }
}

function setDiaryRating(el, r) {
    document.querySelectorAll('.j-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active'); diaryRating = r; fetchDiary(true);
}

// The diary API returns canonical `title`/`year`. Older CSV-backed responses used the
// Letterboxd column names `Name`/`Year`, so both are accepted here.
function diaryTitle(f) { return f.title || f.Name || 'Untitled'; }
function diaryYear(f) { const y = f.year || f.Year; return y ? String(y).replace('.0', '') : ''; }

function openSpotlightFromDiary(f) {
    const movieObj = {
        id: f.movie_id,
        movie_id: f.movie_id,
        title: diaryTitle(f),
        release_date: diaryYear(f),
        year: diaryYear(f),
        poster_path: f.poster_path || '',
        backdrop_path: f.backdrop_path || f.poster_path || '',
        overview: f.overview || '',
        vote_average: f.Rating ? parseFloat(f.Rating).toFixed(1) : '7.5',
        user_rating: f.Rating ? parseFloat(f.Rating).toFixed(1) : null,
        ai_score: f.Rating ? parseFloat(f.Rating) : 3.8
    };
    openSpotlight(movieObj, true);
}

function renderJournal(films) {
    const listEl = document.getElementById('journal-list');
    const gridEl = document.getElementById('journal-grid');
    const pagEl = document.getElementById('journal-pagination');
    
    if (!films.length) {
        const emptyHtml = '<div style="grid-column:1/-1;text-align:center;padding:50px 0;color:var(--text3);">No diary entries found.</div>';
        if (listEl) listEl.innerHTML = emptyHtml;
        if (gridEl) gridEl.innerHTML = emptyHtml;
        if (pagEl) pagEl.innerHTML = '';
        return;
    }

    const totalPages = Math.ceil(films.length / DIARY_PAGE_SIZE);
    if (diaryCurrentPage > totalPages) diaryCurrentPage = totalPages;
    if (diaryCurrentPage < 1) diaryCurrentPage = 1;

    const startIdx = (diaryCurrentPage - 1) * DIARY_PAGE_SIZE;
    const pageFilms = films.slice(startIdx, startIdx + DIARY_PAGE_SIZE);

    if (diaryViewMode === 'list') {
        if (listEl) {
            listEl.innerHTML = '';
            pageFilms.forEach(f => {
                const row = document.createElement('div');
                row.className = 'j-row';
                row.onclick = () => openSpotlightFromDiary(f);

                const date = f.Date ? fmtDate(f.Date) : '';
                const stars = f.Rating ? fmtStars(f.Rating) : '<span style="color:var(--text3)">Unrated</span>';
                const poster = f.poster_path ? `${IMG200}${f.poster_path}` : '';
                const genres = f.genres || '';
                const title = diaryTitle(f);
                const year = diaryYear(f);

                row.innerHTML = `
                    <div class="j-poster-wrap">
                        ${poster ? `<img src="${poster}" alt="${title}" class="j-poster" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;display:flex;align-items:center;justify-content:center;color:#555;font-size:10px;">🎬</div>'}
                    </div>
                    <div class="j-date">${date}</div>
                    <div class="j-info">
                        <div class="j-title">${title}<span class="j-year">${year ? '(' + year + ')' : ''}</span></div>
                        ${genres ? `<div class="j-genres">${genres}</div>` : ''}
                    </div>
                    <div class="j-stars">${stars}</div>
                `;
                listEl.appendChild(row);
            });
        }
    } else {
        if (gridEl) {
            gridEl.innerHTML = '';
            pageFilms.forEach(f => {
                const card = document.createElement('div');
                card.className = 'poster-card';
                card.onclick = () => openSpotlightFromDiary(f);

                const poster = f.poster_path ? `${IMG500}${f.poster_path}` : '';
                const year = diaryYear(f);
                const title = diaryTitle(f);
                const rating = f.Rating ? parseFloat(f.Rating).toFixed(1) : '';

                card.innerHTML = `
                    ${poster ? `<img src="${poster}" alt="${title}" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;"></div>'}
                    ${rating ? `<span class="poster-badge journal-badge">★ ${rating}</span>` : ''}
                    <div class="poster-info">
                        <div class="poster-name">${title}</div>
                        <div class="poster-sub">${year}${f.Date ? ' · Logged ' + fmtDate(f.Date) : ''}</div>
                    </div>
                `;
                gridEl.appendChild(card);
            });
        }
    }

    renderJournalPagination(films.length);
}

function renderJournalPagination(totalFilms) {
    const pagEl = document.getElementById('journal-pagination');
    if (!pagEl) return;
    const totalPages = Math.ceil(totalFilms / DIARY_PAGE_SIZE);
    if (totalPages <= 1) {
        pagEl.innerHTML = '';
        return;
    }

    let html = `
        <button class="j-page-btn" onclick="changeDiaryPage(${diaryCurrentPage - 1})" ${diaryCurrentPage <= 1 ? 'disabled' : ''}>
            ‹ Prev
        </button>
    `;

    let startPage = Math.max(1, diaryCurrentPage - 2);
    let endPage = Math.min(totalPages, diaryCurrentPage + 2);
    if (diaryCurrentPage <= 3) endPage = Math.min(5, totalPages);
    if (diaryCurrentPage > totalPages - 3) startPage = Math.max(1, totalPages - 4);

    if (startPage > 1) {
        html += `<button class="j-page-btn" onclick="changeDiaryPage(1)">1</button>`;
        if (startPage > 2) html += `<span class="j-page-info">...</span>`;
    }

    for (let p = startPage; p <= endPage; p++) {
        html += `<button class="j-page-btn ${p === diaryCurrentPage ? 'active' : ''}" onclick="changeDiaryPage(${p})">${p}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) html += `<span class="j-page-info">...</span>`;
        html += `<button class="j-page-btn" onclick="changeDiaryPage(${totalPages})">${totalPages}</button>`;
    }

    html += `
        <button class="j-page-btn" onclick="changeDiaryPage(${diaryCurrentPage + 1})" ${diaryCurrentPage >= totalPages ? 'disabled' : ''}>
            Next ›
        </button>
        <span class="j-page-info">Page ${diaryCurrentPage} of ${totalPages} (${totalFilms} films)</span>
    `;

    pagEl.innerHTML = html;
}

function changeDiaryPage(page) {
    diaryCurrentPage = page;
    renderJournal(currentDiaryFilms);
    const viewJournal = document.getElementById('view-journal');
    if (viewJournal) viewJournal.scrollIntoView({ behavior: 'smooth' });
}

function fmtDate(s) { try { const d = new Date(s); return isNaN(d) ? s : d.toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}); } catch { return s; } }
function fmtStars(r) { const n = parseFloat(r); return isNaN(n) ? '' : '★'.repeat(Math.floor(n)) + (n % 1 ? '½' : '') + ` (${n.toFixed(1)})`; }

// ── Taste Radar ──
async function loadTasteRadar() {
    try {
        const u = (await MBMRStorage.get('letterboxd_username')) || (await MBMRStorage.get('mbmr_active_user')) || '';
        const d = await (await fetch(`${API}/api/taste_radar?user=${encodeURIComponent(u)}`)).json();
        drawRadar(d.radar || []);
        renderBadges(d.badges || []);
    } catch(e) { console.error('Taste error', e); }
}
function drawRadar(genres) {
    const cv = document.getElementById('radar-canvas'); if (!cv) return;
    const size = 440;
    cv.width = size;
    cv.height = size;
    const ctx = cv.getContext('2d'), w = size, h = size, cx = w/2, cy = h/2;
    const R = 150;
    ctx.clearRect(0,0,w,h);
    if (!genres || genres.length < 3) return;
    const n = genres.length, step = Math.PI*2/n;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1.5;
    for (let r = 0.25; r <= 1; r += 0.25) {
        ctx.beginPath();
        for (let i = 0; i < n; i++) { const a = i*step-Math.PI/2; const x = cx+Math.cos(a)*R*r; const y = cy+Math.sin(a)*R*r; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
        ctx.closePath(); ctx.stroke();
    }
    ctx.fillStyle = '#A3A3A3'; ctx.font = 'bold 13px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) { 
        const a = i*step-Math.PI/2; 
        ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(a)*R,cy+Math.sin(a)*R); ctx.stroke(); 
        const labelOffset = R + 26;
        ctx.fillText(genres[i].genre, cx+Math.cos(a)*labelOffset, cy+Math.sin(a)*labelOffset); 
    }
    ctx.beginPath();
    const pts = [];
    for (let i = 0; i < n; i++) { const a = i*step-Math.PI/2; const v = (genres[i].pct||50)/100; const x = cx+Math.cos(a)*R*v; const y = cy+Math.sin(a)*R*v; pts.push({x,y}); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
    ctx.closePath(); ctx.fillStyle = 'rgba(74,222,128,0.25)'; ctx.fill(); ctx.strokeStyle = '#4ADE80'; ctx.lineWidth = 2.5; ctx.stroke();
    pts.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,4.5,0,Math.PI*2); ctx.fillStyle = '#4ADE80'; ctx.fill(); });
}
function renderBadges(badges) {
    const l = document.getElementById('badges-list'); l.innerHTML = '';
    badges.forEach(b => { const c = document.createElement('div'); c.className = 'badge-card'; c.innerHTML = `<strong>${b.title}</strong><p>${b.desc}</p>`; l.appendChild(c); });
}

// ── Modern Toast Notification ──
let _toastTimer = null;
function showToast(message, glyph = '✓') {
    let t = document.getElementById('app-toast');
    if (!t) {
        t = document.createElement('div');
        t.id = 'app-toast';
        t.className = 'app-toast';
        document.body.appendChild(t);
    }
    t.innerHTML = `<span style="color:var(--accent);font-size:16px;">${glyph}</span> <span>${message}</span>`;
    t.classList.add('show');
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => {
        t.classList.remove('show');
    }, 3200);
}

// ── Log Modal ──
function openLogModal() { 
    document.getElementById('log-modal').classList.add('open'); 
    const searchInput = document.getElementById('log-search-input');
    if (searchInput) {
        if (!selectedLogMovie) {
            resetLogModalPreview();
        }
        searchInput.focus();
    }
}

function closeLogModal() { 
    document.getElementById('log-modal').classList.remove('open'); 
    const dd = document.getElementById('log-search-dropdown');
    if (dd) dd.style.display = 'none';
}

function openLogForCurrentSpotlight() {
    if (!currentSpotlight) return;
    selectMovieForLog(currentSpotlight);
    updateSliderLabel(4.5);
    openLogModal();
}

function closeModalBg(e, id) { 
    if (e.target.id === id) document.getElementById(id).classList.remove('open'); 
}

function selectMovieForLog(m) {
    selectedLogMovie = m;
    const dd = document.getElementById('log-search-dropdown');
    if (dd) dd.style.display = 'none';
    
    const titleEl = document.getElementById('modal-log-title');
    const yearEl = document.getElementById('modal-log-year');
    const posterEl = document.getElementById('modal-log-poster');
    const predEl = document.getElementById('modal-ai-pred');
    
    if (titleEl) titleEl.textContent = m.title || 'Selected Film';
    if (yearEl) yearEl.textContent = (m.release_date || m.year || '').split('-')[0];
    if (posterEl) {
        if (m.poster_path) {
            posterEl.src = `${IMG200}${m.poster_path}`;
            posterEl.style.display = 'block';
        } else {
            posterEl.src = '';
            posterEl.style.display = 'none';
        }
    }
    if (predEl) predEl.textContent = `★ ${(m.ai_score || 3.8).toFixed(1)}`;
    const slider = document.getElementById('rating-slider');
    if (slider) updateSliderLabel(slider.value);
}

async function searchTMDBLog(q) {
    const dd = document.getElementById('log-search-dropdown');
    if (!q || q.length < 2) { dd.style.display = 'none'; return; }
    try {
        const d = await (await fetch(`${API}/api/search_tmdb?q=${encodeURIComponent(q)}`)).json();
        dd.innerHTML = '';
        if (d.results?.length) {
            dd.style.display = 'block';
            d.results.forEach(m => {
                const it = document.createElement('div'); 
                it.className = 'log-dd-item';
                it.innerHTML = `<span>${m.title}</span><span style="color:var(--text3)">${(m.release_date||'').split('-')[0]}</span>`;
                
                it.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    selectMovieForLog(m);
                });
                dd.appendChild(it);
            });
        } else { dd.style.display = 'none'; }
    } catch { dd.style.display = 'none'; }
}

function updateSliderLabel(val) {
    const sliderVal = document.getElementById('slider-val');
    const userRate = document.getElementById('modal-user-rate');
    if (sliderVal) sliderVal.textContent = `${val}★`;
    if (userRate) userRate.textContent = `★ ${val}`;
    
    const predText = (document.getElementById('modal-ai-pred')?.textContent || '').replace('★','').trim();
    const pred = parseFloat(predText) || 3.8;
    const diff = (parseFloat(val) - pred).toFixed(1);
    const pill = document.getElementById('modal-diff-pill');
    if (pill) {
        pill.textContent = diff > 0 ? `+${diff} Above Prediction` : diff < 0 ? `${diff} Below Prediction` : 'Exact Match!';
    }
}

async function submitLogMovie() {
    if (!selectedLogMovie) { 
        showToast('Please select or search for a film first.', '⚠️'); 
        return; 
    }
    const mId = selectedLogMovie.id || selectedLogMovie.movie_id;
    if (!mId) {
        showToast('Invalid film selection.', '⚠️');
        return;
    }
    
    const slider = document.getElementById('rating-slider');
    const ratingVal = slider ? parseFloat(slider.value) : 4.0;
    const contextVal = document.getElementById('context-select')?.value || 'Alone';
    const genresVal = Array.isArray(selectedLogMovie.genres) 
        ? selectedLogMovie.genres.join(', ') 
        : (selectedLogMovie.genres || '');
    const yearVal = (selectedLogMovie.release_date || selectedLogMovie.year || '').split('-')[0].replace('.0', '');

    try {
        const res = await fetch(`${API}/api/log_movie`, { 
            method: 'POST', 
            headers: {'Content-Type':'application/json'}, 
            body: JSON.stringify({ 
                movie_id: mId, 
                title: selectedLogMovie.title || 'Untitled', 
                rating: ratingVal, 
                context: contextVal, 
                genres: genresVal,
                year: yearVal,
                poster_path: selectedLogMovie.poster_path || '',
                backdrop_path: selectedLogMovie.backdrop_path || '',
                overview: selectedLogMovie.overview || '' 
            }) 
        });
        const d = await res.json();
        
        closeLogModal(); 
        closeSpotlight();
        
        watchlistIds.delete(mId);
        watchlistCache.clear();
        diaryCache.clear();
        
        loadStatus(); 
        updateWatchlistBadge(watchlistIds.size);
        
        if (currentView === 'watchlist') fetchWatchlist();
        if (currentView === 'journal') fetchDiary();
        if (currentView === 'discover') generateRecommendations();
        
        showToast(d.message || `Logged "${selectedLogMovie.title}" (${ratingVal}★) to Diary!`);
        selectedLogMovie = null;
        resetLogModalPreview();
    } catch(err) { 
        console.error('Error logging movie:', err);
        showToast('Failed to log film. Please try again.', '⚠️'); 
    }
}

function resetLogModalPreview() {
    selectedLogMovie = null;
    const titleEl = document.getElementById('modal-log-title');
    const yearEl = document.getElementById('modal-log-year');
    const posterEl = document.getElementById('modal-log-poster');
    const predEl = document.getElementById('modal-ai-pred');
    const searchInput = document.getElementById('log-search-input');
    const slider = document.getElementById('rating-slider');
    
    if (titleEl) titleEl.textContent = 'Select a Movie';
    if (yearEl) yearEl.textContent = '';
    if (posterEl) {
        posterEl.src = '';
        posterEl.style.display = 'none';
    }
    if (predEl) predEl.textContent = '★ --';
    if (searchInput) searchInput.value = '';
    if (slider) {
        slider.value = 4.5;
        updateSliderLabel(4.5);
    }
}

// ── Sync Modal ──
function openSyncModal() { document.getElementById('sync-modal').classList.add('open'); }
function closeSyncModal() { document.getElementById('sync-modal').classList.remove('open'); }
async function triggerRSSSync() {
    const msg = document.getElementById('sync-status-msg');
    if (msg) msg.textContent = 'Syncing diary RSS...';
    
    const localUser = await MBMRStorage.get('letterboxd_username');
    const inputUser = document.getElementById('sync-user-input')?.value.trim();
    const username = inputUser || localUser || '';
    const tmdb = await MBMRStorage.get('tmdb_key');

    if (!username) {
        if (msg) msg.textContent = 'Please enter your username first.';
        return;
    }

    try {
        const d = await (await fetch(`${API}/api/sync_letterboxd`, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ username, tmdb_key: tmdb })
        })).json();
        if (d.job_id) {
            const final = await pollImportJob(d.job_id, (s) => {
                if (msg) msg.textContent = `${s.progress || 0}% — ${s.message || 'Syncing...'}`;
            });
            if (msg) msg.textContent = final.message;
        } else {
            if (msg) msg.textContent = d.message || 'Done!';
        }
        await loadStatus();
        await loadWatchlistIds();
        await fetchWatchlist();
        fetchDiary();
    } catch {
        if (msg) msg.textContent = 'Sync failed.';
    }
}
async function triggerRetrainAI() {
    const msg = document.getElementById('sync-status-msg');
    if (msg) msg.textContent = 'Retraining AI Model in RAM...';

    // The server needs to know which user's model to rebuild.
    const localUser = await MBMRStorage.get('letterboxd_username');
    const inputUser = document.getElementById('sync-user-input')?.value.trim();
    const username = inputUser || localUser || '';
    if (!username) {
        if (msg) msg.textContent = 'Please enter your username first.';
        return;
    }

    try {
        const d = await (await fetch(`${API}/api/retrain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        })).json();
        if (msg) msg.textContent = d.message || 'Done!';
        await loadStatus();
    } catch {
        if (msg) msg.textContent = 'Failed.';
    }
}

// ── Onboarding & Multi-Device PIN Authentication ──

function openOnboardingModal() {
    const modal = document.getElementById('onboarding-modal');
    if (modal) {
        modal.style.display = 'flex';
        modal.classList.add('open');
    }
}

function closeOnboardingModal() {
    const modal = document.getElementById('onboarding-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.classList.remove('open');
    }
}

async function checkOnboarding() {
    const user = await MBMRStorage.get('letterboxd_username');
    if (!user || user === 'guest') {
        openOnboardingModal();
    } else {
        const syncInput = document.getElementById('sync-user-input');
        if (syncInput) syncInput.value = user;
    }
}

function switchOnboardTab(tab) {
    const btnSetup = document.getElementById('tab-btn-setup');
    const btnLogin = document.getElementById('tab-btn-login');
    const formSetup = document.getElementById('onboard-form-setup');
    const formLogin = document.getElementById('onboard-form-login');

    if (tab === 'setup') {
        btnSetup?.classList.add('active');
        btnLogin?.classList.remove('active');
        if (formSetup) formSetup.style.display = 'block';
        if (formLogin) formLogin.style.display = 'none';
    } else {
        btnLogin?.classList.add('active');
        btnSetup?.classList.remove('active');
        if (formLogin) formLogin.style.display = 'block';
        if (formSetup) formSetup.style.display = 'none';
    }
}

function nextOnboardStep(step) {
    const s1 = document.getElementById('onboard-step-1');
    const s2 = document.getElementById('onboard-step-2');
    const u = document.getElementById('onboard-username')?.value.trim();

    if (step === 2) {
        if (!u) {
            alert('Please enter your Letterboxd username.');
            return;
        }
        if (s1) s1.style.display = 'none';
        if (s2) s2.style.display = 'block';
    } else if (step === 1) {
        if (s2) s2.style.display = 'none';
        if (s1) s1.style.display = 'block';
    }
}

async function submitLoginProfile() {
    const username = document.getElementById('login-username')?.value.trim();
    const pin = document.getElementById('login-pin')?.value.trim();
    const errEl = document.getElementById('login-error-msg');

    if (!username) {
        if (errEl) { errEl.textContent = 'Please enter your Letterboxd username.'; errEl.style.display = 'block'; }
        return;
    }

    try {
        const res = await fetch(`${API}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, pin })
        });
        const data = await res.json();
        if (!data.success || !data.user) {
            if (errEl) { errEl.textContent = data.message || 'Invalid PIN.'; errEl.style.display = 'block'; }
            return;
        }

        // Save authenticated credentials locally
        await MBMRStorage.set('letterboxd_username', data.user.username);
        if (data.user.tmdb_key) await MBMRStorage.set('tmdb_key', data.user.tmdb_key);
        if (data.user.gemini_key) await MBMRStorage.set('gemini_key', data.user.gemini_key);

        const syncInput = document.getElementById('sync-user-input');
        if (syncInput) syncInput.value = data.user.username;

        closeOnboardingModal();
        await loadStatus();
        await loadWatchlistIds();
        await fetchWatchlist();
        generateRecommendations();
    } catch(e) {
        if (errEl) { errEl.textContent = 'Connection error. Please try again.'; errEl.style.display = 'block'; }
    }
}

async function completeOnboarding() {
    const username = document.getElementById('onboard-username')?.value.trim().replace(/^@/, '');
    const pin = document.getElementById('onboard-pin')?.value.trim();
    const tmdb = document.getElementById('onboard-tmdb')?.value.trim();
    const gemini = document.getElementById('onboard-gemini')?.value.trim();

    if (!username) {
        alert('Please provide your Letterboxd username.');
        return;
    }

    // Save locally
    await MBMRStorage.set('letterboxd_username', username);
    if (tmdb) await MBMRStorage.set('tmdb_key', tmdb);
    if (gemini) await MBMRStorage.set('gemini_key', gemini);

    const s2 = document.getElementById('onboard-step-2');
    const sp = document.getElementById('onboard-progress');
    if (s2) s2.style.display = 'none';
    if (sp) sp.style.display = 'block';

    const pTitle = document.getElementById('onboard-progress-title');
    const pSub = document.getElementById('onboard-progress-sub');
    const pFill = document.getElementById('onboard-progress-bar-fill');
    const pPct = document.getElementById('onboard-pct-text');

    try {
        // Start async onboarding job
        const startRes = await fetch(`${API}/api/onboarding/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, pin, tmdb_key: tmdb, gemini_key: gemini })
        });
        const startData = await startRes.json();
        if (!startData.success || !startData.job_id) {
            throw new Error(startData.message || 'Failed to start onboarding');
        }

        const jobId = startData.job_id;

        // Poll job status until complete
        const pollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch(`${API}/api/onboarding/status?job_id=${jobId}`);
                const statusData = await statusRes.json();

                const pct = statusData.progress || 10;
                if (pFill) pFill.style.width = `${pct}%`;
                if (pPct) pPct.textContent = `${pct}%`;
                if (pTitle) pTitle.textContent = statusData.message || 'Calibrating...';
                if (pSub && statusData.stage) pSub.textContent = `Stage: ${statusData.stage}`;

                if (statusData.status === 'completed') {
                    clearInterval(pollInterval);
                    if (pFill) pFill.style.width = '100%';
                    if (pPct) pPct.textContent = '100%';
                    setTimeout(async () => {
                        closeOnboardingModal();
                        const syncInput = document.getElementById('sync-user-input');
                        if (syncInput) syncInput.value = username;
                        await loadStatus();
                        await loadWatchlistIds();
                        await fetchWatchlist();
                        generateRecommendations();
                    }, 800);
                } else if (statusData.status === 'failed') {
                    clearInterval(pollInterval);
                    alert(statusData.error || 'Onboarding encountered an issue.');
                    closeOnboardingModal();
                }
            } catch (pollErr) {
                console.error("Status poll error:", pollErr);
            }
        }, 800);

    } catch (err) {
        alert(`Onboarding error: ${err.message}`);
        closeOnboardingModal();
    }
}

// ── Letterboxd Full History CSV Upload Handlers ──

// A full export runs as a background job (hundreds of TMDB lookups), so the upload
// response only carries a job_id. Poll it until it settles.
function pollImportJob(jobId, onProgress) {
    return new Promise((resolve) => {
        const timer = setInterval(async () => {
            try {
                const res = await fetch(`${API}/api/onboarding/status?job_id=${jobId}`);
                const data = await res.json();

                if (onProgress) onProgress(data);

                if (data.status === 'completed') {
                    clearInterval(timer);
                    resolve({ success: true, message: data.message || 'Import complete!' });
                } else if (data.status === 'failed' || data.status === 'not_found') {
                    clearInterval(timer);
                    resolve({ success: false, message: data.error || data.message || 'Import failed.' });
                }
            } catch (err) {
                console.error('Import poll error:', err);
            }
        }, 1000);
    });
}

async function handleOnboardCSVSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    const usernameInput = document.getElementById('onboard-username');
    const username = usernameInput?.value.trim().replace(/^@/, '') || file.name.replace(/\.csv$/i, '');
    if (!usernameInput.value.trim()) usernameInput.value = username;

    const label = document.getElementById('onboard-csv-label');
    if (label) label.innerHTML = `⏳ Reading <strong>${file.name}</strong>...`;

    const reader = new FileReader();
    reader.onload = async (e) => {
        const text = e.target.result;
        if (label) label.innerHTML = `⚡ Uploading & calibrating AI on <strong>${file.name}</strong>...`;
        try {
            const res = await fetch(`${API}/api/import_csv`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    csv_content: text,
                    is_watchlist: file.name.toLowerCase().includes('watchlist')
                })
            });
            const data = await res.json();
            if (!data.success || !data.job_id) {
                if (label) label.innerHTML = `⚠️ ${data.message || 'Import error'}`;
                return;
            }

            const final = await pollImportJob(data.job_id, (s) => {
                if (label) label.innerHTML = `⚡ ${s.progress || 0}% — ${s.message || 'Importing...'}`;
            });

            if (final.success) {
                if (label) label.innerHTML = `✓ <strong>${final.message}</strong>`;
                await MBMRStorage.set('letterboxd_username', username);
            } else {
                if (label) label.innerHTML = `⚠️ ${final.message}`;
            }
        } catch(err) {
            if (label) label.innerHTML = `⚠️ Upload failed. Please try again.`;
        }
    };
    reader.readAsText(file);
}

async function handleSyncCSVSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    const localUser = await MBMRStorage.get('letterboxd_username');
    const inputUser = document.getElementById('sync-user-input')?.value.trim();
    const username = inputUser || localUser || file.name.replace(/\.csv$/i, '');

    const msg = document.getElementById('sync-status-msg');
    if (msg) msg.textContent = `Importing ${file.name}...`;

    const reader = new FileReader();
    reader.onload = async (e) => {
        const text = e.target.result;
        try {
            const res = await fetch(`${API}/api/import_csv`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    csv_content: text,
                    is_watchlist: file.name.toLowerCase().includes('watchlist')
                })
            });
            const data = await res.json();
            if (!data.success || !data.job_id) {
                if (msg) msg.textContent = data.message || 'CSV import failed.';
                return;
            }

            const final = await pollImportJob(data.job_id, (s) => {
                if (msg) msg.textContent = `${s.progress || 0}% — ${s.message || 'Importing...'}`;
            });

            if (msg) msg.textContent = final.message;
            await loadStatus();
            await loadWatchlistIds();
            await fetchWatchlist();
            if (typeof fetchDiary === 'function') fetchDiary();
        } catch(err) {
            if (msg) msg.textContent = 'CSV import failed.';
        }
    };
    reader.readAsText(file);
}


