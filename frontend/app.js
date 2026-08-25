// ================================================================
// CINEAI — RAYCAST-STYLE CONTROLLER WITH WATCHLIST & MATCHMAKER
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

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    loadStatus();
    loadWatchlistIds();
    const promptInput = document.getElementById('prompt-input');
    if (!promptInput.value.trim()) {
        promptInput.value = 'Mind-Bending';
    }
    generateRecommendations();
    promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') generateRecommendations();
    });
    promptInput.addEventListener('input', () => {
        document.querySelectorAll('.vibe').forEach(v => v.classList.remove('active'));
    });
});

async function loadStatus() {
    try {
        const d = await (await fetch(`${API}/api/status`)).json();
        if (d.total_films) {
            document.getElementById('nav-count').textContent = d.total_films;
            document.getElementById('journal-total-count').textContent = d.total_films;
        }
        if (d.avg_rating) document.getElementById('journal-avg-rating').textContent = d.avg_rating;
        if (d.username) document.getElementById('profile-user').textContent = `@${d.username}`;
        if (d.watchlist_count !== undefined) {
            updateWatchlistBadge(d.watchlist_count);
        }
    } catch(e) { console.warn('Status fetch failed', e); }
}

async function loadWatchlistIds() {
    try {
        const d = await (await fetch(`${API}/api/watchlist`)).json();
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
    document.querySelectorAll('.rail-icon').forEach(b => b.classList.toggle('active', b.dataset.view === name));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const v = document.getElementById(`view-${name}`);
    if (v) v.classList.add('active');
    
    const labels = { 
        discover: 'Discover New Movies', 
        watchlist: 'Your Curated Watchlist',
        journal: 'Your Film Journal (740 Lifetime Films)', 
        taste: 'Taste Radar & Achievements' 
    };
    document.getElementById('dock-label').textContent = labels[name] || '';
    
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
        const res = await fetch(`${API}/api/recommend`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                context: document.getElementById('context-select').value,
                streaming: document.getElementById('stream-select').value
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

function renderGrid(movies) {
    const grid = document.getElementById('films-grid');
    grid.innerHTML = '';
    if (!movies.length) {
        grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px 0;color:var(--text3);">No unwatched films found. Try a different mood.</div>';
        return;
    }
    movies.forEach(m => {
        const card = document.createElement('div');
        card.className = 'poster-card';
        card.onclick = () => openSpotlight(m);

        const poster = m.poster_path ? `${IMG500}${m.poster_path}` : '';
        const year = (m.release_date || '').split('-')[0] || '';
        const pct = Math.min(99, Math.max(60, Math.round((m.ai_score || 3.5) * 20)));
        const isSaved = watchlistIds.has(m.id);

        let badgeClass = 'poster-badge';
        let badgeText = `Match: ${pct}%`;
        if (m.is_direct_match) {
            if (m.is_watched) {
                badgeClass = 'poster-badge in-diary';
                badgeText = `👁️ In Diary · ${pct}%`;
            } else {
                badgeClass = 'poster-badge direct-match';
                badgeText = `🎯 Match · ${pct}%`;
            }
        }

        card.innerHTML = `
            ${poster ? `<img src="${poster}" alt="${m.title}" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;"></div>'}
            <div class="wl-card-actions" onclick="event.stopPropagation()">
                <button class="wl-act-btn bookmark-add ${isSaved ? 'in-watchlist' : ''}" title="${isSaved ? 'In Watchlist' : 'Add to Watchlist'}" onclick="toggleWatchlistFromCard(${JSON.stringify(m).replace(/"/g, '&quot;')}, this)">
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

async function fetchWatchlist() {
    const stream = document.getElementById('wl-stream-select').value;
    const sort = document.getElementById('wl-sort-select').value;
    const url = `${API}/api/watchlist?cluster=${encodeURIComponent(currentWatchlistCluster)}&sort=${encodeURIComponent(sort)}&platform=${encodeURIComponent(stream)}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        const items = data.watchlist || [];
        renderWatchlistGrid(items);
        updateWatchlistBadge(data.total || items.length);

        if (items.length > 0) {
            const avg = (items.reduce((acc, m) => acc + (m.ai_score || 3.8), 0) / items.length).toFixed(1);
            document.getElementById('wl-avg-score').textContent = `${avg}★`;
        }
    } catch(e) { console.error('Watchlist fetch error', e); }
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
            <span class="poster-badge">Predicted: ${score}★</span>
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
    if (btn) btn.textContent = '🔄 Syncing from Letterboxd...';
    try {
        const res = await fetch(`${API}/api/watchlist/sync`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: 'sarthi_watcher' })
        });
        const d = await res.json();
        alert(d.message || 'Watchlist synced!');
        loadWatchlistIds();
        fetchWatchlist();
    } catch(e) {
        alert('Sync error.');
    } finally {
        if (btn) btn.textContent = '🔄 Sync Letterboxd Watchlist';
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

async function fetchDiary() {
    const s = document.getElementById('diary-search').value.trim();
    const sort = document.getElementById('diary-sort').value;
    try {
        const d = await (await fetch(`${API}/api/diary?search=${encodeURIComponent(s)}&rating=${encodeURIComponent(diaryRating)}&sort=${encodeURIComponent(sort)}`)).json();
        currentDiaryFilms = d.films || [];
        renderJournal(currentDiaryFilms);
        if (d.total) { document.getElementById('journal-total-count').textContent = d.total; document.getElementById('nav-count').textContent = d.total; }
    } catch(e) { console.error('Diary error', e); }
}

function setDiaryRating(el, r) {
    document.querySelectorAll('.j-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active'); diaryRating = r; fetchDiary();
}

function openSpotlightFromDiary(f) {
    const movieObj = {
        id: f.movie_id,
        movie_id: f.movie_id,
        title: f.Name,
        release_date: f.Year ? String(f.Year) : '',
        year: f.Year ? String(f.Year) : '',
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
    
    if (!films.length) {
        const emptyHtml = '<div style="grid-column:1/-1;text-align:center;padding:50px 0;color:var(--text3);">No diary entries found.</div>';
        if (listEl) listEl.innerHTML = emptyHtml;
        if (gridEl) gridEl.innerHTML = emptyHtml;
        return;
    }

    if (diaryViewMode === 'list') {
        if (listEl) {
            listEl.innerHTML = '';
            films.forEach(f => {
                const row = document.createElement('div');
                row.className = 'j-row';
                row.onclick = () => openSpotlightFromDiary(f);

                const date = f.Date ? fmtDate(f.Date) : '';
                const stars = f.Rating ? fmtStars(f.Rating) : '<span style="color:var(--text3)">Unrated</span>';
                const poster = f.poster_path ? `${IMG200}${f.poster_path}` : '';
                const genres = f.genres || '';

                row.innerHTML = `
                    <div class="j-poster-wrap">
                        ${poster ? `<img src="${poster}" alt="${f.Name}" class="j-poster" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;display:flex;align-items:center;justify-content:center;color:#555;font-size:10px;">🎬</div>'}
                    </div>
                    <div class="j-date">${date}</div>
                    <div class="j-info">
                        <div class="j-title">${f.Name}<span class="j-year">${f.Year ? '(' + f.Year + ')' : ''}</span></div>
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
            films.forEach(f => {
                const card = document.createElement('div');
                card.className = 'poster-card';
                card.onclick = () => openSpotlightFromDiary(f);

                const poster = f.poster_path ? `${IMG500}${f.poster_path}` : '';
                const year = f.Year || '';
                const rating = f.Rating ? parseFloat(f.Rating).toFixed(1) : '';

                card.innerHTML = `
                    ${poster ? `<img src="${poster}" alt="${f.Name}" loading="lazy">` : '<div style="width:100%;height:100%;background:#222;"></div>'}
                    ${rating ? `<span class="poster-badge journal-badge">★ ${rating}</span>` : ''}
                    <div class="poster-info">
                        <div class="poster-name">${f.Name}</div>
                        <div class="poster-sub">${year}${f.Date ? ' · Logged ' + fmtDate(f.Date) : ''}</div>
                    </div>
                `;
                gridEl.appendChild(card);
            });
        }
    }
}

function fmtDate(s) { try { const d = new Date(s); return isNaN(d) ? s : d.toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}); } catch { return s; } }
function fmtStars(r) { const n = parseFloat(r); return isNaN(n) ? '' : '★'.repeat(Math.floor(n)) + (n % 1 ? '½' : '') + ` (${n.toFixed(1)})`; }

// ── Taste Radar ──
async function loadTasteRadar() {
    try {
        const d = await (await fetch(`${API}/api/taste_radar`)).json();
        drawRadar(d.radar || []); renderBadges(d.badges || []);
    } catch(e) { console.error('Taste error', e); }
}
function drawRadar(genres) {
    const cv = document.getElementById('radar-canvas'); if (!cv) return;
    const ctx = cv.getContext('2d'), w = cv.width, h = cv.height, cx = w/2, cy = h/2, R = 160;
    ctx.clearRect(0,0,w,h);
    if (genres.length < 3) return;
    const n = genres.length, step = Math.PI*2/n;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
    for (let r = 0.25; r <= 1; r += 0.25) {
        ctx.beginPath();
        for (let i = 0; i < n; i++) { const a = i*step-Math.PI/2; const x = cx+Math.cos(a)*R*r; const y = cy+Math.sin(a)*R*r; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
        ctx.closePath(); ctx.stroke();
    }
    ctx.fillStyle = '#A3A3A3'; ctx.font = '11px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) { const a = i*step-Math.PI/2; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(a)*R,cy+Math.sin(a)*R); ctx.stroke(); ctx.fillText(genres[i].genre,cx+Math.cos(a)*(R+22),cy+Math.sin(a)*(R+22)); }
    ctx.beginPath();
    const pts = [];
    for (let i = 0; i < n; i++) { const a = i*step-Math.PI/2; const v = (genres[i].pct||50)/100; const x = cx+Math.cos(a)*R*v; const y = cy+Math.sin(a)*R*v; pts.push({x,y}); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
    ctx.closePath(); ctx.fillStyle = 'rgba(74,222,128,0.2)'; ctx.fill(); ctx.strokeStyle = '#4ADE80'; ctx.lineWidth = 2; ctx.stroke();
    pts.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fillStyle = '#4ADE80'; ctx.fill(); });
}
function renderBadges(badges) {
    const l = document.getElementById('badges-list'); l.innerHTML = '';
    badges.forEach(b => { const c = document.createElement('div'); c.className = 'badge-card'; c.innerHTML = `<strong>${b.title}</strong><p>${b.desc}</p>`; l.appendChild(c); });
}

// ── Log Modal ──
function openLogModal() { document.getElementById('log-modal').classList.add('open'); }
function closeLogModal() { document.getElementById('log-modal').classList.remove('open'); }
function openLogForCurrentSpotlight() {
    if (!currentSpotlight) return;
    selectedLogMovie = currentSpotlight;
    document.getElementById('modal-log-title').textContent = selectedLogMovie.title;
    document.getElementById('modal-log-year').textContent = (selectedLogMovie.release_date||selectedLogMovie.year||'').split('-')[0];
    document.getElementById('modal-log-poster').src = selectedLogMovie.poster_path ? `${IMG200}${selectedLogMovie.poster_path}` : '';
    document.getElementById('modal-ai-pred').textContent = `★ ${(selectedLogMovie.ai_score||3.8).toFixed(1)}`;
    updateSliderLabel(4.5);
    openLogModal();
}
function closeModalBg(e, id) { if (e.target.id === id) document.getElementById(id).classList.remove('open'); }

async function searchTMDBLog(q) {
    const dd = document.getElementById('log-search-dropdown');
    if (!q || q.length < 2) { dd.style.display = 'none'; return; }
    try {
        const d = await (await fetch(`${API}/api/search_tmdb?q=${encodeURIComponent(q)}`)).json();
        dd.innerHTML = '';
        if (d.results?.length) {
            dd.style.display = 'block';
            d.results.forEach(m => {
                const it = document.createElement('div'); it.className = 'log-dd-item';
                it.innerHTML = `<span>${m.title}</span><span style="color:var(--text3)">${(m.release_date||'').split('-')[0]}</span>`;
                it.onclick = () => { 
                    selectedLogMovie = m; 
                    dd.style.display = 'none'; 
                    document.getElementById('modal-log-title').textContent = m.title; 
                    document.getElementById('modal-log-year').textContent = (m.release_date||'').split('-')[0]; 
                    document.getElementById('modal-log-poster').src = m.poster_path ? `${IMG200}${m.poster_path}` : ''; 
                    document.getElementById('modal-ai-pred').textContent = `★ ${(m.ai_score||3.8).toFixed(1)}`; 
                    updateSliderLabel(document.getElementById('rating-slider').value); 
                };
                dd.appendChild(it);
            });
        } else { dd.style.display = 'none'; }
    } catch { dd.style.display = 'none'; }
}

function updateSliderLabel(val) {
    document.getElementById('slider-val').textContent = `${val}★`;
    document.getElementById('modal-user-rate').textContent = `★ ${val}`;
    const pred = parseFloat((document.getElementById('modal-ai-pred').textContent||'').replace('★','').trim()) || 3.8;
    const diff = (parseFloat(val) - pred).toFixed(1);
    const pill = document.getElementById('modal-diff-pill');
    pill.textContent = diff > 0 ? `+${diff} Above Prediction` : diff < 0 ? `${diff} Below Prediction` : 'Exact Match!';
}

async function submitLogMovie() {
    if (!selectedLogMovie) { alert('Select a movie first.'); return; }
    const mId = selectedLogMovie.id || selectedLogMovie.movie_id;
    try {
        const d = await (await fetch(`${API}/api/log_movie`, { 
            method: 'POST', 
            headers: {'Content-Type':'application/json'}, 
            body: JSON.stringify({ 
                movie_id: mId, 
                title: selectedLogMovie.title, 
                rating: parseFloat(document.getElementById('rating-slider').value), 
                context: document.getElementById('context-select').value, 
                overview: selectedLogMovie.overview||'' 
            }) 
        })).json();
        
        closeLogModal(); 
        watchlistIds.delete(mId);
        loadStatus(); 
        updateWatchlistBadge(watchlistIds.size);
        alert(d.message || 'Logged!');
    } catch { alert('Error logging.'); }
}

// ── Sync Modal ──
function openSyncModal() { document.getElementById('sync-modal').classList.add('open'); }
function closeSyncModal() { document.getElementById('sync-modal').classList.remove('open'); }
async function triggerRSSSync() {
    const msg = document.getElementById('sync-status-msg'); msg.textContent = 'Syncing diary RSS...';
    try { const d = await (await fetch(`${API}/api/sync_letterboxd`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username:'sarthi_watcher'}) })).json(); msg.textContent = d.message || 'Done!'; loadStatus(); } catch { msg.textContent = 'Sync failed.'; }
}
async function triggerRetrainAI() {
    const msg = document.getElementById('sync-status-msg'); msg.textContent = 'Retraining AI Model...';
    try { const d = await (await fetch(`${API}/api/retrain`, {method:'POST'})).json(); msg.textContent = d.message || 'Done!'; } catch { msg.textContent = 'Failed.'; }
}
