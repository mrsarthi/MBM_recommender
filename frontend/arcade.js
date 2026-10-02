// ================================================================
// CINEMA LOUNGE ARCADE ENGINE
// ================================================================
// Cold-start games that run while the backend is waking up.
// Games: Trivia Blitz (OpenTDB film trivia), Title Scramble
// ================================================================

(function () {
    'use strict';

    const OPEN_TDB_URL = 'https://opentdb.com/api.php?amount=10&category=11&type=multiple';
    const OPEN_TDB_TOKEN_URL = 'https://opentdb.com/api_token.php';
    const TOKEN_KEY = 'opentdb_session';
    const SEEN_KEY = 'mbmr_trivia_seen';
    const SEEN_LIMIT = 500;
    const TRIVIA_SECONDS = 15;
    const SCRAMBLE_SECONDS = 60;
    const ROUND_SIZE = 10;

    // ── Offline fallback bank (used only when OpenTDB is unreachable) ──
    const triviaBank = [
        { q: "Which film won Best Picture at the 2020 Academy Awards?", c: "Parasite", i: ["1917", "Joker", "Once Upon a Time in Hollywood"] },
        { q: "Who directed 'Pulp Fiction' (1994)?", c: "Quentin Tarantino", i: ["Martin Scorsese", "Spike Lee", "Oliver Stone"] },
        { q: "In 'The Matrix' (1999), which pill does Neo take?", c: "Red", i: ["Blue", "Green", "Yellow"] },
        { q: "Who directed 'Spirited Away' (2001)?", c: "Hayao Miyazaki", i: ["Isao Takahata", "Mamoru Hosoda", "Satoshi Kon"] },
        { q: "Who played the Joker in 'The Dark Knight' (2008)?", c: "Heath Ledger", i: ["Joaquin Phoenix", "Jared Leto", "Jack Nicholson"] },
        { q: "Who directed '2001: A Space Odyssey' (1968)?", c: "Stanley Kubrick", i: ["Ridley Scott", "Steven Spielberg", "George Lucas"] },
        { q: "Who played Ellen Ripley in 'Alien' (1979)?", c: "Sigourney Weaver", i: ["Linda Hamilton", "Jamie Lee Curtis", "Carrie Fisher"] },
        { q: "Who directed 'Citizen Kane' (1941)?", c: "Orson Welles", i: ["Alfred Hitchcock", "John Huston", "Billy Wilder"] },
        { q: "Who directed 'Psycho' (1960)?", c: "Alfred Hitchcock", i: ["Roman Polanski", "John Carpenter", "Brian De Palma"] },
        { q: "Who played Aragorn in 'The Lord of the Rings' trilogy?", c: "Viggo Mortensen", i: ["Russell Crowe", "Orlando Bloom", "Sean Bean"] },
        { q: "Who voiced Shrek in 'Shrek' (2001)?", c: "Mike Myers", i: ["Eddie Murphy", "Jim Carrey", "Antonio Banderas"] },
        { q: "Who directed 'The Godfather' (1972)?", c: "Francis Ford Coppola", i: ["Martin Scorsese", "Sidney Lumet", "Brian De Palma"] },
        { q: "Han Solo first appeared in which 1977 film?", c: "Star Wars", i: ["Close Encounters of the Third Kind", "Superman", "Logan's Run"] },
        { q: "Which film features the line \"Here's looking at you, kid\"?", c: "Casablanca", i: ["The Maltese Falcon", "Sunset Boulevard", "Double Indemnity"] },
        { q: "Which film features the line \"Life is like a box of chocolates\"?", c: "Forrest Gump", i: ["The Shawshank Redemption", "Big", "Rain Man"] },
        { q: "Who played the Bride in 'Kill Bill: Volume 1' (2003)?", c: "Uma Thurman", i: ["Lucy Liu", "Vivica A. Fox", "Daryl Hannah"] },
        { q: "Who directed 'Jurassic Park' (1993)?", c: "Steven Spielberg", i: ["James Cameron", "Robert Zemeckis", "Joe Johnston"] },
        { q: "In 'Inception' (2010), what does Cobb use as his totem?", c: "A spinning top", i: ["A loaded die", "A chess bishop", "A poker chip"] },
        { q: "In 'Interstellar' (2014), the wormhole appears near which planet?", c: "Saturn", i: ["Jupiter", "Neptune", "Mars"] },
        { q: "Which 1975 Steven Spielberg film is about a great white shark?", c: "Jaws", i: ["Orca", "Piranha", "The Deep"] },
        { q: "Which film features the line \"I'm the king of the world!\"?", c: "Titanic", i: ["The Wolf of Wall Street", "Gladiator", "Braveheart"] },
        { q: "In 'Back to the Future' (1985), what speed must the DeLorean reach to time travel?", c: "88 mph", i: ["77 mph", "99 mph", "121 mph"] },
        { q: "Who directed 'Parasite' (2019)?", c: "Bong Joon-ho", i: ["Park Chan-wook", "Lee Chang-dong", "Kim Jee-woon"] },
        { q: "Who played Andy Dufresne in 'The Shawshank Redemption' (1994)?", c: "Tim Robbins", i: ["Morgan Freeman", "Kevin Costner", "Tom Hanks"] },
        { q: "Who composed the score for 'Star Wars' (1977)?", c: "John Williams", i: ["Hans Zimmer", "Ennio Morricone", "Howard Shore"] },
        { q: "Which film won the first Academy Award for Best Animated Feature?", c: "Shrek", i: ["Monsters, Inc.", "Jimmy Neutron: Boy Genius", "Spirited Away"] },
        { q: "Who played Hannibal Lecter in 'The Silence of the Lambs' (1991)?", c: "Anthony Hopkins", i: ["Brian Cox", "Gary Oldman", "Ralph Fiennes"] },
        { q: "Who directed 'Mulholland Drive' (2001)?", c: "David Lynch", i: ["David Cronenberg", "David Fincher", "Paul Thomas Anderson"] },
        { q: "What is the name of the hotel in 'The Shining' (1980)?", c: "The Overlook Hotel", i: ["The Stanley Hotel", "The Bates Motel", "The Grand Budapest Hotel"] },
        { q: "In which Pixar film is a lone robot left to clean up an abandoned Earth?", c: "WALL-E", i: ["Up", "Ratatouille", "Toy Story 3"] },
        { q: "Who plays the title role in 'John Wick' (2014)?", c: "Keanu Reeves", i: ["Liam Neeson", "Jason Statham", "Bob Odenkirk"] },
        { q: "Which 1982 Ridley Scott film features replicants?", c: "Blade Runner", i: ["Alien", "Tron", "The Thing"] },
        { q: "Who directed 'Seven Samurai' (1954)?", c: "Akira Kurosawa", i: ["Yasujirō Ozu", "Kenji Mizoguchi", "Masaki Kobayashi"] }
    ];

    // ── Scramble bank (film anagrams with clues) ──
    const scrambleBank = [
        { answer: "INCEPTION", year: "2010", clue: "Dream heist thriller about planting ideas in minds" },
        { answer: "TENET", year: "2020", clue: "Time inversion spy thriller with an orange briefcase" },
        { answer: "MEMENTO", year: "2000", clue: "Memory-loss protagonist who tattoos his body" },
        { answer: "THE DARK KNIGHT", year: "2008", clue: "Batman's most acclaimed film with the Joker" },
        { answer: "INTERSTELLAR", year: "2014", clue: "Epic space odyssey through a wormhole near Saturn" },
        { answer: "DUNKIRK", year: "2017", clue: "WWII evacuation told across three timelines" },
        { answer: "OPPENHEIMER", year: "2023", clue: "Biopic of the 'father of the atomic bomb'" },
        { answer: "THE PRESTIGE", year: "2006", clue: "Rivalry between magicians with a Tesla twist" },
        { answer: "INSOMNIA", year: "2002", clue: "Al Pacino detective film set in perpetual daylight" }
    ];

    // ── Utility: Shuffle array ──
    function shuffle(arr) {
        const a = [...arr];
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    // ── Utility: HTML Entity Decoder (OpenTDB returns HTML-encoded text) ──
    function decodeHtml(html) {
        if (!html) return '';
        return new DOMParser().parseFromString(html, 'text/html').body.textContent;
    }

    // ── Utility: Safe per-browser storage (may be unavailable in private mode) ──
    function storeGet(storage, key) {
        try { return storage.getItem(key); } catch (_) { return null; }
    }
    function storeSet(storage, key, value) {
        try { storage.setItem(key, value); } catch (_) { /* ignore */ }
    }
    function storeRemove(storage, key) {
        try { storage.removeItem(key); } catch (_) { /* ignore */ }
    }

    // Questions already shown in this browser, so new rounds don't repeat them.
    function loadSeen() {
        try { return new Set(JSON.parse(storeGet(localStorage, SEEN_KEY) || '[]')); }
        catch (_) { return new Set(); }
    }
    function markSeen(question) {
        const seen = loadSeen();
        seen.delete(question);
        seen.add(question);
        storeSet(localStorage, SEEN_KEY, JSON.stringify([...seen].slice(-SEEN_LIMIT)));
    }

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function setText(id, value) {
        const node = document.getElementById(id);
        if (node) node.textContent = value;
    }

    // ================================================================
    // CINEMA ARCADE ENGINE
    // ================================================================
    const CinemaArcadeEngine = {
        currentGame: null,
        gameStates: {},
        _gen: 0,
        _interval: null,
        _timeouts: [],

        // Stops every running countdown and pending callback. Bumping the
        // generation also invalidates async work (fetches) from a previous game.
        stopAll() {
            this._gen++;
            if (this._interval) clearInterval(this._interval);
            this._interval = null;
            this._timeouts.forEach(clearTimeout);
            this._timeouts = [];
        },

        // setTimeout that is cancelled when the game changes or the lounge closes.
        later(fn, ms) {
            const gen = this._gen;
            const id = setTimeout(() => { if (gen === this._gen) fn(); }, ms);
            this._timeouts.push(id);
        },

        stopCountdown() {
            if (this._interval) clearInterval(this._interval);
            this._interval = null;
        },

        // Deadline-based countdown: the display stays accurate even if the
        // browser throttles timers, and only one countdown runs at a time.
        startCountdown(seconds, timerId, onExpire) {
            this.stopCountdown();
            const gen = this._gen;
            const deadline = Date.now() + seconds * 1000;
            let shown = seconds;
            setText(timerId, shown);
            this._interval = setInterval(() => {
                if (gen !== this._gen) return;
                const remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
                if (remaining !== shown) {
                    shown = remaining;
                    setText(timerId, shown);
                }
                if (remaining <= 0) {
                    this.stopCountdown();
                    onExpire();
                }
            }, 200);
        },

        // ── Show Lounge Overlay ──
        showLounge() {
            const overlay = document.getElementById('cinema-lounge-overlay');
            if (overlay) overlay.style.display = 'flex';
            this.switchGame('trivia-blitz');
        },

        // ── Hide Lounge Overlay ──
        hideLounge() {
            this.stopAll();
            const overlay = document.getElementById('cinema-lounge-overlay');
            if (overlay) overlay.style.display = 'none';
        },

        // ── Switch Game Tab ──
        switchGame(gameName) {
            document.querySelectorAll('.lounge-tab').forEach(t => {
                t.classList.toggle('active', t.dataset.game === gameName);
            });
            this.showGame(gameName);
        },

        // ── Render a Game ──
        showGame(gameName) {
            this.stopAll();
            this.currentGame = gameName;
            const container = document.getElementById('arcade-game-container');
            if (!container) return;
            container.innerHTML = '';

            if (gameName === 'trivia-blitz') this.initTriviaBlitz(container);
            else if (gameName === 'title-scramble') this.initScrambleGame(container);
        },

        // ── Trivia Blitz (OpenTDB, offline bank as fallback) ──
        initTriviaBlitz(container) {
            const state = { questions: [], currentQ: 0, score: 0 };
            this.gameStates.trivia = state;

            container.innerHTML = `
                <div class="lounge-game-header">
                    <h3>❓ Trivia Blitz</h3>
                    <div class="lounge-score">Score: <span id="trivia-score">0</span> · Q: <span id="trivia-qnum">0</span>/<span id="trivia-qtotal">${ROUND_SIZE}</span> · Time: <span id="trivia-timer">${TRIVIA_SECONDS}</span>s</div>
                </div>
                <div id="trivia-status" class="trivia-loading">Loading questions...</div>
                <div id="trivia-content" class="trivia-content" style="display:none;"></div>
            `;

            const gen = this._gen;
            this.loadTriviaRound().then(questions => {
                if (gen !== this._gen) return;
                state.questions = questions;
                const statusEl = document.getElementById('trivia-status');
                const contentEl = document.getElementById('trivia-content');
                if (statusEl) statusEl.style.display = 'none';
                if (contentEl) contentEl.style.display = 'block';
                setText('trivia-qtotal', questions.length);
                this.renderTrivia(state);
            });
        },

        // Builds a round of unseen questions: OpenTDB first, topped up from
        // the offline bank if OpenTDB is unreachable or returns repeats.
        async loadTriviaRound() {
            const seen = loadSeen();
            const round = [];
            const inRound = new Set();
            const add = (q) => {
                if (round.length < ROUND_SIZE && !seen.has(q.question) && !inRound.has(q.question)) {
                    inRound.add(q.question);
                    round.push(q);
                }
            };

            try {
                (await this.fetchOpenTdb()).forEach(add);
            } catch (err) {
                console.warn('OpenTDB unavailable, using offline trivia:', err);
            }

            const offline = shuffle(triviaBank).map(q => ({ question: q.q, correct: q.c, incorrect: q.i }));
            offline.forEach(add);

            // Every offline question has been seen: start the offline bank over.
            if (round.length === 0) offline.slice(0, ROUND_SIZE).forEach(q => round.push(q));
            return round;
        },

        async fetchOpenTdb() {
            let token = storeGet(localStorage, TOKEN_KEY);
            if (!token) token = await this.requestToken();

            for (let attempt = 0; attempt < 3; attempt++) {
                const url = token ? `${OPEN_TDB_URL}&token=${encodeURIComponent(token)}` : OPEN_TDB_URL;
                const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
                const data = await res.json();

                if (data.response_code === 0) {
                    return (data.results || []).map(r => ({
                        question: decodeHtml(r.question),
                        correct: decodeHtml(r.correct_answer),
                        incorrect: (r.incorrect_answers || []).map(decodeHtml)
                    }));
                }
                if (data.response_code === 3) {
                    // Token expired (6h idle) or unknown: get a fresh one
                    storeRemove(localStorage, TOKEN_KEY);
                    token = await this.requestToken();
                } else if (data.response_code === 4 && token) {
                    // Token has served every question in the category: reset it
                    await fetch(`${OPEN_TDB_TOKEN_URL}?command=reset&token=${encodeURIComponent(token)}`, { signal: AbortSignal.timeout(5000) });
                } else if (data.response_code === 5) {
                    // Rate limited (1 request per 5s per IP)
                    await new Promise(r => setTimeout(r, 5200));
                } else {
                    throw new Error(`OpenTDB response_code ${data.response_code}`);
                }
            }
            throw new Error('OpenTDB retries exhausted');
        },

        async requestToken() {
            try {
                const res = await fetch(`${OPEN_TDB_TOKEN_URL}?command=request`, { signal: AbortSignal.timeout(5000) });
                const data = await res.json();
                if (data.response_code === 0 && data.token) {
                    storeSet(localStorage, TOKEN_KEY, data.token);
                    return data.token;
                }
            } catch (err) {
                console.warn('OpenTDB token request failed:', err);
            }
            return null;
        },

        renderTrivia(state) {
            const content = document.getElementById('trivia-content');
            if (!content) return;

            const q = state.questions[state.currentQ];
            if (!q) {
                this.showTriviaComplete(state);
                return;
            }
            markSeen(q.question);
            setText('trivia-qnum', state.currentQ + 1);

            const options = shuffle([...q.incorrect, q.correct]);
            content.innerHTML = '';
            content.appendChild(el('div', 'trivia-question', q.question));
            const list = el('div', 'trivia-options');
            options.forEach((opt, idx) => {
                const btn = el('button', 'trivia-opt-btn', `${String.fromCharCode(65 + idx)}. ${opt}`);
                btn.dataset.correct = String(opt === q.correct);
                btn.addEventListener('click', () => this.answerTrivia(state, btn));
                list.appendChild(btn);
            });
            content.appendChild(list);

            this.startCountdown(TRIVIA_SECONDS, 'trivia-timer', () => this.answerTrivia(state, null));
        },

        // btn is null when the timer runs out.
        answerTrivia(state, btn) {
            this.stopCountdown();
            const buttons = document.querySelectorAll('#trivia-content .trivia-opt-btn');
            if (!buttons.length || buttons[0].disabled) return;
            buttons.forEach(b => {
                b.disabled = true;
                if (b.dataset.correct === 'true') b.classList.add('correct');
            });
            if (btn) {
                btn.classList.add('selected');
                if (btn.dataset.correct === 'true') {
                    state.score++;
                    setText('trivia-score', state.score);
                } else {
                    btn.classList.add('incorrect');
                }
            }
            this.later(() => {
                state.currentQ++;
                this.renderTrivia(state);
            }, 1500);
        },

        showTriviaComplete(state) {
            const content = document.getElementById('trivia-content');
            if (!content) return;
            content.innerHTML = `
                <div class="trivia-complete">
                    <div class="trivia-final-score">Final Score: ${state.score} / ${state.questions.length}</div>
                    <button class="lounge-restart-btn" onclick="showLoungeGame('trivia-blitz')">Play Again</button>
                </div>
            `;
        },

        // ── Title Scramble ──
        initScrambleGame(container) {
            const state = { questions: shuffle(scrambleBank), currentQ: 0, score: 0 };
            this.gameStates.scramble = state;

            container.innerHTML = `
                <div class="lounge-game-header">
                    <h3>🔤 Title Scramble — Film Anagrams</h3>
                    <div class="lounge-score">Score: <span id="scramble-score">0</span> · Q: <span id="scramble-qnum">1</span>/${state.questions.length} · Time: <span id="scramble-timer">${SCRAMBLE_SECONDS}</span>s</div>
                </div>
                <div id="scramble-content" class="trivia-content"></div>
            `;

            this.renderScramble(state);
        },

        renderScramble(state) {
            const content = document.getElementById('scramble-content');
            if (!content) return;

            const q = state.questions[state.currentQ];
            if (!q) {
                this.showScrambleComplete(state);
                return;
            }

            const plain = q.answer.replace(/ /g, '');
            let letters = shuffle(plain.split('')).join('');
            if (letters === plain && plain.length > 1) letters = plain.slice(1) + plain[0];

            setText('scramble-qnum', state.currentQ + 1);
            setText('scramble-score', state.score);

            content.innerHTML = `
                <div class="scramble-clue">
                    <div class="scramble-clue-row"><strong>Year:</strong> ${q.year}</div>
                    <div class="scramble-clue-row"><strong>Clue:</strong> ${q.clue}</div>
                </div>
                <div class="scramble-letters">${letters.split('').join(' ')}</div>
                <div class="scramble-input-wrap">
                    <input type="text" id="scramble-answer" class="scramble-answer-input" placeholder="Type the film title..." autocomplete="off">
                    <button class="scramble-submit" id="scramble-submit">Submit</button>
                </div>
            `;

            const input = document.getElementById('scramble-answer');
            const submit = () => this.checkScrambleAnswer(state);
            if (input) {
                input.addEventListener('keyup', e => { if (e.key === 'Enter') submit(); });
                input.focus();
            }
            document.getElementById('scramble-submit')?.addEventListener('click', submit);

            this.startCountdown(SCRAMBLE_SECONDS, 'scramble-timer', () => this.finishScramble(state, '⏰ Time\'s up!'));
        },

        checkScrambleAnswer(state) {
            const input = document.getElementById('scramble-answer');
            if (!input) return;
            const normalize = s => s.trim().toLowerCase().replace(/\s+/g, ' ');
            const userAnswer = normalize(input.value);
            if (!userAnswer) return;

            const q = state.questions[state.currentQ];
            if (userAnswer === normalize(q.answer)) {
                state.score++;
                this.finishScramble(state, '✅ Correct!', true);
            } else {
                this.finishScramble(state, '❌ Wrong!');
            }
        },

        finishScramble(state, message, correct = false) {
            this.stopCountdown();
            const q = state.questions[state.currentQ];
            const content = document.getElementById('scramble-content');
            if (content) {
                const result = el('div', correct ? 'scramble-correct' : 'scramble-incorrect',
                    `${message} ${correct ? '' : 'The answer was '}${q.answer} (${q.year})`);
                content.innerHTML = '';
                content.appendChild(result);
            }
            setText('scramble-score', state.score);
            this.later(() => {
                state.currentQ++;
                this.renderScramble(state);
            }, 1500);
        },

        showScrambleComplete(state) {
            const content = document.getElementById('scramble-content');
            if (!content) return;
            content.innerHTML = `
                <div class="trivia-complete">
                    <div class="trivia-final-score">Scramble Complete! Score: ${state.score} / ${state.questions.length}</div>
                    <button class="lounge-restart-btn" onclick="showLoungeGame('title-scramble')">Play Again</button>
                </div>
            `;
        }
    };

    // ── Global functions for HTML onclick ──
    window.showLoungeGame = function (gameName) {
        CinemaArcadeEngine.switchGame(gameName);
    };

    window.CinemaArcadeEngine = CinemaArcadeEngine;
})();
