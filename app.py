import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="MovRec",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# PREMIUM CSS — CINEMATIC DARK THEME
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --red:    #E8232A;
    --red-dim: #9B1217;
    --gold:   #F0A500;
    --bg0:    #0A0A0C;
    --bg1:    #111116;
    --bg2:    #18181F;
    --bg3:    #21212B;
    --border: rgba(255,255,255,0.07);
    --text:   #F0EDE8;
    --muted:  #8A8790;
    --sans:   'DM Sans', sans-serif;
    --display:'Bebas Neue', sans-serif;
}

/* ── App shell ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: var(--display);
    font-size: 2.4rem;
    letter-spacing: 3px;
    color: var(--red);
    line-height: 1;
    margin-bottom: 0.25rem;
}
.sidebar-tagline {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.75rem;
}
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
}

/* ── Radio nav ── */
[data-testid="stRadio"] label { font-size: 0.85rem !important; color: var(--muted) !important; }
[data-testid="stRadio"] div[data-baseweb="radio"] > div { background: transparent !important; }

/* ── Selectbox / text input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] svg, [data-testid="stTextInput"] svg { fill: var(--muted) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    background: var(--red-dim) !important;
    transform: translateY(-1px) !important;
}

/* ── Page header ── */
.page-header {
    font-family: var(--display);
    font-size: 3rem;
    letter-spacing: 4px;
    color: var(--text);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.page-sub {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2rem;
}
.section-rule {
    height: 1px;
    background: linear-gradient(to right, var(--red), transparent);
    margin: 1rem 0 1.75rem;
}

/* ── Movie card ── */
.movie-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.25s;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-4px);
    border-color: var(--red-dim);
}
.movie-card-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.movie-card-poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: var(--bg3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    color: var(--border);
}
.movie-card-body {
    padding: 0.75rem 0.85rem 1rem;
}
.movie-card-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text);
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-card-genre {
    font-size: 0.72rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-card-score {
    display: inline-block;
    margin-top: 0.4rem;
    background: var(--red-dim);
    color: #FFD0D0;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── List-style card (for search & similar) ── */
.list-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.list-card:hover { border-color: rgba(232,35,42,0.4); }
.list-card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.3rem;
}
.list-card-title {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
}
.list-card-score {
    font-size: 0.75rem;
    color: var(--gold);
    font-weight: 600;
}
.list-card-genre {
    font-size: 0.78rem;
    color: var(--red);
    margin-bottom: 0.4rem;
}
.list-card-desc {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
}

/* ── Hero section ── */
.hero-wrap {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.hero-img {
    width: 100%;
    height: 420px;
    object-fit: cover;
    display: block;
    filter: brightness(0.45);
}
.hero-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 2.5rem 2rem 2rem;
    background: linear-gradient(to top, rgba(10,10,12,0.98) 0%, transparent 100%);
}
.hero-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: var(--display);
    font-size: 2.8rem;
    letter-spacing: 3px;
    color: var(--text);
    line-height: 1;
    margin-bottom: 0.5rem;
}
.hero-genre {
    font-size: 0.8rem;
    color: var(--muted);
}

/* ── MOTD ── */
.motd-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    height: 100%;
}
.motd-badge {
    display: inline-block;
    background: var(--red);
    color: #fff;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}
.motd-title {
    font-family: var(--display);
    font-size: 1.9rem;
    letter-spacing: 2px;
    color: var(--text);
    margin-bottom: 0.35rem;
}
.motd-genre { font-size: 0.8rem; color: var(--red); margin-bottom: 0.75rem; }
.motd-desc { font-size: 0.85rem; color: var(--muted); line-height: 1.6; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.stat-label { font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem; }
.stat-value { font-family: var(--display); font-size: 1.8rem; letter-spacing: 2px; color: var(--text); }
.stat-accent { color: var(--red); }

/* ── Mood pills ── */
.mood-grid { display: flex; flex-wrap: wrap; gap: 10px; margin: 1rem 0 1.5rem; }
.mood-pill {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 40px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.2s;
    font-family: var(--sans);
}
.mood-pill:hover { border-color: var(--red); color: var(--text); }
.mood-pill.active { background: var(--red); border-color: var(--red); color: #fff; }

/* ── Section title ── */
.section-title {
    font-family: var(--display);
    font-size: 1.5rem;
    letter-spacing: 3px;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.section-desc { font-size: 0.8rem; color: var(--muted); margin-bottom: 1.25rem; }

/* ── Footer ── */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 1px;
}

/* ── Streamlit overrides ── */
.stMarkdown p { color: var(--text) !important; }
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
div[data-testid="column"] { padding: 0 6px !important; }

/* Matplotlib dark bg */
.stPlotlyChart, .stPyplotGlobalUse { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    df = joblib.load('movies.pkl')
    tfidf = joblib.load('tfidf.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    df['combined_features'] = df.get('combined_features', df.get('description', ''))
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return df, tfidf, tfidf_matrix, indices

df, tfidf, tfidf_matrix, indices = load_model()


# =============================
# TMDB API
# =============================
API_KEY = "0758644da67e27b71fac69a53fab875e"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        data = requests.get(url, timeout=4).json()
        poster = data['results'][0]['poster_path']
        return "https://image.tmdb.org/t/p/w342/" + poster
    except:
        return None


# =============================
# MOOD CONFIG
# =============================
mood_config = {
    "Happy":       "comedy family fun",
    "Emotional":   "drama emotional heartbreaking",
    "Action":       "action thriller adventure",
    "Romantic":    "romance love relationship",
    "Sci-Fi":      "science fiction space future",
    "Chill":       "documentary calm slow",
    "Horror":      "horror scary suspense",
    "Drama":       "drama serious intense",
}


# =============================
# HELPERS
# =============================
def recommend_by_mood(mood, top_n=10):
    keywords = mood_config[mood]
    user_vec = tfidf.transform([keywords])
    sim = cosine_similarity(user_vec, tfidf_matrix)[0]
    scores = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[:top_n]
    return scores

def recommend_by_title(title, top_n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scores = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return scores

def render_movie_grid(index_score_pairs, cols=5, show_score=False):
    """Render a responsive poster grid."""
    rows = [index_score_pairs[i:i+cols] for i in range(0, len(index_score_pairs), cols)]
    for row in rows:
        col_list = st.columns(cols)
        for j, (idx, score) in enumerate(row):
            row_obj = df.iloc[idx]
            title   = row_obj.get('title', '—')
            genre   = row_obj.get('listed_in', '')
            poster  = fetch_poster(title)

            score_html = f"<span class='movie-card-score'>{round(score*100)}% match</span>" if show_score else ""

            if poster:
                poster_html = f"<img class='movie-card-poster' src='{poster}' alt='{title}' />"
            else:
                poster_html = "<div class='movie-card-poster-placeholder'>🎬</div>"

            genre_short = genre[:40] + "…" if len(genre) > 40 else genre

            with col_list[j]:
                st.markdown(f"""
                <div class='movie-card'>
                    {poster_html}
                    <div class='movie-card-body'>
                        <div class='movie-card-title' title='{title}'>{title}</div>
                        <div class='movie-card-genre'>{genre_short}</div>
                        {score_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_list_cards(index_score_pairs):
    """Render detailed list cards with description."""
    for idx, score in index_score_pairs:
        row = df.iloc[idx]
        title = row.get('title', '—')
        genre = row.get('listed_in', '')
        desc  = row.get('description', '')[:180]
        if desc: desc += "…"
        score_html = f"<span class='list-card-score'>{round(score*100)}% match</span>" if score > 0 else ""
        st.markdown(f"""
        <div class='list-card'>
            <div class='list-card-header'>
                <span class='list-card-title'>{title}</span>
                {score_html}
            </div>
            <div class='list-card-genre'>{genre}</div>
            <div class='list-card-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# =============================
# SESSION STATE
# =============================
if 'motd' not in st.session_state:
    st.session_state.motd = df.sample(1).iloc[0]
if 'random_pick' not in st.session_state:
    st.session_state.random_pick = None


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>MOVREC</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Netflix Recommendation Engine</div>", unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["Home", "Mood", "Cari Film Sejenis", "Surprise Me!", "Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    total    = len(df)
    n_movies = len(df[df.get('type', pd.Series(['Movie']*len(df))) == 'Movie']) if 'type' in df.columns else "—"
    n_shows  = len(df[df.get('type', pd.Series(['TV Show']*len(df))) == 'TV Show']) if 'type' in df.columns else "—"

    st.markdown(f"""
    <div style='font-size:0.75rem; color:var(--muted);'>
        <div style='margin-bottom:0.5rem;'><b style='color:var(--text);'>{total:,}</b> judul tersedia</div>
        <div style='margin-bottom:0.5rem;'><b style='color:var(--text);'>{n_movies}</b> Movies</div>
        <div>📺 <b style='color:var(--text);'>{n_shows}</b> TV Shows</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:var(--muted); line-height:1.7;'>
        <div style='margin-bottom:0.3rem; color:var(--text); font-size:0.78rem; font-weight:600;'>Kelompok</div>
        Data Science Beginner<br>
        • Nayla Dwinta P. M.<br>
        • Lathisya Sheza A.<br>
        • Regina Juliyanti M.
    </div>
    """, unsafe_allow_html=True)


# =============================
# HOME
# =============================
if menu == "Home":
    motd = st.session_state.motd
    motd_title  = motd.get('title', 'Film Hari Ini')
    motd_genre  = motd.get('listed_in', '')
    motd_desc   = motd.get('description', '')
    motd_poster = fetch_poster(motd_title)

    # Hero
    if motd_poster:
        st.markdown(f"""
        <div class='hero-wrap'>
            <img class='hero-img' src='{motd_poster}' alt='{motd_title}'/>
            <div class='hero-overlay'>
                <div class='hero-label'>✦ Pilihan Hari Ini</div>
                <div class='hero-title'>{motd_title.upper()}</div>
                <div class='hero-genre'>{motd_genre}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:var(--bg2);border-radius:16px;padding:3rem 2rem;margin-bottom:2.5rem;'>
            <div class='hero-label'>✦ Pilihan Hari Ini</div>
            <div class='hero-title'>{motd_title.upper()}</div>
            <div class='hero-genre'>{motd_genre}</div>
        </div>
        """, unsafe_allow_html=True)

    # Featured row
    st.markdown("<div class='section-title'>TRENDING SEKARANG</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Konten populer dari database Netflix</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    featured = df.sample(10)
    pairs = [(i, 0.0) for i in featured.index]
    render_movie_grid(pairs, cols=5)

    # Quick shortcuts
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Rekomendasi via Mood", use_container_width=True):
            st.session_state['nav'] = "Mood"
            st.rerun()
    with c2:
        if st.button("Cari Film Sejenis", use_container_width=True):
            st.session_state['nav'] = "Cari Film Sejenis"
            st.rerun()
    with c3:
        if st.button("Surprise Me!", use_container_width=True):
            st.session_state['nav'] = "Surprise Me!"
            st.rerun()


# =============================
# MOOD
# =============================
elif menu == "Mood":
    st.markdown("<div class='page-header'>MOOD MATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Sistem mencarikan film yang paling pas dengan vibes kamu</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    selected_mood = st.selectbox(
        "Pilih suasana hati kamu:",
        list(mood_config.keys()),
        label_visibility="visible"
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run = st.button("Temukan Film →", use_container_width=True)

    if run:
        with st.spinner("Mencari film terbaik untukmu…"):
            results = recommend_by_mood(selected_mood, top_n=10)

        st.markdown(f"<br><div class='section-title'>HASIL UNTUK {selected_mood.upper()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

        # Grid view
        render_movie_grid(results, cols=5, show_score=True)

        # List view with descriptions
        st.markdown("<br><div class='section-title'>DETAIL REKOMENDASI</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        render_list_cards(results)


# =============================
# CARI FILM SEJENIS
# =============================
elif menu == "Cari Film Sejenis":
    st.markdown("<div class='page-header'>FILM SEJENIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Masukkan judul favoritmu, algoritma Cosine Similarity akan mencari kembarannya</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    query = st.text_input("Cari judul film…", placeholder="Contoh: Inception, Transformer, The Crown…")

    # Live search suggestions
    if query:
        matched = df[df['title'].str.contains(query, case=False, na=False)]['title'].head(6).tolist()

        if matched:
            st.markdown("<div style='font-size:0.78rem;color:var(--muted);margin-bottom:0.5rem;'>Pilih judul yang tepat:</div>", unsafe_allow_html=True)
            chosen_title = st.selectbox("", matched, label_visibility="collapsed")

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                search = st.button("Cari Film Sejenis →", use_container_width=True)

            if search:
                with st.spinner("Menganalisis kemiripan…"):
                    results = recommend_by_title(chosen_title, top_n=10)

                if results:
                    st.markdown(f"<br><div class='section-title'>MIRIP DENGAN: {chosen_title.upper()}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
                    render_movie_grid(results, cols=5, show_score=True)

                    st.markdown("<br><div class='section-title'>DETAIL</div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
                    render_list_cards(results)
                else:
                    st.error("Maaf, tidak ditemukan film serupa untuk judul tersebut.")
        else:
            st.markdown("<div style='color:var(--muted); font-size:0.85rem;'>Tidak ada judul yang cocok. Coba kata kunci lain.</div>", unsafe_allow_html=True)


# =============================
# SURPRISE ME!
# =============================
elif menu == "Surprise Me!":
    st.markdown("<div class='page-header'>SURPRISE ME!</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Bingung mau nonton apa? Biarkan sistem yang memilih</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        motd = st.session_state.motd
        motd_title  = motd.get('title', '—')
        motd_genre  = motd.get('listed_in', '')
        motd_desc   = motd.get('description', '')
        motd_poster = fetch_poster(motd_title)

        poster_html = f"<img src='{motd_poster}' style='width:100%;border-radius:10px;margin-bottom:1rem;'/>" if motd_poster else ""

        st.markdown(f"""
        <div class='motd-card'>
            <div class='motd-badge'>✦ Movie of The Day</div>
            {poster_html}
            <div class='motd-title'>{motd_title.upper()}</div>
            <div class='motd-genre'>{motd_genre}</div>
            <div class='motd-desc'>{motd_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:1.5rem;'>
            <div class='motd-badge'>Random Roulette</div>
            <div style='font-size:0.85rem;color:var(--muted);margin-bottom:1.25rem;'>
                Klik tombol di bawah untuk mendapatkan pilihan film acak dari ribuan judul Netflix.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Putar Roulette!", use_container_width=True):
            st.session_state.random_pick = df.sample(1).iloc[0]

        if st.session_state.random_pick is not None:
            r = st.session_state.random_pick
            r_title  = r.get('title', '—')
            r_genre  = r.get('listed_in', '')
            r_desc   = r.get('description', '')
            r_poster = fetch_poster(r_title)

            poster_html = f"<img src='{r_poster}' style='width:100%;border-radius:10px;margin:1rem 0;'/>" if r_poster else ""
            st.markdown(f"""
            <div style='background:var(--bg3);border-radius:12px;padding:1.25rem;margin-top:0.75rem;'>
                {poster_html}
                <div style='font-family:var(--display);font-size:1.6rem;letter-spacing:2px;color:var(--text);'>{r_title.upper()}</div>
                <div style='font-size:0.8rem;color:var(--red);margin:0.3rem 0 0.6rem;'>{r_genre}</div>
                <div style='font-size:0.82rem;color:var(--muted);line-height:1.6;'>{r_desc}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================
# ANALYTICS
# =============================
elif menu == "Analytics":
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    st.markdown("<div class='page-header'>ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Eksplorasi distribusi dataset Netflix secara visual</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    # Stats row
    total = len(df)
    n_movies = len(df[df['type'] == 'Movie']) if 'type' in df.columns else "—"
    n_shows  = len(df[df['type'] == 'TV Show']) if 'type' in df.columns else "—"
    n_genres = df['listed_in'].str.split(', ').explode().nunique() if 'listed_in' in df.columns else "—"

    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-card'>
            <div class='stat-label'>Total Judul</div>
            <div class='stat-value'>{total:,}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Movies</div>
            <div class='stat-value'><span class='stat-accent'>{n_movies}</span></div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>TV Shows</div>
            <div class='stat-value'>{n_shows}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Genre Unik</div>
            <div class='stat-value'>{n_genres}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#18181F',
        'axes.facecolor':   '#18181F',
        'axes.edgecolor':   '#2A2A35',
        'axes.labelcolor':  '#8A8790',
        'xtick.color':      '#8A8790',
        'ytick.color':      '#8A8790',
        'text.color':       '#F0EDE8',
        'grid.color':       '#2A2A35',
        'grid.linestyle':   '--',
        'grid.alpha':       0.4,
    })

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>MOVIE VS TV SHOW</div>", unsafe_allow_html=True)
        if 'type' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df['type'].value_counts()
            colors = ['#E8232A', '#F0A500']
            bars = ax.bar(counts.index, counts.values, color=colors, width=0.45, zorder=3)
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                        f'{bar.get_height():,}', ha='center', va='bottom',
                        fontsize=10, color='#F0EDE8', fontweight='bold')
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Kolom 'type' tidak tersedia.")

    with col2:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>TREN RILIS PER TAHUN</div>", unsafe_allow_html=True)
        if 'release_year' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            year_data = df['release_year'].value_counts().sort_index()
            ax2.plot(year_data.index, year_data.values, color='#E8232A', linewidth=2, zorder=3)
            ax2.fill_between(year_data.index, year_data.values, alpha=0.12, color='#E8232A')
            ax2.yaxis.grid(True, zorder=0)
            ax2.set_axisbelow(True)
            ax2.spines[:].set_visible(False)
            ax2.tick_params(length=0)
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Kolom 'release_year' tidak tersedia.")

    st.markdown("<br><div class='section-title' style='font-size:1.1rem;'>TOP 10 GENRE TERBESAR</div>", unsafe_allow_html=True)
    if 'listed_in' in df.columns:
        genre_counts = df['listed_in'].str.split(', ').explode().value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        bars = ax3.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                        color='#E8232A', height=0.55, zorder=3)
        ax3.xaxis.grid(True, zorder=0)
        ax3.set_axisbelow(True)
        ax3.spines[:].set_visible(False)
        ax3.tick_params(length=0)
        for bar in bars:
            ax3.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                     f'{int(bar.get_width()):,}', va='center',
                     fontsize=9, color='#8A8790')
        fig3.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("Kolom 'listed_in' tidak tersedia.")


# =============================
# FOOTER
# =============================
st.markdown("""
<div class='footer'>
    MOVREC · Content-Based Filtering · NLP · TF-IDF + Cosine Similarity<br>
    <span style='opacity:0.5;'>Data Science Beginner · 2024</span>
</div>
""", unsafe_allow_html=True)
