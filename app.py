import streamlit as st
import joblib
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovRec",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --red: #E8232A; --red-dim: #9B1217; --gold: #F0A500;
    --bg0: #0A0A0C; --bg1: #111116; --bg2: #18181F; --bg3: #21212B;
    --border: rgba(255,255,255,0.07); --border2: rgba(255,255,255,0.12);
    --text: #F0EDE8; --muted: #8A8790;
    --sans: 'DM Sans', sans-serif; --display: 'Bebas Neue', sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

.sidebar-logo    { font-family: var(--display); font-size: 2.4rem; letter-spacing: 3px; color: var(--red); line-height: 1; margin-bottom: .25rem; }
.sidebar-tagline { font-size: .7rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 1.75rem; }
.sidebar-divider { height: 1px; background: var(--border); margin: 1rem 0; }

/* ── INPUTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
}
.stButton > button {
    background: var(--red) !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: var(--sans) !important; font-weight: 600 !important;
    font-size: .85rem !important; letter-spacing: .5px !important;
    padding: .55rem 1.4rem !important; transition: background .2s, transform .15s !important;
}
.stButton > button:hover { background: var(--red-dim) !important; transform: translateY(-1px) !important; }

/* ── GRID / CARDS ── */
.grid-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 14px; margin-bottom: 14px;
}
.mc {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    text-decoration: none; display: block;
    transition: transform .25s, border-color .25s;
}
.mc:hover { transform: translateY(-4px); border-color: var(--red-dim); }
.mc img   { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mc-nop   {
    width: 100%; aspect-ratio: 2/3; background: var(--bg3);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; color: var(--muted);
    font-family: var(--display); letter-spacing: 2px;
}
.mc-body  { padding: .75rem .85rem 1rem; }
.mc-title { font-weight: 600; font-size: .88rem; color: var(--text); margin-bottom: .3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mc-genre { font-size: .72rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mc-score { display: inline-block; margin-top: .4rem; background: var(--red-dim); color: #FFD0D0; font-size: .68rem; font-weight: 600; padding: .15rem .5rem; border-radius: 20px; letter-spacing: .5px; }
.mc-type  { display: inline-block; margin-top: .4rem; margin-left: 4px; background: var(--bg3); color: var(--muted); font-size: .65rem; font-weight: 600; padding: .15rem .5rem; border-radius: 20px; letter-spacing: .5px; border: 1px solid var(--border2); }

/* ── LIST CARDS ── */
.lc {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: .75rem;
    display: flex; gap: 1rem; align-items: flex-start;
    transition: border-color .2s; text-decoration: none;
}
.lc:hover { border-color: rgba(232,35,42,.45); }
.lc-poster { width: 64px; min-width: 64px; border-radius: 6px; aspect-ratio: 2/3; object-fit: cover; }
.lc-nop    { width: 64px; min-width: 64px; border-radius: 6px; aspect-ratio: 2/3; background: var(--bg3); display: flex; align-items: center; justify-content: center; font-size: .7rem; color: var(--muted); }
.lc-content { flex: 1; min-width: 0; }
.lc-top     { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: .25rem; }
.lc-title   { font-weight: 600; font-size: .95rem; color: var(--text); }
.lc-score   { font-size: .75rem; color: var(--gold); font-weight: 600; white-space: nowrap; }
.lc-genre   { font-size: .75rem; color: var(--red); margin-bottom: .35rem; }
.lc-desc    { font-size: .81rem; color: var(--muted); line-height: 1.55; }
.lc-actions { display: flex; gap: 8px; margin-top: .5rem; flex-wrap: wrap; }
.btn-t { font-size: .72rem; font-weight: 600; padding: .22rem .7rem; background: var(--red); color: #fff; border-radius: 4px; text-decoration: none; letter-spacing: .4px; }
.btn-t:hover { background: var(--red-dim); }
.btn-s { font-size: .72rem; font-weight: 600; padding: .22rem .7rem; background: var(--bg3); color: var(--muted); border-radius: 4px; text-decoration: none; letter-spacing: .4px; border: 1px solid var(--border); }
.btn-s:hover { color: var(--text); }

/* ── HERO ── */
.hero-wrap { position: relative; border-radius: 16px; overflow: hidden; margin-bottom: 2.5rem; }
.hero-img  { width: 100%; height: 420px; object-fit: cover; display: block; filter: brightness(.4); }
.hero-overlay { position: absolute; bottom: 0; left: 0; right: 0; padding: 2.5rem 2rem 2rem; background: linear-gradient(to top, rgba(10,10,12,.98) 0%, transparent 100%); }
.hero-label { font-size: .7rem; letter-spacing: 3px; text-transform: uppercase; color: var(--red); margin-bottom: .4rem; }
.hero-title { font-family: var(--display); font-size: 2.8rem; letter-spacing: 3px; color: var(--text); line-height: 1; margin-bottom: .5rem; }
.hero-genre { font-size: .8rem; color: var(--muted); }

/* ── MOTD ── */
.motd-card  { background: var(--bg2); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; height: 100%; }
.motd-badge { display: inline-block; background: var(--red); color: #fff; font-size: .65rem; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; padding: .2rem .6rem; border-radius: 4px; margin-bottom: .75rem; }
.motd-title { font-family: var(--display); font-size: 1.9rem; letter-spacing: 2px; color: var(--text); margin-bottom: .35rem; }
.motd-genre { font-size: .8rem; color: var(--red); margin-bottom: .75rem; }
.motd-desc  { font-size: .85rem; color: var(--muted); line-height: 1.6; }

/* ── STAT ROW ── */
.stat-row  { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.stat-card { flex: 1; background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.1rem; }
.stat-label { font-size: .72rem; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: .35rem; }
.stat-value { font-family: var(--display); font-size: 1.8rem; letter-spacing: 2px; color: var(--text); }
.stat-accent { color: var(--red); }

/* ── TYPOGRAPHY ── */
.ph { font-family: var(--display); font-size: 3rem; letter-spacing: 4px; color: var(--text); line-height: 1; margin-bottom: .3rem; }
.ps { font-size: .8rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 2rem; }
.sr { height: 1px; background: linear-gradient(to right, var(--red), transparent); margin: .75rem 0 1.5rem; }
.st2 { font-family: var(--display); font-size: 1.5rem; letter-spacing: 3px; color: var(--text); margin-bottom: .2rem; }

/* ── MOOD PILLS ── */
.mood-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }
.mood-pill {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem 1rem; cursor: pointer; transition: all .2s;
    display: flex; align-items: center; gap: 10px;
}
.mood-pill:hover { border-color: var(--red); background: rgba(232,35,42,.06); }
.mood-pill .mp-icon { font-size: 1.6rem; }
.mood-pill .mp-label { font-weight: 600; font-size: .85rem; color: var(--text); }
.mood-pill .mp-sub   { font-size: .72rem; color: var(--muted); }

/* ── MISC ── */
.footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border); text-align: center; font-size: .75rem; color: var(--muted); letter-spacing: 1px; }
.stMarkdown p { color: var(--text) !important; }
div[data-testid="column"] { padding: 0 6px !important; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    d   = joblib.load("movies.pkl")
    tf  = joblib.load("tfidf.pkl")
    tfm = joblib.load("tfidf_matrix.pkl")
    idx = pd.Series(d.index, index=d["title"]).drop_duplicates()
    return d, tf, tfm, idx

df, tfidf, tfidf_matrix, indices = load_model()

# ─── TMDB API ─────────────────────────────────────────────────────────────────
API_KEY = "0758644da67e27b71fac69a53fab875e"

@st.cache_data(show_spinner=False)
def fetch_tmdb(title):
    try:
        url  = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={urllib.parse.quote(title)}"
        data = requests.get(url, timeout=4).json()
        r    = data["results"][0]
        poster = ("https://image.tmdb.org/t/p/w342" + r["poster_path"]) if r.get("poster_path") else None
        return poster, r["id"]
    except:
        return None, None

def trailer_url(title, tmdb_id=None):
    q = urllib.parse.quote(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"

def tmdb_page_url(tmdb_id):
    return f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else "#"


# ─── MOOD CONFIG ──────────────────────────────────────────────────────────────
MOOD_CONFIG = {
    "😊  Happy / Feel-Good":     "comedy family fun",
    "😢  Sad / Emotional":       "drama emotional heartbreaking",
    "⚡  Action / Thrilling":    "action thriller adventure",
    "💕  Romantic / Love":       "romance love relationship",
    "🚀  Sci-Fi / Mind-Bending": "science fiction space future",
    "☕  Chill / Documentary":   "documentary calm slow",
    "👻  Horror / Suspense":     "horror scary suspense",
    "🎭  Intense Drama":         "drama serious intense",
}


# ─── RECOMMENDATION FUNCTIONS ─────────────────────────────────────────────────
def rec_mood(mood_key, n=10):
    vec = tfidf.transform([MOOD_CONFIG[mood_key]])
    sim = cosine_similarity(vec, tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[:n]

def rec_title(title, n=10):
    if title not in indices:
        return []
    i   = indices[title]
    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:n+1]


# ─── RENDER HELPERS ───────────────────────────────────────────────────────────
def card_html(title, genre, content_type, poster, tmdb_id, score=None):
    gs  = (genre[:36] + "…") if len(genre) > 36 else genre
    ts  = title.replace("'", "&#39;").replace('"', '&quot;')
    tu  = trailer_url(title, tmdb_id)
    img = (f'<img src="{poster}" alt="{ts}" loading="lazy">'
           if poster else f'<div class="mc-nop">{title[:2].upper()}</div>')
    sc  = (f'<span class="mc-score">{round(score * 100)}% match</span>'
           if (score and score > 0) else "")
    tp  = (f'<span class="mc-type">{content_type}</span>'
           if content_type else "")
    return (
        f'<a class="mc" href="{tu}" target="_blank" rel="noopener">'
        f'{img}'
        f'<div class="mc-body">'
        f'<div class="mc-title" title="{ts}">{title}</div>'
        f'<div class="mc-genre">{gs}</div>'
        f'{sc}{tp}'
        f'</div></a>'
    )

def render_grid(pairs, show_score=False, cols=5):
    rows = [pairs[i:i+cols] for i in range(0, len(pairs), cols)]
    for row in rows:
        cards = ""
        for (idx, score) in row:
            ro   = df.iloc[idx]
            t    = str(ro.get("title", ""))
            g    = str(ro.get("listed_in", ""))
            ct   = str(ro.get("type", ""))
            p, tid = fetch_tmdb(t)
            cards += card_html(t, g, ct, p, tid, score if show_score else None)
        st.markdown(f'<div class="grid-row">{cards}</div>', unsafe_allow_html=True)

def render_list(pairs):
    html = ""
    for idx, score in pairs:
        ro  = df.iloc[idx]
        t   = str(ro.get("title", ""))
        g   = str(ro.get("listed_in", ""))
        d   = str(ro.get("description", ""))[:200]
        if d:
            d += "…"
        ts  = t.replace("'", "&#39;")
        p, tid = fetch_tmdb(t)
        tu  = trailer_url(t, tid)
        pu  = tmdb_page_url(tid)
        sc  = (f'<span class="lc-score">{round(score * 100)}% match</span>'
               if score > 0 else "")
        th  = (f'<img class="lc-poster" src="{p}" alt="{ts}" loading="lazy">'
               if p else '<div class="lc-nop">N/A</div>')
        html += (
            f'<div class="lc">{th}'
            f'<div class="lc-content">'
            f'<div class="lc-top"><span class="lc-title">{t}</span>{sc}</div>'
            f'<div class="lc-genre">{g}</div>'
            f'<div class="lc-desc">{d}</div>'
            f'<div class="lc-actions">'
            f'<a class="btn-t" href="{tu}" target="_blank" rel="noopener">▶ Trailer</a>'
            f'<a class="btn-s" href="{pu}" target="_blank" rel="noopener">TMDB</a>'
            f'</div></div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "motd" not in st.session_state:
    st.session_state.motd = df.sample(1).iloc[0]
if "random_pick" not in st.session_state:
    st.session_state.random_pick = None


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>MOVREC</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Netflix Recommendation Engine</div>", unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🏠  Home", "🎭  Mood", "🔍  Cari Film Sejenis", "🎲  Surprise Me", "📊  Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    total = len(df)
    nm    = len(df[df["type"] == "Movie"])   if "type" in df.columns else "—"
    ns    = len(df[df["type"] == "TV Show"]) if "type" in df.columns else "—"

    st.markdown(f"""
    <div style='font-size:.75rem;color:var(--muted);'>
        <div style='margin-bottom:.5rem;'><b style='color:var(--text);'>{total:,}</b> judul tersedia</div>
        <div style='margin-bottom:.5rem;'><b style='color:var(--text);'>{nm}</b> Movies</div>
        <div><b style='color:var(--text);'>{ns}</b> TV Shows</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:.72rem;color:var(--muted);line-height:1.7;'>
        <div style='margin-bottom:.3rem;color:var(--text);font-size:.78rem;font-weight:600;'>Kelompok</div>
        Data Science Beginner<br>
        Nayla Dwinta P. M.<br>
        Lathisya Sheza A.<br>
        Regina Juliyanti M.
    </div>""", unsafe_allow_html=True)


# ─── PAGES ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠  Home":
    motd   = st.session_state.motd
    mt     = str(motd.get("title", "Featured"))
    mg     = str(motd.get("listed_in", ""))
    mp, mid = fetch_tmdb(mt)
    tu     = trailer_url(mt, mid)

    # Hero Banner
    if mp:
        st.markdown(f"""
        <div class="hero-wrap">
            <img class="hero-img" src="{mp}" alt="{mt}"/>
            <div class="hero-overlay">
                <div class="hero-label">✦ Pilihan Hari Ini</div>
                <div class="hero-title">{mt.upper()}</div>
                <div class="hero-genre">{mg}</div>
                <a href="{tu}" target="_blank" rel="noopener"
                   style="display:inline-block;margin-top:1rem;background:var(--red);
                          color:#fff;padding:.5rem 1.4rem;border-radius:6px;
                          font-size:.82rem;font-weight:600;letter-spacing:.5px;
                          text-decoration:none;">▶ Tonton Trailer</a>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:var(--bg2);border-radius:16px;padding:3rem 2rem;margin-bottom:2.5rem;
                    border:1px solid var(--border);">
            <div class="hero-label">✦ Pilihan Hari Ini</div>
            <div class="hero-title">{mt.upper()}</div>
            <div class="hero-genre">{mg}</div>
            <a href="{tu}" target="_blank" rel="noopener"
               style="display:inline-block;margin-top:1rem;background:var(--red);
                      color:#fff;padding:.5rem 1.4rem;border-radius:6px;
                      font-size:.82rem;font-weight:600;text-decoration:none;">▶ Trailer</a>
        </div>""", unsafe_allow_html=True)

    # Trending
    st.markdown("<div class='st2'>TRENDING SEKARANG</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.8rem;color:var(--muted);margin-bottom:.5rem;'>Konten populer dari database Netflix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    render_grid([(i, 0.0) for i in df.sample(10).index])

    # Baru Ditambahkan
    st.markdown("<br><div class='st2'>BARU DITAMBAHKAN</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.8rem;color:var(--muted);margin-bottom:.5rem;'>Film & serial terbaru di dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    if "date_added" in df.columns:
        recent = df.dropna(subset=["date_added"]).copy()
        recent["date_added"] = pd.to_datetime(recent["date_added"], errors="coerce")
        recent = recent.sort_values("date_added", ascending=False).head(10)
        render_grid([(i, 0.0) for i in recent.index])
    else:
        render_grid([(i, 0.0) for i in df.sample(10).index])


# ══════════════════════════════════════════════════════════════════════════════
# MOOD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎭  Mood":
    st.markdown("<div class='ph'>MOOD MATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Sistem mencarikan film yang paling pas dengan vibes kamu</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    sel = st.selectbox("Pilih suasana hati kamu:", list(MOOD_CONFIG.keys()))
    cb, _ = st.columns([1, 4])
    with cb:
        run = st.button("Temukan Film", use_container_width=True)

    if run:
        with st.spinner("Mencari film terbaik…"):
            res = rec_mood(sel)
        st.markdown(f"<br><div class='st2'>HASIL: {sel.upper()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
        render_grid(res, show_score=True)
        st.markdown("<br><div class='st2'>DETAIL REKOMENDASI</div>", unsafe_allow_html=True)
        st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
        render_list(res)


# ══════════════════════════════════════════════════════════════════════════════
# CARI FILM SEJENIS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍  Cari Film Sejenis":
    st.markdown("<div class='ph'>FILM SEJENIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Masukkan judul favoritmu — cosine similarity mencarikan kembarannya</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    query = st.text_input("Cari judul film…", placeholder="Contoh: Inception, Transformers, The Crown…")
    if query:
        matched = df[df["title"].str.contains(query, case=False, na=False)]["title"].head(8).tolist()
        if matched:
            st.markdown("<div style='font-size:.78rem;color:var(--muted);margin-bottom:.5rem;'>Pilih judul yang tepat:</div>", unsafe_allow_html=True)
            chosen = st.selectbox("", matched, label_visibility="collapsed")
            cb, _ = st.columns([1, 4])
            with cb:
                go_btn = st.button("Cari Film Sejenis", use_container_width=True)
            if go_btn:
                with st.spinner("Menganalisis kemiripan…"):
                    res = rec_title(chosen)
                if res:
                    st.markdown(f"<br><div class='st2'>MIRIP DENGAN: {chosen.upper()}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
                    render_grid(res, show_score=True)
                    st.markdown("<br><div class='st2'>DETAIL</div>", unsafe_allow_html=True)
                    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
                    render_list(res)
                else:
                    st.error("Tidak ditemukan film serupa.")
        else:
            st.markdown(
                "<div style='color:var(--muted);font-size:.85rem;'>Tidak ada judul yang cocok. Coba kata kunci lain.</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# SURPRISE ME
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎲  Surprise Me":
    st.markdown("<div class='ph'>SURPRISE ME</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Bingung mau nonton apa? Biarkan sistem yang memilih</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    # ── Film Hari Ini
    with c1:
        motd   = st.session_state.motd
        mt     = str(motd.get("title", ""))
        mg     = str(motd.get("listed_in", ""))
        md_txt = str(motd.get("description", ""))
        mp, mid = fetch_tmdb(mt)
        tu  = trailer_url(mt, mid)
        pu  = tmdb_page_url(mid)
        ph  = f'<img src="{mp}" style="width:100%;border-radius:10px;margin-bottom:1rem;" loading="lazy">' if mp else ""
        st.markdown(f"""
        <div class="motd-card">
            <div class="motd-badge">Film Hari Ini</div>
            {ph}
            <div class="motd-title">{mt.upper()}</div>
            <div class="motd-genre">{mg}</div>
            <div class="motd-desc">{md_txt}</div>
            <div style="margin-top:1rem;display:flex;gap:8px;">
                <a class="btn-t" href="{tu}" target="_blank" rel="noopener">▶ Trailer</a>
                <a class="btn-s" href="{pu}" target="_blank" rel="noopener">TMDB</a>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Roulette
    with c2:
        st.markdown("""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:14px;
                    padding:1.5rem;margin-bottom:1rem;">
            <div class="motd-badge" style="background:var(--bg3);color:var(--muted);border:1px solid var(--border);">
                Random Roulette
            </div>
            <div style="font-size:.85rem;color:var(--muted);margin-top:.75rem;line-height:1.65;">
                Klik tombol di bawah untuk mendapatkan pilihan film acak dari ribuan judul Netflix.
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🎲  Putar Roulette", use_container_width=True):
            st.session_state.random_pick = df.sample(1).iloc[0]

        if st.session_state.random_pick is not None:
            r   = st.session_state.random_pick
            rt  = str(r.get("title", ""))
            rg  = str(r.get("listed_in", ""))
            rd  = str(r.get("description", ""))
            rp, rid = fetch_tmdb(rt)
            rtu = trailer_url(rt, rid)
            rpu = tmdb_page_url(rid)
            rph = f'<img src="{rp}" style="width:100%;border-radius:10px;margin:1rem 0;" loading="lazy">' if rp else ""
            st.markdown(f"""
            <div style="background:var(--bg3);border-radius:12px;padding:1.25rem;margin-top:.75rem;
                        border:1px solid var(--border);">
                {rph}
                <div style="font-family:var(--display);font-size:1.6rem;letter-spacing:2px;color:var(--text);">
                    {rt.upper()}
                </div>
                <div style="font-size:.8rem;color:var(--red);margin:.3rem 0 .6rem;">{rg}</div>
                <div style="font-size:.82rem;color:var(--muted);line-height:1.6;margin-bottom:.75rem;">{rd}</div>
                <a class="btn-t" href="{rtu}" target="_blank" rel="noopener">▶ Trailer</a>
                <a class="btn-s" href="{rpu}" target="_blank" rel="noopener" style="margin-left:8px;">TMDB</a>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS  — Plotly (no matplotlib dependency)
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊  Analytics":
    st.markdown("<div class='ph'>ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Eksplorasi distribusi dataset Netflix secara visual</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    total  = len(df)
    nm     = len(df[df["type"] == "Movie"])   if "type" in df.columns else 0
    ns     = len(df[df["type"] == "TV Show"]) if "type" in df.columns else 0
    ng     = df["listed_in"].str.split(", ").explode().nunique() if "listed_in" in df.columns else 0

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-label">Total Judul</div><div class="stat-value">{total:,}</div></div>
        <div class="stat-card"><div class="stat-label">Movies</div><div class="stat-value"><span class="stat-accent">{nm:,}</span></div></div>
        <div class="stat-card"><div class="stat-label">TV Shows</div><div class="stat-value">{ns:,}</div></div>
        <div class="stat-card"><div class="stat-label">Genre Unik</div><div class="stat-value">{ng:,}</div></div>
    </div>""", unsafe_allow_html=True)

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8A8790", family="DM Sans, sans-serif", size=12),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    col1, col2 = st.columns(2, gap="medium")

    # ── Chart 1: Movie vs TV Show
    with col1:
        st.markdown("<div class='st2' style='font-size:1.1rem;'>MOVIE VS TV SHOW</div>", unsafe_allow_html=True)
        if "type" in df.columns:
            ct = df["type"].value_counts()
            fig = go.Figure(go.Bar(
                x=ct.index.tolist(),
                y=ct.values.tolist(),
                marker_color=["#E8232A", "#F0A500"],
                text=[f"{v:,}" for v in ct.values],
                textposition="outside",
                textfont=dict(color="#F0EDE8", size=13),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8A8790")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(color="#8A8790")),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Tren per Tahun
    with col2:
        st.markdown("<div class='st2' style='font-size:1.1rem;'>TREN RILIS PER TAHUN</div>", unsafe_allow_html=True)
        if "release_year" in df.columns:
            yd = df["release_year"].value_counts().sort_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=yd.index.tolist(), y=yd.values.tolist(),
                mode="lines", line=dict(color="#E8232A", width=2.5),
                fill="tozeroy", fillcolor="rgba(232,35,42,0.08)",
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8A8790")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(color="#8A8790")),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Top 10 Genre
    st.markdown("<br><div class='st2' style='font-size:1.1rem;'>TOP 10 GENRE</div>", unsafe_allow_html=True)
    if "listed_in" in df.columns:
        gc = df["listed_in"].str.split(", ").explode().value_counts().head(10)
        fig3 = go.Figure(go.Bar(
            x=gc.values[::-1].tolist(),
            y=gc.index[::-1].tolist(),
            orientation="h",
            marker_color="#E8232A",
            text=[f"{v:,}" for v in gc.values[::-1]],
            textposition="outside",
            textfont=dict(color="#8A8790", size=11),
        ))
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(color="#8A8790")),
            yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#F0EDE8", size=12)),
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Rating Distribution
    if "rating" in df.columns:
        st.markdown("<br><div class='st2' style='font-size:1.1rem;'>DISTRIBUSI RATING</div>", unsafe_allow_html=True)
        rd = df["rating"].value_counts().sort_values(ascending=False)
        fig4 = go.Figure(go.Bar(
            x=rd.index.tolist(),
            y=rd.values.tolist(),
            marker_color=[
                "#E8232A" if i % 2 == 0 else "#9B1217"
                for i in range(len(rd))
            ],
            text=[f"{v:,}" for v in rd.values],
            textposition="outside",
            textfont=dict(color="#F0EDE8", size=11),
        ))
        fig4.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8A8790")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(color="#8A8790")),
            height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MOVREC &nbsp;·&nbsp; Content-Based Filtering &nbsp;·&nbsp; TF-IDF + Cosine Similarity<br>
    <span style="opacity:.5;">Data Science Beginner &nbsp;·&nbsp; UPN Veteran Jakarta &nbsp;·&nbsp; 2024</span>
</div>""", unsafe_allow_html=True)
