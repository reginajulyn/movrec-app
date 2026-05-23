import streamlit as st
import joblib
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse  # ✅ FIX: import eksplisit submodul urllib.parse

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MovRec – Netflix Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --red:#E8232A; --red-dim:#9B1217; --red-glow:rgba(232,35,42,0.15);
    --gold:#F0A500; --gold-dim:#B07800;
    --bg0:#08080A; --bg1:#0F0F14; --bg2:#16161E; --bg3:#1E1E28; --bg4:#26262F;
    --border:rgba(255,255,255,0.06); --border2:rgba(255,255,255,0.11);
    --text:#F2EFE9; --muted:#7C7A84; --subtle:#3A3A45;
    --sans:'Inter',sans-serif; --display:'Bebas Neue',sans-serif;
    --radius:10px; --radius-lg:14px; --radius-xl:18px;
}

html,body,.stApp,[data-testid="stAppViewContainer"]{
    background:var(--bg0)!important; color:var(--text)!important;
    font-family:var(--sans);
}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stToolbar"]{display:none!important;}
.block-container{padding:1.75rem 2rem 4rem!important; max-width:1360px!important;}

::-webkit-scrollbar{width:5px; height:5px;}
::-webkit-scrollbar-track{background:var(--bg1);}
::-webkit-scrollbar-thumb{background:var(--subtle); border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--red-dim);}

[data-testid="stSidebar"]{
    background:var(--bg1)!important;
    border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{font-family:var(--sans)!important;}
[data-testid="stSidebarContent"]{padding:1.25rem 0.9rem!important;}

.sb-logo{font-family:var(--display); font-size:2.2rem; letter-spacing:4px; color:var(--red); line-height:1; padding:0 4px;}
.sb-sub{font-size:.65rem; letter-spacing:2.5px; text-transform:uppercase; color:var(--muted); margin-bottom:1.5rem; padding:0 4px;}
.sb-divider{height:1px; background:var(--border); margin:.9rem 0;}
.sb-stat-wrap{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:.85rem 1rem; margin-bottom:.75rem;}
.sb-stat-row{display:flex; gap:1rem;}
.sb-stat .v{font-size:1.3rem; font-weight:700; color:var(--text); line-height:1;}
.sb-stat .v.red{color:var(--red);}
.sb-stat .l{font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:2px;}
.sb-team{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:.85rem 1rem; font-size:.72rem; color:var(--muted); line-height:1.8;}
.sb-team .tt{font-size:.7rem; font-weight:600; color:var(--text); text-transform:uppercase; letter-spacing:1px; margin-bottom:.4rem;}

[data-testid="stSelectbox"]>div>div,
[data-testid="stTextInput"]>div>div{
    background:var(--bg2)!important; border:1px solid var(--border2)!important;
    border-radius:var(--radius)!important; color:var(--text)!important; font-size:.875rem!important;
}
[data-testid="stSelectbox"]>div>div:focus-within,
[data-testid="stTextInput"]>div>div:focus-within{
    border-color:var(--red)!important; box-shadow:0 0 0 3px var(--red-glow)!important;
}
.stButton>button{
    background:var(--red)!important; color:#fff!important; border:none!important;
    border-radius:var(--radius)!important; font-family:var(--sans)!important; font-weight:600!important;
    font-size:.82rem!important; letter-spacing:.4px!important; padding:.5rem 1.3rem!important;
    transition:background .2s,transform .15s,box-shadow .2s!important;
}
.stButton>button:hover{background:var(--red-dim)!important; transform:translateY(-1px)!important; box-shadow:0 4px 16px var(--red-glow)!important;}
.stButton>button:active{transform:translateY(0)!important;}

.grid-row{display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:12px;}
.mc{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius-lg); overflow:hidden; text-decoration:none; display:block; transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease; position:relative;}
.mc:hover{transform:translateY(-5px) scale(1.01); border-color:rgba(232,35,42,.4); box-shadow:0 12px 32px rgba(0,0,0,.5);}
.mc img{width:100%; aspect-ratio:2/3; object-fit:cover; display:block;}
.mc-nop{width:100%; aspect-ratio:2/3; background:var(--bg3); display:flex; align-items:center; justify-content:center; font-family:var(--display); font-size:1.6rem; color:var(--subtle); letter-spacing:2px;}
.mc-body{padding:.7rem .8rem .85rem;}
.mc-title{font-weight:600; font-size:.83rem; color:var(--text); margin-bottom:.25rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.mc-genre{font-size:.68rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.mc-tags{display:flex; gap:4px; margin-top:.4rem; flex-wrap:wrap;}
.tag-score{font-size:.62rem; font-weight:700; padding:.12rem .45rem; background:var(--red-dim); color:#FFD0D0; border-radius:20px; letter-spacing:.4px;}
.tag-type{font-size:.62rem; font-weight:600; padding:.12rem .45rem; background:var(--bg4); color:var(--muted); border-radius:20px; border:1px solid var(--border2);}

.lc{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius-lg); padding:.9rem 1.1rem; margin-bottom:.6rem; display:flex; gap:.9rem; align-items:flex-start; transition:border-color .2s,background .2s;}
.lc:hover{border-color:rgba(232,35,42,.35); background:var(--bg3);}
.lc-poster{width:60px; min-width:60px; border-radius:6px; aspect-ratio:2/3; object-fit:cover;}
.lc-nop{width:60px; min-width:60px; border-radius:6px; aspect-ratio:2/3; background:var(--bg3); display:flex; align-items:center; justify-content:center; font-size:.65rem; color:var(--muted);}
.lc-content{flex:1; min-width:0;}
.lc-top{display:flex; justify-content:space-between; align-items:baseline; gap:6px; margin-bottom:.2rem;}
.lc-title{font-weight:600; font-size:.9rem; color:var(--text);}
.lc-score{font-size:.72rem; color:var(--gold); font-weight:700; white-space:nowrap;}
.lc-genre{font-size:.7rem; color:var(--red); margin-bottom:.3rem;}
.lc-desc{font-size:.78rem; color:var(--muted); line-height:1.55;}
.lc-actions{display:flex; gap:7px; margin-top:.45rem; flex-wrap:wrap;}
.btn-t{font-size:.68rem; font-weight:600; padding:.2rem .65rem; background:var(--red); color:#fff; border-radius:5px; text-decoration:none; letter-spacing:.3px; transition:background .15s;}
.btn-t:hover{background:var(--red-dim);}
.btn-s{font-size:.68rem; font-weight:600; padding:.2rem .65rem; background:var(--bg3); color:var(--muted); border-radius:5px; text-decoration:none; border:1px solid var(--border2); transition:color .15s, border-color .15s;}
.btn-s:hover{color:var(--text); border-color:var(--border2);}

.hero-wrap{position:relative; border-radius:var(--radius-xl); overflow:hidden; margin-bottom:2rem; border:1px solid var(--border);}
.hero-img{width:100%; height:400px; object-fit:cover; display:block; filter:brightness(.35) saturate(1.1);}
.hero-overlay{position:absolute; bottom:0; left:0; right:0; padding:2rem 2rem 1.75rem; background:linear-gradient(to top, rgba(8,8,10,.99) 0%, transparent 100%);}
.hero-badge{display:inline-block; background:var(--red); color:#fff; font-size:.6rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; padding:.25rem .65rem; border-radius:4px; margin-bottom:.55rem;}
.hero-title{font-family:var(--display); font-size:2.6rem; letter-spacing:3px; color:var(--text); line-height:1; margin-bottom:.4rem;}
.hero-meta{font-size:.75rem; color:var(--muted); margin-bottom:1.1rem;}
.hero-btn{display:inline-flex; align-items:center; gap:6px; background:var(--red); color:#fff; padding:.5rem 1.3rem; border-radius:var(--radius); font-size:.8rem; font-weight:600; letter-spacing:.4px; text-decoration:none; transition:background .2s, box-shadow .2s;}
.hero-btn:hover{background:var(--red-dim); box-shadow:0 4px 20px var(--red-glow);}

.motd-card{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius-xl); padding:1.35rem; height:100%;}
.motd-badge{display:inline-block; background:var(--red); color:#fff; font-size:.6rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; padding:.2rem .6rem; border-radius:4px; margin-bottom:.7rem;}
.motd-title{font-family:var(--display); font-size:1.75rem; letter-spacing:2px; color:var(--text); margin-bottom:.3rem;}
.motd-genre{font-size:.75rem; color:var(--red); margin-bottom:.65rem;}
.motd-desc{font-size:.82rem; color:var(--muted); line-height:1.6;}

.stat-row{display:flex; gap:10px; margin-bottom:1.5rem;}
.stat-card{flex:1; background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:.9rem 1rem;}
.stat-label{font-size:.65rem; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-bottom:.3rem;}
.stat-value{font-family:var(--display); font-size:1.7rem; letter-spacing:2px; color:var(--text);}
.stat-accent{color:var(--red);}

.ph{font-family:var(--display); font-size:2.8rem; letter-spacing:4px; color:var(--text); line-height:1; margin-bottom:.25rem;}
.ps{font-size:.72rem; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:1.75rem;}
.sr{height:1px; margin:.6rem 0 1.35rem; background:linear-gradient(to right, var(--red), transparent);}
.st2{font-family:var(--display); font-size:1.4rem; letter-spacing:2.5px; color:var(--text); margin-bottom:.15rem;}

.mood-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1.75rem;}
.mood-pill{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius-lg); padding:1rem .9rem; cursor:pointer; transition:all .2s; display:flex; align-items:center; gap:.75rem;}
.mood-pill:hover{border-color:var(--red); background:var(--red-glow);}
.mood-icon{font-size:1.5rem;}
.mood-label{font-weight:600; font-size:.82rem; color:var(--text);}
.mood-sub{font-size:.67rem; color:var(--muted);}

.info-box{background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:1.1rem 1.2rem; font-size:.82rem; color:var(--muted); line-height:1.65;}
.info-box strong{color:var(--text);}

.roulette-card{background:var(--bg3); border:1px solid var(--border2); border-radius:var(--radius-lg); padding:1.1rem; margin-top:.65rem;}

.footer{margin-top:4rem; padding-top:1.25rem; border-top:1px solid var(--border); text-align:center; font-size:.7rem; color:var(--muted); letter-spacing:1px;}

.stMarkdown p{color:var(--text)!important;}
div[data-testid="column"]{padding:0 5px!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--red)!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    data      = joblib.load("movies.pkl")
    tfidf_vec = joblib.load("tfidf.pkl")
    tfidf_mat = joblib.load("tfidf_matrix.pkl")

    # ✅ FIX: Normalise index — jika index DataFrame adalah kolom title
    # (akibat set_index("title") di notebook), reset supaya jadi RangeIndex
    # dan pastikan kolom "title" selalu ada sebagai kolom biasa.
    if data.index.name == "title" or (
        data.index.dtype == object and "title" not in data.columns
    ):
        data = data.reset_index()          # index "title" → jadi kolom
    elif "title" not in data.columns:
        # Fallback: coba kolom pertama sebagai title
        data = data.rename(columns={data.columns[0]: "title"})

    # Pastikan index adalah RangeIndex bersih (0, 1, 2, …)
    data = data.reset_index(drop=True)

    idx_series = pd.Series(data.index, index=data["title"]).drop_duplicates()
    return data, tfidf_vec, tfidf_mat, idx_series

df, tfidf, tfidf_matrix, indices = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# TMDB HELPERS
# ══════════════════════════════════════════════════════════════════════════════
TMDB_KEY = "0758644da67e27b71fac69a53fab875e"

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_tmdb(title: str):
    """Ambil poster + TMDB id. Cached 24 jam."""
    try:
        # ✅ FIX: urllib.parse.quote sudah bisa dipanggil karena import eksplisit di atas
        encoded = urllib.parse.quote(str(title))
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_KEY}&query={encoded}&language=id-ID"
        )
        res  = requests.get(url, timeout=5).json()
        # ✅ FIX: guard jika results kosong
        if not res.get("results"):
            return None, None
        hit  = res["results"][0]
        poster = (
            "https://image.tmdb.org/t/p/w342" + hit["poster_path"]
            if hit.get("poster_path") else None
        )
        return poster, hit.get("id")
    except Exception:
        return None, None

def yt_url(title: str) -> str:
    # ✅ FIX: pastikan title di-cast ke str sebelum di-quote
    q = urllib.parse.quote(f"{str(title)} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"

def tmdb_url(tmdb_id) -> str:
    return f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else "#"


# ══════════════════════════════════════════════════════════════════════════════
# MOOD CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MOODS = {
    "Happy / Feel-Good":      ("comedy family fun animation",           "Comedy · Family"),
    "Sad / Emotional":        ("drama emotional heartbreaking loss",     "Drama · Tearjerker"),
    "Action / Thrilling":     ("action thriller adventure fight",        "Action · Thriller"),
    "Romantic / Love":        ("romance love relationship wedding",      "Romance · Drama"),
    "Sci-Fi / Mind-Bending":  ("science fiction space future robot AI",  "Sci-Fi · Fantasy"),
    "Chill / Documentary":    ("documentary nature calm travel culture", "Documentary"),
    "Horror / Suspense":      ("horror scary suspense paranormal",       "Horror · Suspense"),
    "Intense Drama":          ("drama crime mystery psychological",      "Crime · Drama"),
}


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def recommend_by_mood(mood_key: str, n: int = 10):
    query_text = MOODS[mood_key][0]
    vec = tfidf.transform([query_text])
    sim = cosine_similarity(vec, tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[:n]

def recommend_by_title(title: str, n: int = 10):
    if title not in indices:
        return []
    i   = indices[title]
    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1 : n + 1]


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _card(title, genre, ctype, poster, tid, score=None):
    # ✅ FIX: pastikan semua input adalah string sebelum diproses
    title = str(title) if title is not None else ""
    genre = str(genre) if genre is not None else ""
    ctype = str(ctype) if ctype is not None else ""

    ts  = title.replace("'", "&#39;").replace('"', "&quot;")
    gs  = (genre[:34] + "…") if len(genre) > 34 else genre
    img = (f'<img src="{poster}" alt="{ts}" loading="lazy">'
           if poster else f'<div class="mc-nop">{title[:2].upper()}</div>')
    sc  = (f'<span class="tag-score">{round(score*100)}%</span>' if score and score > 0 else "")
    tp  = (f'<span class="tag-type">{ctype}</span>' if ctype else "")
    return (f'<a class="mc" href="{yt_url(title)}" target="_blank" rel="noopener">'
            f'{img}<div class="mc-body">'
            f'<div class="mc-title" title="{ts}">{title}</div>'
            f'<div class="mc-genre">{gs}</div>'
            f'<div class="mc-tags">{sc}{tp}</div>'
            f'</div></a>')

def render_grid(pairs, show_score=False, cols=5):
    rows = [pairs[i : i + cols] for i in range(0, len(pairs), cols)]
    for row in rows:
        html = ""
        for idx, score in row:
            ro = df.iloc[idx]
            t  = str(ro.get("title", ""))
            g  = str(ro.get("listed_in", ""))
            ct = str(ro.get("type", ""))
            p, tid = fetch_tmdb(t)
            html += _card(t, g, ct, p, tid, score if show_score else None)
        st.markdown(f'<div class="grid-row">{html}</div>', unsafe_allow_html=True)

def render_list(pairs):
    html = ""
    for idx, score in pairs:
        ro   = df.iloc[idx]
        t    = str(ro.get("title", ""))
        g    = str(ro.get("listed_in", ""))
        desc = str(ro.get("description", ""))
        d    = desc[:200] + ("…" if len(desc) > 200 else "")
        ts   = t.replace("'", "&#39;")
        p, tid = fetch_tmdb(t)
        sc  = (f'<span class="lc-score">★ {round(score*100)}% match</span>' if score > 0 else "")
        th  = (f'<img class="lc-poster" src="{p}" alt="{ts}" loading="lazy">'
               if p else '<div class="lc-nop">N/A</div>')
        html += (f'<div class="lc">{th}<div class="lc-content">'
                 f'<div class="lc-top"><span class="lc-title">{t}</span>{sc}</div>'
                 f'<div class="lc-genre">{g}</div>'
                 f'<div class="lc-desc">{d}</div>'
                 f'<div class="lc-actions">'
                 f'<a class="btn-t" href="{yt_url(t)}" target="_blank" rel="noopener">▶ Trailer</a>'
                 f'<a class="btn-s" href="{tmdb_url(tid)}" target="_blank" rel="noopener">TMDB</a>'
                 f'</div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "motd" not in st.session_state:
    st.session_state.motd = df.sample(1).iloc[0]
if "roulette" not in st.session_state:
    st.session_state.roulette = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sb-logo'>MOVREC</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sub'>Netflix Recommendation Engine</div>", unsafe_allow_html=True)

    menu = st.radio(
        "nav", ["Home", "Mood", "Film Sejenis", "Surprise Me", "Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)

    total = len(df)
    nm    = int((df["type"] == "Movie").sum())   if "type" in df.columns else 0
    ns    = int((df["type"] == "TV Show").sum()) if "type" in df.columns else 0

    st.markdown(f"""
    <div class='sb-stat-wrap'>
        <div style='font-size:.65rem;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:.6rem;'>
            Dataset
        </div>
        <div style='font-size:1.55rem;font-weight:700;color:var(--text);line-height:1;margin-bottom:.2rem;'>
            {total:,}
        </div>
        <div style='font-size:.65rem;color:var(--muted);margin-bottom:.75rem;'>judul tersedia</div>
        <div class='sb-stat-row'>
            <div class='sb-stat'>
                <div class='v red'>{nm:,}</div>
                <div class='l'>Movies</div>
            </div>
            <div class='sb-stat'>
                <div class='v'>{ns:,}</div>
                <div class='l'>TV Shows</div>
            </div>
        </div>
    </div>
    <div class='sb-team'>
        <div class='tt'>Kelompok</div>
        Data Science Beginner<br>
        Nayla Dwinta P. M.<br>
        Lathisya Sheza A.<br>
        Regina Juliyanti M.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7C7A84", family="Inter, sans-serif", size=11),
    margin=dict(l=0, r=10, t=28, b=0),
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠  Home":
    motd = st.session_state.motd
    mt   = str(motd.get("title", "Featured"))
    mg   = str(motd.get("listed_in", ""))
    mp, mid = fetch_tmdb(mt)
    tu   = yt_url(mt)

    if mp:
        st.markdown(f"""
        <div class='hero-wrap'>
            <img class='hero-img' src='{mp}' alt='{mt}'/>
            <div class='hero-overlay'>
                <div class='hero-badge'>✦ Pilihan Hari Ini</div>
                <div class='hero-title'>{mt.upper()}</div>
                <div class='hero-meta'>{mg}</div>
                <a class='hero-btn' href='{tu}' target='_blank' rel='noopener'>▶ Tonton Trailer</a>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-xl);
                    padding:2.5rem 2rem;margin-bottom:2rem;'>
            <div class='hero-badge'>✦ Pilihan Hari Ini</div>
            <div class='hero-title'>{mt.upper()}</div>
            <div class='hero-meta'>{mg}</div>
            <a class='hero-btn' href='{tu}' target='_blank' rel='noopener'
               style='margin-top:.9rem;display:inline-flex;'>▶ Trailer</a>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='st2'>TRENDING SEKARANG</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;color:var(--muted);margin-bottom:.4rem;'>Konten populer dari database Netflix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    render_grid([(i, 0.0) for i in df.sample(10).index])

    st.markdown("<br><div class='st2'>BARU DITAMBAHKAN</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;color:var(--muted);margin-bottom:.4rem;'>Film & serial terbaru di dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    if "date_added" in df.columns:
        rec = df.dropna(subset=["date_added"]).copy()
        rec["date_added"] = pd.to_datetime(rec["date_added"], errors="coerce")
        rec = rec.sort_values("date_added", ascending=False).head(10)
        render_grid([(i, 0.0) for i in rec.index])
    else:
        render_grid([(i, 0.0) for i in df.sample(10).index])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — MOOD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎭  Mood":
    st.markdown("<div class='ph'>MOOD MATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Sistem mencarikan film yang paling pas dengan vibes kamu</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    mood_keys = list(MOODS.keys())
    pills_html = '<div class="mood-grid">'
    for mk in mood_keys:
        icon  = mk.split()[0]
        label = " ".join(mk.split()[1:])
        sub   = MOODS[mk][1]
        pills_html += (f'<div class="mood-pill">'
                       f'<span class="mood-icon">{icon}</span>'
                       f'<div><div class="mood-label">{label}</div>'
                       f'<div class="mood-sub">{sub}</div></div></div>')
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    sel = st.selectbox("Pilih suasana hati:", mood_keys)
    cb, _ = st.columns([1, 4])
    with cb:
        run = st.button("Temukan Film", use_container_width=True)

    if run:
        with st.spinner("Mencari film terbaik…"):
            res = recommend_by_mood(sel)
        label = " ".join(sel.split()[1:])
        st.markdown(f"<br><div class='st2'>HASIL: {label.upper()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
        render_grid(res, show_score=True)
        st.markdown("<br><div class='st2'>DETAIL REKOMENDASI</div>", unsafe_allow_html=True)
        st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
        render_list(res)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — FILM SEJENIS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍  Film Sejenis":
    st.markdown("<div class='ph'>FILM SEJENIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Masukkan judul favoritmu — cosine similarity mencarikan kembarannya</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <strong>Cara pakai:</strong> Ketik sebagian judul film → pilih dari dropdown →
        klik <em>Cari Film Sejenis</em>. Algoritma TF-IDF + Cosine Similarity akan
        menghitung kemiripan berdasarkan genre, deskripsi, dan metadata lainnya.
    </div><br>""", unsafe_allow_html=True)

    query = st.text_input("🔍  Cari judul film…", placeholder="Contoh: Inception, The Crown, Money Heist…")
    if query:
        matched = df[df["title"].str.contains(query, case=False, na=False)]["title"].head(10).tolist()
        if matched:
            st.markdown("<div style='font-size:.75rem;color:var(--muted);margin:.5rem 0 .25rem;'>Pilih judul yang tepat:</div>", unsafe_allow_html=True)
            chosen = st.selectbox("", matched, label_visibility="collapsed")
            cb, _ = st.columns([1, 4])
            with cb:
                go_btn = st.button("Cari Film Sejenis", use_container_width=True)
            if go_btn:
                with st.spinner("Menganalisis kemiripan konten…"):
                    res = recommend_by_title(chosen)
                if res:
                    st.markdown(f"<br><div class='st2'>MIRIP DENGAN: {chosen.upper()}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
                    render_grid(res, show_score=True)
                    st.markdown("<br><div class='st2'>DETAIL REKOMENDASI</div>", unsafe_allow_html=True)
                    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
                    render_list(res)
                else:
                    st.error("Film tidak ditemukan dalam index. Coba judul lain.")
        else:
            st.markdown("<div class='info-box'>Tidak ada judul yang cocok. Coba kata kunci lain.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — SURPRISE ME
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎲  Surprise Me":
    st.markdown("<div class='ph'>SURPRISE ME</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Bingung mau nonton apa? Biarkan sistem yang memilih</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        motd    = st.session_state.motd
        mt      = str(motd.get("title", ""))
        mg      = str(motd.get("listed_in", ""))
        md_txt  = str(motd.get("description", ""))
        mp, mid = fetch_tmdb(mt)
        ph = (f'<img src="{mp}" style="width:100%;border-radius:8px;margin-bottom:.9rem;" loading="lazy">'
              if mp else "")
        st.markdown(f"""
        <div class='motd-card'>
            <div class='motd-badge'>Film Hari Ini</div>
            {ph}
            <div class='motd-title'>{mt.upper()}</div>
            <div class='motd-genre'>{mg}</div>
            <div class='motd-desc'>{md_txt}</div>
            <div style='margin-top:.9rem;display:flex;gap:7px;'>
                <a class='btn-t' href='{yt_url(mt)}' target='_blank' rel='noopener'>▶ Trailer</a>
                <a class='btn-s' href='{tmdb_url(mid)}' target='_blank' rel='noopener'>TMDB</a>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:var(--bg2);border:1px solid var(--border);
                    border-radius:var(--radius-xl);padding:1.35rem;margin-bottom:.85rem;'>
            <div class='motd-badge' style='background:var(--bg4);color:var(--muted);
                        border:1px solid var(--border2);'>Random Roulette</div>
            <div style='font-size:.82rem;color:var(--muted);margin-top:.65rem;line-height:1.65;'>
                Klik tombol di bawah untuk mendapatkan pilihan film acak dari ribuan judul Netflix.
                Setiap klik menghasilkan rekomendasi baru yang berbeda.
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🎲  Putar Roulette", use_container_width=True):
            st.session_state.roulette = df.sample(1).iloc[0]

        if st.session_state.roulette is not None:
            r   = st.session_state.roulette
            rt  = str(r.get("title", ""))
            rg  = str(r.get("listed_in", ""))
            rd  = str(r.get("description", ""))
            rp, rid = fetch_tmdb(rt)
            rph = (f'<img src="{rp}" style="width:100%;border-radius:8px;margin:0.8rem 0;" loading="lazy">'
                   if rp else "")
            st.markdown(f"""
            <div class='roulette-card'>
                {rph}
                <div class='motd-title' style='font-size:1.5rem;'>{rt.upper()}</div>
                <div class='motd-genre'>{rg}</div>
                <div class='motd-desc'>{rd}</div>
                <div style='margin-top:.75rem;display:flex;gap:7px;'>
                    <a class='btn-t' href='{yt_url(rt)}' target='_blank' rel='noopener'>▶ Trailer</a>
                    <a class='btn-s' href='{tmdb_url(rid)}' target='_blank' rel='noopener'>TMDB</a>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊  Analytics":
    st.markdown("<div class='ph'>ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Eksplorasi distribusi dataset Netflix secara visual</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)

    total = len(df)
    nm    = int((df["type"] == "Movie").sum())   if "type" in df.columns else 0
    ns    = int((df["type"] == "TV Show").sum()) if "type" in df.columns else 0
    ng    = int(df["listed_in"].str.split(", ").explode().nunique()) if "listed_in" in df.columns else 0

    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-card'><div class='stat-label'>Total Judul</div><div class='stat-value'>{total:,}</div></div>
        <div class='stat-card'><div class='stat-label'>Movies</div><div class='stat-value'><span class='stat-accent'>{nm:,}</span></div></div>
        <div class='stat-card'><div class='stat-label'>TV Shows</div><div class='stat-value'>{ns:,}</div></div>
        <div class='stat-card'><div class='stat-label'>Genre Unik</div><div class='stat-value'>{ng:,}</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("<div class='st2' style='font-size:1.05rem;'>MOVIE VS TV SHOW</div>", unsafe_allow_html=True)
        if "type" in df.columns:
            ct  = df["type"].value_counts()
            fig = go.Figure(go.Bar(
                x=ct.index.tolist(), y=ct.values.tolist(),
                marker_color=["#E8232A", "#F0A500"],
                text=[f"{v:,}" for v in ct.values],
                textposition="outside",
                textfont=dict(color="#F2EFE9", size=12),
                width=0.45,
            ))
            fig.update_layout(
                **PL,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#7C7A84", size=12)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#7C7A84")),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='st2' style='font-size:1.05rem;'>TREN RILIS PER TAHUN</div>", unsafe_allow_html=True)
        if "release_year" in df.columns:
            yd   = df["release_year"].value_counts().sort_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=yd.index.tolist(), y=yd.values.tolist(),
                mode="lines", line=dict(color="#E8232A", width=2.2),
                fill="tozeroy", fillcolor="rgba(232,35,42,0.07)",
                hovertemplate="<b>%{x}</b>: %{y} judul<extra></extra>",
            ))
            fig2.update_layout(
                **PL,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#7C7A84")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#7C7A84")),
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br><div class='st2' style='font-size:1.05rem;'>TOP 10 GENRE</div>", unsafe_allow_html=True)
    if "listed_in" in df.columns:
        gc   = df["listed_in"].str.split(", ").explode().value_counts().head(10)
        fig3 = go.Figure(go.Bar(
            x=gc.values[::-1].tolist(),
            y=gc.index[::-1].tolist(),
            orientation="h",
            marker=dict(
                color=gc.values[::-1].tolist(),
                colorscale=[[0, "#9B1217"], [1, "#E8232A"]],
                showscale=False,
            ),
            text=[f"{v:,}" for v in gc.values[::-1]],
            textposition="outside",
            textfont=dict(color="#7C7A84", size=10),
            hovertemplate="<b>%{y}</b>: %{x:,} judul<extra></extra>",
        ))
        fig3.update_layout(
            **PL,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#7C7A84")),
            yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#F2EFE9", size=11)),
            height=360,
        )
        st.plotly_chart(fig3, use_container_width=True)

    if "rating" in df.columns:
        st.markdown("<br><div class='st2' style='font-size:1.05rem;'>DISTRIBUSI RATING KONTEN</div>", unsafe_allow_html=True)
        rd = df["rating"].dropna().value_counts().sort_values(ascending=False)
        colors = np.where(np.arange(len(rd)) == 0, "#E8232A", "#3A3A45")
        fig4 = go.Figure(go.Bar(
            x=rd.index.tolist(), y=rd.values.tolist(),
            marker_color=colors.tolist(),
            text=[f"{v:,}" for v in rd.values],
            textposition="outside",
            textfont=dict(color="#F2EFE9", size=10),
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))
        fig4.update_layout(
            **PL,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#7C7A84", size=11)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#7C7A84")),
            height=280,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    MOVREC &nbsp;·&nbsp; Content-Based Filtering &nbsp;·&nbsp; TF-IDF + Cosine Similarity<br>
    <span style='opacity:.45;'>Data Science Beginner &nbsp;·&nbsp; UPN Veteran Jakarta &nbsp;·&nbsp; 2024</span>
</div>""", unsafe_allow_html=True)
