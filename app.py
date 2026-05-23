import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

st.set_page_config(
    page_title="MovRec",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --red:#E8232A; --red-dim:#9B1217; --gold:#F0A500;
    --bg0:#0A0A0C; --bg1:#111116; --bg2:#18181F; --bg3:#21212B;
    --border:rgba(255,255,255,0.07);
    --text:#F0EDE8; --muted:#8A8790;
    --sans:'DM Sans',sans-serif; --display:'Bebas Neue',sans-serif;
}
html,body,.stApp,[data-testid="stAppViewContainer"]{background-color:var(--bg0)!important;color:var(--text)!important;font-family:var(--sans);}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stToolbar"]{display:none;}
.block-container{padding:2rem 2.5rem 4rem!important;max-width:1400px!important;}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{font-family:var(--sans)!important;}
[data-testid="stSidebarContent"]{padding:1.5rem 1rem!important;}
.sidebar-logo{font-family:var(--display);font-size:2.4rem;letter-spacing:3px;color:var(--red);line-height:1;margin-bottom:0.25rem;}
.sidebar-tagline{font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:1.75rem;}
.sidebar-divider{height:1px;background:var(--border);margin:1rem 0;}
[data-testid="stSelectbox"]>div>div,[data-testid="stTextInput"]>div>div{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
.stButton>button{background:var(--red)!important;color:#fff!important;border:none!important;border-radius:6px!important;font-family:var(--sans)!important;font-weight:600!important;font-size:0.85rem!important;letter-spacing:0.5px!important;padding:0.55rem 1.4rem!important;transition:background 0.2s,transform 0.15s!important;}
.stButton>button:hover{background:var(--red-dim)!important;transform:translateY(-1px)!important;}
.grid-row{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:14px;}
.mc{background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden;text-decoration:none;display:block;transition:transform 0.25s,border-color 0.25s;}
.mc:hover{transform:translateY(-4px);border-color:var(--red-dim);}
.mc img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;}
.mc-nop{width:100%;aspect-ratio:2/3;background:var(--bg3);display:flex;align-items:center;justify-content:center;font-size:1.5rem;color:var(--muted);font-family:var(--display);letter-spacing:2px;}
.mc-body{padding:0.75rem 0.85rem 1rem;}
.mc-title{font-weight:600;font-size:0.88rem;color:var(--text);margin-bottom:0.3rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.mc-genre{font-size:0.72rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.mc-score{display:inline-block;margin-top:0.4rem;background:var(--red-dim);color:#FFD0D0;font-size:0.68rem;font-weight:600;padding:0.15rem 0.5rem;border-radius:20px;letter-spacing:0.5px;}
.lc{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem;display:flex;gap:1rem;align-items:flex-start;transition:border-color 0.2s;text-decoration:none;}
.lc:hover{border-color:rgba(232,35,42,0.45);}
.lc-poster{width:64px;min-width:64px;border-radius:6px;aspect-ratio:2/3;object-fit:cover;}
.lc-nop{width:64px;min-width:64px;border-radius:6px;aspect-ratio:2/3;background:var(--bg3);display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:var(--muted);}
.lc-content{flex:1;min-width:0;}
.lc-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:0.25rem;}
.lc-title{font-weight:600;font-size:0.95rem;color:var(--text);}
.lc-score{font-size:0.75rem;color:var(--gold);font-weight:600;white-space:nowrap;}
.lc-genre{font-size:0.75rem;color:var(--red);margin-bottom:0.35rem;}
.lc-desc{font-size:0.81rem;color:var(--muted);line-height:1.55;}
.lc-actions{display:flex;gap:8px;margin-top:0.5rem;flex-wrap:wrap;}
.btn-t{font-size:0.72rem;font-weight:600;padding:0.22rem 0.7rem;background:var(--red);color:#fff;border-radius:4px;text-decoration:none;letter-spacing:0.4px;}
.btn-t:hover{background:var(--red-dim);}
.btn-s{font-size:0.72rem;font-weight:600;padding:0.22rem 0.7rem;background:var(--bg3);color:var(--muted);border-radius:4px;text-decoration:none;letter-spacing:0.4px;border:1px solid var(--border);}
.btn-s:hover{color:var(--text);}
.hero-wrap{position:relative;border-radius:16px;overflow:hidden;margin-bottom:2.5rem;}
.hero-img{width:100%;height:420px;object-fit:cover;display:block;filter:brightness(0.4);}
.hero-overlay{position:absolute;bottom:0;left:0;right:0;padding:2.5rem 2rem 2rem;background:linear-gradient(to top,rgba(10,10,12,0.98) 0%,transparent 100%);}
.hero-label{font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:var(--red);margin-bottom:0.4rem;}
.hero-title{font-family:var(--display);font-size:2.8rem;letter-spacing:3px;color:var(--text);line-height:1;margin-bottom:0.5rem;}
.hero-genre{font-size:0.8rem;color:var(--muted);}
.motd-card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:1.5rem;height:100%;}
.motd-badge{display:inline-block;background:var(--red);color:#fff;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;font-weight:700;padding:0.2rem 0.6rem;border-radius:4px;margin-bottom:0.75rem;}
.motd-title{font-family:var(--display);font-size:1.9rem;letter-spacing:2px;color:var(--text);margin-bottom:0.35rem;}
.motd-genre{font-size:0.8rem;color:var(--red);margin-bottom:0.75rem;}
.motd-desc{font-size:0.85rem;color:var(--muted);line-height:1.6;}
.stat-row{display:flex;gap:12px;margin-bottom:1.5rem;}
.stat-card{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1rem 1.1rem;}
.stat-label{font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:0.35rem;}
.stat-value{font-family:var(--display);font-size:1.8rem;letter-spacing:2px;color:var(--text);}
.stat-accent{color:var(--red);}
.ph{font-family:var(--display);font-size:3rem;letter-spacing:4px;color:var(--text);line-height:1;margin-bottom:0.3rem;}
.ps{font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:2rem;}
.sr{height:1px;background:linear-gradient(to right,var(--red),transparent);margin:0.75rem 0 1.5rem;}
.st2{font-family:var(--display);font-size:1.5rem;letter-spacing:3px;color:var(--text);margin-bottom:0.2rem;}
.footer{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--border);text-align:center;font-size:0.75rem;color:var(--muted);letter-spacing:1px;}
.stMarkdown p{color:var(--text)!important;}
div[data-testid="column"]{padding:0 6px!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    d   = joblib.load("movies.pkl")
    tf  = joblib.load("tfidf.pkl")
    tfm = joblib.load("tfidf_matrix.pkl")
    idx = pd.Series(d.index, index=d["title"]).drop_duplicates()
    return d, tf, tfm, idx

df, tfidf, tfidf_matrix, indices = load_model()

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

def trailer_url(title, tmdb_id):
    q = urllib.parse.quote(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"

def tmdb_page_url(tmdb_id):
    return f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None

mood_config = {
    "Happy / Feel-Good":     "comedy family fun",
    "Sad / Emotional":       "drama emotional heartbreaking",
    "Action / Thrilling":    "action thriller adventure",
    "Romantic / Love":       "romance love relationship",
    "Sci-Fi / Mind-Bending": "science fiction space future",
    "Chill / Documentary":   "documentary calm slow",
    "Horror / Suspense":     "horror scary suspense",
    "Intense Drama":         "drama serious intense",
}

def rec_mood(mood, n=10):
    vec = tfidf.transform([mood_config[mood]])
    sim = cosine_similarity(vec, tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[:n]

def rec_title(title, n=10):
    if title not in indices: return []
    idx = indices[title]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    return sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:n+1]

def card_html(title, genre, poster, tmdb_id, score=None):
    gs  = (genre[:38]+"…") if len(genre)>38 else genre
    ts  = title.replace("'","&#39;").replace('"','&quot;')
    tu  = trailer_url(title, tmdb_id)
    img = f'<img src="{poster}" alt="{ts}" loading="lazy">' if poster else f'<div class="mc-nop">{title[:2].upper()}</div>'
    sc  = f'<span class="mc-score">{round(score*100)}% match</span>' if (score and score>0) else ""
    return f'''<a class="mc" href="{tu}" target="_blank" rel="noopener">{img}<div class="mc-body"><div class="mc-title" title="{ts}">{title}</div><div class="mc-genre">{gs}</div>{sc}</div></a>'''

def render_grid(pairs, show_score=False, cols=5):
    rows = [pairs[i:i+cols] for i in range(0, len(pairs), cols)]
    for row in rows:
        cards = ""
        for (idx, score) in row:
            ro  = df.iloc[idx]
            t   = str(ro.get("title",""))
            g   = str(ro.get("listed_in",""))
            p,tid = fetch_tmdb(t)
            cards += card_html(t, g, p, tid, score if show_score else None)
        st.markdown(f'<div class="grid-row">{cards}</div>', unsafe_allow_html=True)

def render_list(pairs):
    html = ""
    for idx, score in pairs:
        ro  = df.iloc[idx]
        t   = str(ro.get("title",""))
        g   = str(ro.get("listed_in",""))
        d   = str(ro.get("description",""))[:200]
        if d: d += "…"
        ts  = t.replace("'","&#39;")
        p,tid = fetch_tmdb(t)
        tu  = trailer_url(t, tid)
        pu  = tmdb_page_url(tid) or "#"
        sc  = f'<span class="lc-score">{round(score*100)}% match</span>' if score>0 else ""
        th  = f'<img class="lc-poster" src="{p}" alt="{ts}" loading="lazy">' if p else '<div class="lc-nop">N/A</div>'
        html += f'''<div class="lc">{th}<div class="lc-content"><div class="lc-top"><span class="lc-title">{t}</span>{sc}</div><div class="lc-genre">{g}</div><div class="lc-desc">{d}</div><div class="lc-actions"><a class="btn-t" href="{tu}" target="_blank" rel="noopener">Trailer</a><a class="btn-s" href="{pu}" target="_blank" rel="noopener">TMDB</a></div></div></div>'''
    st.markdown(html, unsafe_allow_html=True)

if "motd" not in st.session_state:
    st.session_state.motd = df.sample(1).iloc[0]
if "random_pick" not in st.session_state:
    st.session_state.random_pick = None

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>MOVREC</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Netflix Recommendation Engine</div>", unsafe_allow_html=True)
    menu = st.radio("", ["Home","Mood","Cari Film Sejenis","Surprise Me","Analytics"], label_visibility="collapsed")
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    total   = len(df)
    nm      = len(df[df["type"]=="Movie"])   if "type" in df.columns else "—"
    ns      = len(df[df["type"]=="TV Show"]) if "type" in df.columns else "—"
    st.markdown(f"""<div style='font-size:0.75rem;color:var(--muted);'><div style='margin-bottom:0.5rem;'><b style='color:var(--text);'>{total:,}</b> judul tersedia</div><div style='margin-bottom:0.5rem;'><b style='color:var(--text);'>{nm}</b> Movies</div><div><b style='color:var(--text);'>{ns}</b> TV Shows</div></div>""", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:0.72rem;color:var(--muted);line-height:1.7;'><div style='margin-bottom:0.3rem;color:var(--text);font-size:0.78rem;font-weight:600;'>Kelompok</div>Data Science Beginner<br>Nayla Dwinta P. M.<br>Lathisya Sheza A.<br>Regina Juliyanti M.</div>""", unsafe_allow_html=True)

if menu == "Home":
    motd = st.session_state.motd
    mt   = str(motd.get("title","Featured"))
    mg   = str(motd.get("listed_in",""))
    mp, mid = fetch_tmdb(mt)
    tu   = trailer_url(mt, mid)
    if mp:
        st.markdown(f'''<div class="hero-wrap"><img class="hero-img" src="{mp}" alt="{mt}"/><div class="hero-overlay"><div class="hero-label">Pilihan Hari Ini</div><div class="hero-title">{mt.upper()}</div><div class="hero-genre">{mg}</div><a href="{tu}" target="_blank" rel="noopener" style="display:inline-block;margin-top:1rem;background:var(--red);color:#fff;padding:0.5rem 1.4rem;border-radius:6px;font-size:0.82rem;font-weight:600;letter-spacing:0.5px;text-decoration:none;">Tonton Trailer</a></div></div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div style="background:var(--bg2);border-radius:16px;padding:3rem 2rem;margin-bottom:2.5rem;"><div class="hero-label">Pilihan Hari Ini</div><div class="hero-title">{mt.upper()}</div><div class="hero-genre">{mg}</div></div>''', unsafe_allow_html=True)
    st.markdown("<div class='st2'>TRENDING SEKARANG</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;color:var(--muted);margin-bottom:0.5rem;'>Konten populer dari database Netflix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    render_grid([(i,0.0) for i in df.sample(10).index])

elif menu == "Mood":
    st.markdown("<div class='ph'>MOOD MATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Sistem mencarikan film yang paling pas dengan vibes kamu</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    sel  = st.selectbox("Pilih suasana hati kamu:", list(mood_config.keys()))
    cb, _ = st.columns([1,4])
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

elif menu == "Cari Film Sejenis":
    st.markdown("<div class='ph'>FILM SEJENIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Masukkan judul favoritmu — cosine similarity mencarikan kembarannya</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    query = st.text_input("Cari judul film…", placeholder="Contoh: Inception, Transformers, The Crown…")
    if query:
        matched = df[df["title"].str.contains(query, case=False, na=False)]["title"].head(8).tolist()
        if matched:
            st.markdown("<div style='font-size:0.78rem;color:var(--muted);margin-bottom:0.5rem;'>Pilih judul yang tepat:</div>", unsafe_allow_html=True)
            chosen = st.selectbox("", matched, label_visibility="collapsed")
            cb, _ = st.columns([1,4])
            with cb:
                go = st.button("Cari Film Sejenis", use_container_width=True)
            if go:
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
            st.markdown("<div style='color:var(--muted);font-size:0.85rem;'>Tidak ada judul yang cocok. Coba kata kunci lain.</div>", unsafe_allow_html=True)

elif menu == "Surprise Me":
    st.markdown("<div class='ph'>SURPRISE ME</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Bingung mau nonton apa? Biarkan sistem yang memilih</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        motd = st.session_state.motd
        mt   = str(motd.get("title",""))
        mg   = str(motd.get("listed_in",""))
        md   = str(motd.get("description",""))
        mp, mid = fetch_tmdb(mt)
        tu  = trailer_url(mt, mid)
        pu  = tmdb_page_url(mid) or "#"
        ph  = f'<img src="{mp}" style="width:100%;border-radius:10px;margin-bottom:1rem;" loading="lazy">' if mp else ""
        st.markdown(f'''<div class="motd-card"><div class="motd-badge">Film Hari Ini</div>{ph}<div class="motd-title">{mt.upper()}</div><div class="motd-genre">{mg}</div><div class="motd-desc">{md}</div><div style="margin-top:1rem;display:flex;gap:8px;"><a class="btn-t" href="{tu}" target="_blank" rel="noopener">Trailer</a><a class="btn-s" href="{pu}" target="_blank" rel="noopener">TMDB</a></div></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown('''<div style="background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:1.5rem;"><div class="motd-badge">Random Roulette</div><div style="font-size:0.85rem;color:var(--muted);margin-bottom:1.25rem;">Klik tombol di bawah untuk mendapatkan pilihan film acak dari ribuan judul Netflix.</div></div>''', unsafe_allow_html=True)
        if st.button("Putar Roulette", use_container_width=True):
            st.session_state.random_pick = df.sample(1).iloc[0]
        if st.session_state.random_pick is not None:
            r   = st.session_state.random_pick
            rt  = str(r.get("title",""))
            rg  = str(r.get("listed_in",""))
            rd  = str(r.get("description",""))
            rp, rid = fetch_tmdb(rt)
            rtu = trailer_url(rt, rid)
            rpu = tmdb_page_url(rid) or "#"
            rph = f'<img src="{rp}" style="width:100%;border-radius:10px;margin:1rem 0;" loading="lazy">' if rp else ""
            st.markdown(f'''<div style="background:var(--bg3);border-radius:12px;padding:1.25rem;margin-top:0.75rem;">{rph}<div style="font-family:var(--display);font-size:1.6rem;letter-spacing:2px;color:var(--text);">{rt.upper()}</div><div style="font-size:0.8rem;color:var(--red);margin:0.3rem 0 0.6rem;">{rg}</div><div style="font-size:0.82rem;color:var(--muted);line-height:1.6;margin-bottom:0.75rem;">{rd}</div><a class="btn-t" href="{rtu}" target="_blank" rel="noopener">Trailer</a><a class="btn-s" href="{rpu}" target="_blank" rel="noopener" style="margin-left:8px;">TMDB</a></div>''', unsafe_allow_html=True)

elif menu == "Analytics":
    import matplotlib.pyplot as plt
    st.markdown("<div class='ph'>ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='ps'>Eksplorasi distribusi dataset Netflix secara visual</div>", unsafe_allow_html=True)
    st.markdown("<div class='sr'></div>", unsafe_allow_html=True)
    total  = len(df)
    nm     = len(df[df["type"]=="Movie"])   if "type" in df.columns else "—"
    ns     = len(df[df["type"]=="TV Show"]) if "type" in df.columns else "—"
    ng     = df["listed_in"].str.split(", ").explode().nunique() if "listed_in" in df.columns else "—"
    st.markdown(f'''<div class="stat-row"><div class="stat-card"><div class="stat-label">Total Judul</div><div class="stat-value">{total:,}</div></div><div class="stat-card"><div class="stat-label">Movies</div><div class="stat-value"><span class="stat-accent">{nm}</span></div></div><div class="stat-card"><div class="stat-label">TV Shows</div><div class="stat-value">{ns}</div></div><div class="stat-card"><div class="stat-label">Genre Unik</div><div class="stat-value">{ng}</div></div></div>''', unsafe_allow_html=True)
    plt.rcParams.update({"figure.facecolor":"#18181F","axes.facecolor":"#18181F","axes.edgecolor":"#2A2A35","axes.labelcolor":"#8A8790","xtick.color":"#8A8790","ytick.color":"#8A8790","text.color":"#F0EDE8","grid.color":"#2A2A35","grid.linestyle":"--","grid.alpha":0.4})
    co1,co2 = st.columns(2, gap="medium")
    with co1:
        st.markdown("<div class='st2' style='font-size:1.1rem;'>MOVIE VS TV SHOW</div>", unsafe_allow_html=True)
        if "type" in df.columns:
            fig,ax = plt.subplots(figsize=(5,3.5))
            ct = df["type"].value_counts()
            bs = ax.bar(ct.index, ct.values, color=["#E8232A","#F0A500"], width=0.45, zorder=3)
            ax.yaxis.grid(True,zorder=0); ax.set_axisbelow(True); ax.spines[:].set_visible(False); ax.tick_params(length=0)
            for b in bs:
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+50,f"{b.get_height():,}",ha="center",va="bottom",fontsize=10,color="#F0EDE8",fontweight="bold")
            fig.tight_layout(); st.pyplot(fig)
    with co2:
        st.markdown("<div class='st2' style='font-size:1.1rem;'>TREN RILIS PER TAHUN</div>", unsafe_allow_html=True)
        if "release_year" in df.columns:
            fig2,ax2 = plt.subplots(figsize=(5,3.5))
            yd = df["release_year"].value_counts().sort_index()
            ax2.plot(yd.index,yd.values,color="#E8232A",linewidth=2,zorder=3)
            ax2.fill_between(yd.index,yd.values,alpha=0.12,color="#E8232A")
            ax2.yaxis.grid(True,zorder=0); ax2.set_axisbelow(True); ax2.spines[:].set_visible(False); ax2.tick_params(length=0)
            fig2.tight_layout(); st.pyplot(fig2)
    st.markdown("<br><div class='st2' style='font-size:1.1rem;'>TOP 10 GENRE</div>", unsafe_allow_html=True)
    if "listed_in" in df.columns:
        gc = df["listed_in"].str.split(", ").explode().value_counts().head(10)
        fig3,ax3 = plt.subplots(figsize=(10,3.5))
        bs3 = ax3.barh(gc.index[::-1],gc.values[::-1],color="#E8232A",height=0.55,zorder=3)
        ax3.xaxis.grid(True,zorder=0); ax3.set_axisbelow(True); ax3.spines[:].set_visible(False); ax3.tick_params(length=0)
        for b in bs3:
            ax3.text(b.get_width()+8,b.get_y()+b.get_height()/2,f"{int(b.get_width()):,}",va="center",fontsize=9,color="#8A8790")
        fig3.tight_layout(); st.pyplot(fig3)

st.markdown('''<div class="footer">MOVREC &nbsp;·&nbsp; Content-Based Filtering &nbsp;·&nbsp; TF-IDF + Cosine Similarity<br><span style="opacity:0.5;">Data Science Beginner &nbsp;·&nbsp; 2024</span></div>''', unsafe_allow_html=True)
