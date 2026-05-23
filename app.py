import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import random

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="MovRec",
    layout="wide"
)

# =============================
# LOAD MODEL
# =============================
df = joblib.load('movies.pkl')
tfidf = joblib.load('tfidf.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

# =============================
# TMDB API (POSTER)
# =============================
API_KEY = "0758644da67e27b71fac69a53fab875e"

def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
    data = requests.get(url).json()

    try:
        poster = data['results'][0]['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster
    except:
        return None
    
# TEST API
st.write(fetch_poster("Avatar"))

# =============================
# STYLE
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1 {
    text-align: center;
}
.movie-card {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    transition: 0.2s;
}
.movie-card:hover {
    transform: scale(1.05);
}
.poster {
    border-radius: 10px;
}
.section {
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("<h1>MovRec</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Movie Recommendation System</p>", unsafe_allow_html=True)

# =============================
# MOOD
# =============================
mood_keywords = {
    "Happy": "comedy fun",
    "Sad": "drama emotional",
    "Excited": "action thriller",
    "Romantic": "romance love",
    "Sci-Fi": "science fiction",
    "Chill": "documentary calm"
}

# =============================
# PIPELINE
# =============================
def recommend_by_mood(mood):
    keywords = mood_keywords[mood]
    user_vector = tfidf.transform([keywords])
    similarity = cosine_similarity(user_vector, tfidf_matrix)

    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:10]

# =============================
# DISPLAY GRID
# =============================
def display_movies(indices):
    cols = st.columns(5)

    for i, idx in enumerate(indices):
        with cols[i % 5]:
            title = df.iloc[idx]['title']
            poster = fetch_poster(title)

            if poster:
                st.image(poster)

            st.markdown(f"<div class='movie-card'>{title}</div>", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Mood", "Search", "Genre", "Top", "Random"]
)

# =============================
# HOME (HERO SECTION)
# =============================
if menu == "Home":

    movie = df.sample(1)['title'].values[0]
    poster = fetch_poster(movie)

    st.markdown("<div class='section'>", unsafe_allow_html=True)

    if poster:
        st.image(poster, use_container_width=True)

    st.markdown(f"""
    <h2>{movie}</h2>
    <p style='color:gray;'>Featured today</p>
    """, unsafe_allow_html=True)

# =============================
# MOOD
# =============================
elif menu == "Mood":

    st.subheader("Mood Recommendation")

    mood = st.selectbox("Select Mood", list(mood_keywords.keys()))

    if st.button("Recommend"):

        results = recommend_by_mood(mood)
        indices = [i[0] for i in results]

        display_movies(indices)

# =============================
# SEARCH
# =============================
elif menu == "Search":

    st.subheader("Search Movie")

    query = st.text_input("Enter title")

    if query:
        results = df[df['title'].str.contains(query, case=False, na=False)]

        cols = st.columns(5)

        for i in range(min(10, len(results))):
            with cols[i % 5]:
                title = results.iloc[i]['title']
                poster = fetch_poster(title)

                if poster:
                    st.image(poster)

                st.markdown(f"<div class='movie-card'>{title}</div>", unsafe_allow_html=True)

# =============================
# GENRE
# =============================
elif menu == "Genre":

    st.subheader("Browse by Genre")

    genre = st.selectbox("Genre", ["Action", "Drama", "Comedy", "Romance", "Horror"])

    filtered = df[df['listed_in'].str.contains(genre, case=False, na=False)]

    display_movies(filtered.index[:10])

# =============================
# TOP
# =============================
elif menu == "Top":

    st.subheader("Top Action Movies")

    action = df[df['listed_in'].str.contains("Action", case=False, na=False)]

    display_movies(action.index[:10])

# =============================
# RANDOM
# =============================
elif menu == "Random":

    st.subheader("Random Movie")

    if st.button("Pick"):

        movie = df.sample(1)['title'].values[0]
        poster = fetch_poster(movie)

        if poster:
            st.image(poster)

        st.markdown(f"<h2>{movie}</h2>", unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>MovRec System</p>", unsafe_allow_html=True)