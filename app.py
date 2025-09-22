# ==========================
# Movie Recommender App
# ==========================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --------------------------
# Load dataset
# --------------------------
movies = pd.read_csv("dataset/All_Movies1.csv", engine="python", on_bad_lines="skip")

# Ensure proper columns
movies = movies[['title', 'overview', 'genres', 'release_date', 'runtime', 'vote_average', 'poster_path', 'Director', 'Cast']]
movies.rename(columns={'title': 'Title', 'overview': 'Overview', 'genres':'Genre', 'release_date':'Release', 'poster_path':'Poster', 'vote_average':'Rating'}, inplace=True)
movies['Overview'] = movies['Overview'].fillna('')
movies['Release'] = movies['Release'].astype(str).fillna('0000')  # Fix for release year

# --------------------------
# Dark/Light Mode
# --------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    bg_color = "#181818"
    text_color = "#FFFFFF"
    card_color = "#222222"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    card_color = "#f9f9f9"

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .st-expander {{
            background-color: {card_color};
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {card_color};
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Vectorization for Recommendations
# --------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['Overview']).toarray()
similarity = cosine_similarity(vectors)

# --------------------------
# Recommendation Function
# --------------------------
def recommend(movie):
    if movie not in movies['Title'].values:
        return []
    index = movies[movies['Title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    results = []
    for i in movie_list:
        m = movies.iloc[i[0]]
        results.append((m.Title, m.Overview, m.Poster, m.Genre, m.Release, m.Rating))
    return results

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Hollywood & Bollywood Movie Recommender")
st.markdown("Find movies based on storyline, genre, actor/director, and more!")

# --------------------------
# Filters
# --------------------------
filtered_movies = movies.copy()

# Genre filter
all_genres = sorted(filtered_movies['Genre'].dropna().unique())
selected_genres = st.multiselect("Filter by Genre:", all_genres)
if selected_genres:
    filtered_movies = filtered_movies[filtered_movies['Genre'].isin(selected_genres)]

# Year filter (Fixed)
release_years = pd.to_numeric(filtered_movies['Release'].str[:4], errors='coerce')
min_year, max_year = int(release_years.min()), int(release_years.max())
year_range = st.slider("Release Year:", min_year, max_year, (min_year, max_year))
filtered_movies = filtered_movies[
    (release_years >= year_range[0]) &
    (release_years <= year_range[1])
]

# --------------------------
# Runtime filter (Fixed)
# --------------------------
def parse_runtime(rt):
    if pd.isna(rt):
        return 0
    if isinstance(rt, (int, float)):
        return int(rt)
    match = re.search(r'(\d+)', str(rt))
    return int(match.group(1)) if match else 0

filtered_movies['runtime_minutes'] = filtered_movies['runtime'].apply(parse_runtime)
min_runtime = int(filtered_movies['runtime_minutes'].min())
max_runtime = int(filtered_movies['runtime_minutes'].max())
runtime_range = st.slider("Movie Duration (minutes):", min_runtime, max_runtime, (min_runtime, max_runtime))
filtered_movies = filtered_movies[
    (filtered_movies['runtime_minutes'] >= runtime_range[0]) &
    (filtered_movies['runtime_minutes'] <= runtime_range[1])
]

# Actor/Director search
actor_director_input = st.text_input("Search by Actor or Director:")
if actor_director_input.strip():
    filtered_movies = filtered_movies[
        filtered_movies['Cast'].str.contains(actor_director_input.strip(), case=False, na=False) |
        filtered_movies['Director'].str.contains(actor_director_input.strip(), case=False, na=False)
    ]

# Movie Dropdown
movie_name = st.selectbox("Select a Movie for Recommendation:", filtered_movies['Title'].values)

# --------------------------
# Watchlist
# --------------------------
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# --------------------------
# Top Rated / Popular Movies
# --------------------------
st.subheader("⭐ Top Rated / Popular Movies")
top_movies = filtered_movies.sort_values(by='Rating', ascending=False).head(6)
for i in range(0, len(top_movies), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        if i+j < len(top_movies):
            t = top_movies.iloc[i+j]
            poster_url = "https://image.tmdb.org/t/p/w500" + t.Poster if t.Poster else None
            with col.expander(f"{t.Title} ⭐ {t.Rating}"):
                if poster_url:
                    st.image(poster_url, width=180)
                st.markdown(f"**Genre:** {t.Genre}<br>**Release:** {t.Release}<br>**Rating:** {t.Rating}", unsafe_allow_html=True)
                # YouTube trailer link
                youtube_link = f"https://www.youtube.com/results?search_query={t.Title.replace(' ', '+')}+trailer"
                st.markdown(f"[🎬 Watch Trailer on YouTube]({youtube_link})", unsafe_allow_html=True)
                
# --------------------------
# Display Selected Movie Overview
# --------------------------
if movie_name:
    selected_movie = filtered_movies[filtered_movies['Title'] == movie_name].iloc[0]
    st.subheader(f"🎬 {selected_movie.Title}")
    poster_url = "https://image.tmdb.org/t/p/w500" + selected_movie.Poster if selected_movie.Poster else None
    if poster_url:
        st.image(poster_url, width=250)
    st.markdown(f"**Genre:** {selected_movie.Genre}")
    st.markdown(f"**Release:** {selected_movie.Release}")
    st.markdown(f"**Rating:** {selected_movie.Rating}")
    st.markdown(f"**Overview:** {selected_movie.Overview}")
    youtube_link = f"https://www.youtube.com/results?search_query={selected_movie.Title.replace(' ', '+')}+trailer"
    st.markdown(f"[🎬 Watch Trailer on YouTube]({youtube_link})", unsafe_allow_html=True)


# --------------------------
# Recommendations
# --------------------------
if st.button("Recommend"):
    if movie_name:
        recommendations = recommend(movie_name)
        if recommendations:
            st.subheader("🎯 Recommended Movies")
            for i in range(0, len(recommendations), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i+j < len(recommendations):
                        title, overview, poster, genre, release, rating = recommendations[i+j]
                        poster_url = "https://image.tmdb.org/t/p/w500" + poster if poster else None
                        youtube_link = f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+trailer"

                        with col.expander(f"{title} ⭐ {rating}"):
                            if poster_url:
                                st.image(poster_url, width=180)
                            st.markdown(f"**Title:** {title}")
                            st.markdown(f"**Genre:** {genre}")
                            st.markdown(f"**Release:** {release}")
                            st.markdown(f"**Rating:** {rating}")
                            st.markdown(f"**Overview:** {overview}")
                            st.markdown(f"[🎬 Watch Trailer on YouTube]({youtube_link})", unsafe_allow_html=True)

                            # Watchlist button
                            if st.button("➕ Add to Watchlist", key=title):
                                if title not in st.session_state['watchlist']:
                                    st.session_state['watchlist'].append(title)
                                    st.success(f"Added '{title}' to your watchlist!")

# --------------------------
# Display Watchlist
# --------------------------
if st.session_state['watchlist']:
    st.markdown("---")
    st.subheader("🎬 Your Watchlist")
    for w in st.session_state['watchlist']:
        st.markdown(f"- {w}")
