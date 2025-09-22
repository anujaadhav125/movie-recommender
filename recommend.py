import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("./dataset/tmdb_5000_movies.csv")

# Keep only useful columns (id, title, overview)
movies = movies[['id', 'title', 'overview']]
movies.rename(columns={'id': 'movie_id'}, inplace=True)

# Fill NaN with empty string
movies['overview'] = movies['overview'].fillna('')

# Convert overview text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Function to recommend movies
def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found in dataset"]
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Test
print(recommend('Avatar'))
