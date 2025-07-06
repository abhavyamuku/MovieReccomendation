from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the dataset
movies = pd.read_csv('C:/Users/mukua/OneDrive/Desktop/Movie recco/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/mukua/OneDrive/Desktop/Movie recco/tmdb_5000_credits.csv')

# Preprocess
movies = movies[['title', 'overview', 'genres', 'vote_average', 'release_date']]
movies['overview'] = movies['overview'].fillna('')
movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

# Convert genres (which is a JSON-like string) to list of genre names
import ast

def extract_genre_names(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genre_list]
    except:
        return []

movies['genre_names'] = movies['genres'].apply(extract_genre_names)

# Vectorize overview text
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(movie_title, selected_genre=None, min_year=None, min_rating=None, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found in dataset. Please try another."]

    idx = movies.index[movies['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Exclude the movie itself

    recommended = []
    for i, score in sim_scores:
        row = movies.iloc[i]

        # Apply genre filter
        if selected_genre and selected_genre != 'All':
            if selected_genre not in row['genre_names']:
                continue

        # Apply year filter
        if min_year and not pd.isna(row['release_year']):
            if row['release_year'] < int(min_year):
                continue

        # Apply rating filter
        if min_rating:
            if row['vote_average'] < float(min_rating):
                continue

        recommended.append(row['title'])

        if len(recommended) >= top_n:
            break

    return recommended if recommended else ["No recommendations matched the filters."]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    selected_genre = request.form.get('genre')
    min_year = request.form.get('min_year')
    min_rating = request.form.get('min_rating')

    recommended_movies = recommend_movie(
        movie_title,
        selected_genre=selected_genre,
        min_year=min_year,
        min_rating=min_rating
    )

    return render_template('index.html',
                           movie_title=movie_title,
                           recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
