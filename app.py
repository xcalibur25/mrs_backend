from flask import Flask, request, jsonify,render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from flask_cors import CORS
import ast
import requests

# TMDb API Key
TMDB_API_KEY = 'b7f523542649c0e44bd6c12a9d753c90'

# Global variables for datasets and similarity computation
movies = None
indices = None
cosine_sim = None

# Function to load and preprocess the datasets
def load_movies():
    global movies, indices, cosine_sim  # Declare globals
    if movies is None:  # Load only once
        try:
            # Dataset URLs
            movies_url = "https://drive.google.com/uc?id=1e9T4xEdvGY9HpqB5tEjsxUY6q5mizzgo"
            credits_url = "https://drive.google.com/uc?id=1493yOsypvsKxCWT7y0e1TTXZlvMkLID5"

            # Load and merge datasets
            movies = pd.read_csv(movies_url).merge(
                pd.read_csv(credits_url), left_on='id', right_on='movie_id', how='left'
            )

            # Preprocess data
            def parse_data(data):
                try:
                    return [item['name'] for item in ast.literal_eval(data)]
                except (ValueError, SyntaxError, TypeError):
                    return []

            movies['genres'] = movies['genres'].apply(parse_data)
            movies['keywords'] = movies['keywords'].apply(parse_data)
            movies['overview'] = movies['overview'].fillna('')
            movies['title_x'] = movies['title_x'].fillna('')

            # Compute similarity
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(movies['overview'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Create reverse index
            indices = pd.Series(movies.index, index=movies['title_x']).drop_duplicates()
        except Exception as e:
            raise RuntimeError(f"Failed to load and preprocess datasets: {e}")

# Function to fetch recommendations
def get_recommendations(title):
    global movies, indices, cosine_sim
    load_movies()  # Ensure data is loaded
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title_x'].iloc[movie_indices].tolist()

# Fetch movie poster URL from TMDb
def get_movie_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            poster_path = results[0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/150"  # Fallback image

# Find the closest matching movie title
def get_closest_match(title):
    global movies
    load_movies()  # Ensure data is loaded
    titles = movies['title_x'].tolist()
    closest_match = process.extractOne(title, titles)
    return closest_match[0] if closest_match else None

# Flask app
app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     #return "index.html"
#     return render_template('index.html')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def recommend():
    try:
        # Get movie name from the request
        #data = request.get_json()
        #movie = data.get('movie_name')
        movie = request.form.get('movie')
        #print("Movie name: ", movie)
        if not movie:
            return jsonify({"error": "Movie name not provided"}), 400

        # Find closest match
        closest_title = get_closest_match(movie)
        #print("Closest title: ", closest_title)
        if not closest_title:
            return jsonify({"error": "No matching movie found"}), 404

        # Fetch recommendations and their posters
        recommendations = get_recommendations(closest_title)
        #print("Movie Recommendations: \n", recommendations)
        recommendations_with_posters = [
            {"title": rec, "poster": get_movie_poster(rec)} for rec in recommendations
        ]

        #print(recommendations_with_posters)
        #return recommendations_with_posters
        return render_template('index.html', movie=closest_title, recommendations=recommendations_with_posters)
        #return jsonify({"movie": closest_title, "recommendations": recommendations_with_posters})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #app.run(debug=True)
    #if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)

