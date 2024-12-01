from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process 
from flask_cors import CORS
import ast
import requests


# TMDb API Key
TMDB_API_KEY = 'b7f523542649c0e44bd6c12a9d753c90'

# Load and preprocess data
movies = pd.read_csv('tmdb_5000_movies.csv').merge(
    pd.read_csv('tmdb_5000_credits.csv'), left_on='id', right_on='movie_id', how='left'
)

def parse_data(data):
    try:
        return [item['name'] for item in ast.literal_eval(data)]
    except:
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

# Function to fetch recommendations
def get_recommendations(title):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    print("Recommendations list: ", movies['title_x'].iloc[movie_indices].tolist())
    return movies['title_x'].iloc[movie_indices].tolist()

# Fetch movie poster URL from TMDb
def get_movie_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results')
        if results:
            poster_path = results[0].get('poster_path')
            if poster_path:
                print("Poster path: ", poster_path)
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/150"  # Fallback image

def get_closest_match(title):
    titles = movies['title_x'].tolist()
    closest_match = process.extractOne(title, titles)
    return closest_match[0] if closest_match else None

# Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def recommend():
    print("Inside recommend function")
    #movie = request.form.get('movie')
    data = request.get_json()
    movie = data.get('movie_name')  # Extract 'movie' field

    if not movie:
        return {"error": "Movie name not provided"}, 400


    # Find the closest matching title
    closest_title = get_closest_match(movie)
    if not closest_title:
        return render_template('index.html', movie=None, recommendations=[])

    print("Closest Title is: ",closest_title)
    # Get recommendations for the matched title
    recommendations = get_recommendations(closest_title)
    
    # Fetch posters for recommendations
    recommendations_with_posters = [
        {"title": rec, "poster": get_movie_poster(rec)} for rec in recommendations
    ]
    print(recommendations_with_posters)
    return recommendations_with_posters
    #return render_template('index.html', movie=closest_title, recommendations=recommendations_with_posters)

if __name__ == '__main__':
    app.run(debug=True)
