import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load datasets
movies = pd.read_csv("C:\\Users\\DELL\\Downloads\\ml-movie\\data\\movies.csv")
ratings = pd.read_csv("C:\\Users\\DELL\\Downloads\\ml-movie\\data\\ratings.csv")
tags = pd.read_csv("C:\\Users\\DELL\\Downloads\\ml-movie\\data\\tags.csv")
links = pd.read_csv("C:\\Users\\DELL\\Downloads\\ml-movie\\data\\links.csv")

# Merge datasets
df = ratings.merge(movies, on='movieId')
merged = movies.merge(tags, on="movieId", how="left").fillna("")
merged['all_text'] = merged['title'] + " " + merged['genres'] + " " + merged['tag']

# Create user-item matrix
df['title'] = df['title'] + " (" + df['movieId'].astype(str) + ")"
user_item_matrix = df.pivot(index='userId', columns='title', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Content-based filtering with tags
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(merged['all_text'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
content_similarity_df = pd.DataFrame(content_similarity, index=merged['title'], columns=merged['title'])

# Collaborative filtering function
def recommend_collaborative(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:6]
    similar_users_items = user_item_matrix.loc[similar_users].mean(axis=0)
    recommended_items = similar_users_items.sort_values(ascending=False).index[:num_recommendations]
    return list(recommended_items)

# Content-based filtering function
def recommend_content(movie_title, num_recommendations=5):
    similar_items = content_similarity_df[movie_title].sort_values(ascending=False).index[1:num_recommendations+1]
    return list(similar_items)

# Flask API Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend/collaborative', methods=['GET'])
def recommend_user():
    user_id = int(request.args.get('user_id'))
    recommendations = recommend_collaborative(user_id)
    return jsonify(recommendations)

@app.route('/recommend/content', methods=['GET'])
def recommend_movie():
    movie_title = request.args.get('movie_title')
    recommendations = recommend_content(movie_title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)