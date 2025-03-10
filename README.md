# Movie Recommendation System

## Overview
This project is a **Movie Recommendation System** built using **Flask, Pandas, NumPy, Scikit-learn**, and **TfidfVectorizer**. It provides movie recommendations based on **Collaborative Filtering** and **Content-Based Filtering** techniques.

## Features
- **Collaborative Filtering**: Recommends movies based on user similarity using cosine similarity.
- **Content-Based Filtering**: Recommends movies based on movie genres and tags using TF-IDF.
- **Flask API**: Exposes endpoints for retrieving recommendations.
- **Interactive UI**: A simple web interface to enter a user ID and get movie recommendations.

## Datasets Used
This project uses the **MovieLens dataset** from GroupLens, which includes:
download from here as well 
`https://grouplens.org/datasets/movielens/latest/`

- `movies.csv`: Contains movie IDs, titles, and genres.
- `ratings.csv`: Contains user ratings for movies.
- `tags.csv`: Contains user-assigned tags for movies.
- `links.csv`: Contains IMDb and TMDb IDs for movies.

## Technologies Used
- **Python** (Flask, Pandas, NumPy, Scikit-learn)
- **JavaScript** (Fetch API for making API requests)
- **HTML & CSS** (Frontend UI)
- **OMDB API** (Optional: Used to fetch movie posters)

## How It Works
1. **Collaborative Filtering**
   - Builds a **user-item matrix** using movie ratings.
   - Computes **cosine similarity** between users.
   - Suggests movies based on similar users' preferences.

2. **Content-Based Filtering**
   - Combines movie **title, genres, and tags**.
   - Uses **TF-IDF vectorization** to convert text data into numerical features.
   - Computes **cosine similarity** between movies.
   - Recommends movies similar to the input movie.

## API Endpoints
### 1. Home Page
`GET /`
- Returns the main web page.

### 2. Collaborative Filtering Recommendation
`GET /recommend/collaborative?user_id=<user_id>`
- Input: `user_id` (integer)
- Output: List of recommended movies for the given user.

### 3. Content-Based Recommendation
`GET /recommend/content?movie_title=<movie_title>`
- Input: `movie_title` (string)
- Output: List of recommended movies similar to the given movie.

## Setup Instructions
### **1. Install Dependencies**
```sh
pip install flask pandas numpy scikit-learn requests
```

### **2. Run the Flask App**
```sh
python script.py
```

### **3. Access the Web UI**
Open `http://127.0.0.1:5000/` in your browser.

## Project Structure
```
ml-movie/
│── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── links.csv
│── static/
│   ├── styles.css
│── templates/
│   ├── index.html
│── script.py
│── README.md
```
