import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score

# Load Books Dataset
@st.cache_data
def load_books():
    return pd.read_csv("./input/book-recommendation-dataset/Books.csv")

# Load Ratings Dataset
@st.cache_data
def load_ratings():
    return pd.read_csv("./input/book-recommendation-dataset/Ratings.csv")

# Load pre-trained model
@st.cache_resource
def load_model():
    with open("Trainedmodel/hybrid_models.pkl", "rb") as f:
        return pickle.load(f)

books = load_books()
ratings = load_ratings()
model = load_model()

vectorizer = model['vectorizer']
svd = model['svd']
knn = model['knn']

# Filter & Prepare Data
filtered_data = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].dropna()

# Transform to get embeddings using pretrained models
tfidf_matrix = vectorizer.transform(
    filtered_data['Book-Title'] + " " +
    filtered_data['Book-Author'] + " " +
    filtered_data['Publisher'] + " " +
    filtered_data['Year-Of-Publication'].astype(str)
)
book_embeddings = svd.transform(tfidf_matrix)

# Hybrid Recommendation Function
def recommend_books_with_ratings(book_title, alpha=0.5):
    matches = filtered_data[filtered_data['Book-Title'].str.contains(book_title, case=False, na=False)]
    if matches.empty:
        return None, None, None

    closest_book = matches.iloc[0]
    book_index = filtered_data.index[filtered_data['Book-Title'] == closest_book['Book-Title']].tolist()
    if not book_index:
        return None, None, None
    book_index = book_index[0]

    distances, indices = knn.kneighbors([book_embeddings[book_index]], n_neighbors=min(20, len(filtered_data)))

    recommended_books = [closest_book['Book-Title']]
    predicted_ratings = [5]

    for j in range(1, len(indices[0])):
        i = indices[0][j]
        if i >= len(filtered_data):
            continue

        rec_book = filtered_data.iloc[i]
        if rec_book['Book-Title'] in recommended_books:
            continue

        exclude_keywords = ["Audio", "CD", "Music", "Postcard", "Method", "Guide", "Manual", "Pop-Up", "Nonpareil"]
        if any(keyword in rec_book['Book-Title'] for keyword in exclude_keywords):
            continue

        content_similarity_score = 1 / (1 + distances[0][j])
        isbn_list = books[books['Book-Title'] == rec_book['Book-Title']]['ISBN']
        book_ratings = ratings[ratings['ISBN'].isin(isbn_list)]['Book-Rating']
        collaborative_rating = book_ratings.mean() if len(book_ratings) > 0 else 3

        hybrid_score = alpha * content_similarity_score * 5 + (1 - alpha) * collaborative_rating
        recommended_books.append(rec_book['Book-Title'])
        predicted_ratings.append(hybrid_score)

        if len(recommended_books) >= 10:
            break

    return closest_book, recommended_books, predicted_ratings

# Evaluation Helpers
def get_actual_ratings(recommended_books):
    actual_ratings = []
    for book_title in recommended_books:
        isbn_list = books[books['Book-Title'] == book_title]['ISBN']
        book_ratings = ratings[ratings['ISBN'].isin(isbn_list)]['Book-Rating']
        actual_ratings.append(book_ratings.mean() if len(book_ratings) >= 10 else None)
    return actual_ratings

def evaluate_model(book_title):
    closest_book, recommended_books, predicted_ratings = recommend_books_with_ratings(book_title)
    actual_ratings = get_actual_ratings(recommended_books)
    valid_ratings = [(p, a) for p, a in zip(predicted_ratings, actual_ratings) if a is not None]

    if not valid_ratings:
        return closest_book, recommended_books, None, None, None, None

    predicted_ratings, actual_ratings = zip(*valid_ratings)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    threshold = 3
    y_true = [1 if rating >= threshold else 0 for rating in actual_ratings]
    y_pred = [1 if rating >= threshold else 0 for rating in predicted_ratings]

   

    return closest_book, recommended_books, rmse

# Streamlit UI
st.title("📚 Hybrid Book Recommender System")
st.write("Enter a book title to get hybrid recommendations based on content and user preferences.")

input_book = st.text_input("📖 Enter a book title:")

if st.button("Recommend"):
    if input_book:
        closest_book, recommended_books, rmse = evaluate_model(input_book)

        if not recommended_books:
            st.error("No matching books found!")
        else:
            st.success(f"Closest match: '**{closest_book['Book-Title']}**' by **{closest_book['Book-Author']}**")
            st.subheader(f"📚 Books similar to '**{closest_book['Book-Title']}**':")
            st.divider()
            for book_title in recommended_books:
                st.write(f"📖 **{book_title}**")
                st.divider()

            if rmse is not None:
                st.subheader(f"📊 RMSE: **{rmse:.4f}**")
            else:
                st.warning("⚠ Not enough data to evaluate RMSE/Precision/Recall.")
    else:
        st.error("Please enter a book title!")
