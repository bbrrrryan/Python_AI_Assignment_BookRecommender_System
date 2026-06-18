import pickle
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error

# --- Load Models and Data ---
st.header("📚 Books Recommender System")
model = pickle.load(open('Trainedmodel/model.pkl', 'rb'))
books_name = pickle.load(open('Trainedmodel/books_name.pkl', 'rb'))
final_rating = pickle.load(open('Trainedmodel/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('Trainedmodel/book_pivot.pkl', 'rb'))

# --- Helper Functions ---
def fetch_poster(book_names):
    """
    Fetch poster URLs for the given book names.
    Makes sure the order matches the provided book names.
    """
    poster_urls = []
    
    for name in book_names:
        # Find the book in final_rating dataframe
        if name in final_rating['title'].values:
            idx = np.where(final_rating['title'] == name)[0][0]
            url = final_rating.iloc[idx]['img_url']
            poster_urls.append(url)
        else:
            # If book not found, use a placeholder
            poster_urls.append("https://via.placeholder.com/150?text=No+Image")
    
    return poster_urls

def recommend_books(book_name):
    """
    Get book recommendations and their posters for the given book name.
    Also calculates RMSE for the recommendations.
    """
    actual_ratings = []
    predicted_ratings = []

    # Find the book in the pivot table
    book_id = np.where(book_pivot.index == book_name)[0][0]
    
    # Get recommendations
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    # Initialize book_list with the searched book as the first item
    book_list = [book_name]
    
    # Add the recommendations (skip the first one as it's already the input book)
    for i in range(1, len(suggestion[0])):
        book = book_pivot.index[suggestion[0][i]]
        book_list.append(book)
        
        # Get actual rating
        actual = final_rating[final_rating['title'] == book]['rating'].mean() if book in final_rating['title'].values else 0
        actual_ratings.append(actual)
        
        # Calculate predicted rating based on neighbors
        neighbors = book_pivot.index[suggestion[0][1:]]
        predicted = np.mean([
            final_rating[final_rating['title'] == neighbor]['rating'].mean()
            if neighbor in final_rating['title'].values else 0
            for neighbor in neighbors
        ])
        predicted_ratings.append(predicted)
    
    # Calculate RMSE
    if actual_ratings and predicted_ratings:
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    else:
        rmse = 0
    
    # Get poster URLs in the same order as book_list
    poster_urls = fetch_poster(book_list)
    
    return book_list, poster_urls, rmse

# --- Streamlit User Interface ---
selected_book = st.selectbox(
    "📘 Type or select a book:",
    books_name
)

if st.button('Show Recommendation'):
    recommendation_books, poster_urls, rmse = recommend_books(selected_book)
    
    st.markdown(f"### 📊 RMSE for this recommendation: `{rmse:.2f}`")
    
    # Display recommendations with the selected book first
    cols = st.columns(6)  # Now showing 6 books (searched + 5 recommendations)
    
    for i, col in enumerate(cols):
        if i < len(recommendation_books):
            with col:
                st.text(recommendation_books[i])
                st.image(poster_urls[i])