import streamlit as st

st.title("📚 Book Recommender system")

st.markdown("### 👋 Welcome to the Book Recommender System!")
st.markdown("Here you can explore book recommendations.")

st.markdown("---")  # Optional horizontal line for separation
st.markdown("### 🔗 Go to:")
st.page_link("pages/CollaborativeFiltering.py", label="📖 Collaborative Filtering")
st.page_link("pages/Hybrid.py", label="📔 Hybrid")

st.sidebar.header("📌 Naviagtion")
st.sidebar.page_link("pages/CollaborativeFiltering.py", label="📖  Collaborative Filtering")
st.sidebar.page_link("pages/Hybrid.py", label="📔 Hybrid")
