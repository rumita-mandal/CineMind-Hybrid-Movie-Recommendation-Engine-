import streamlit as st
from model import HybridRecommender
import pandas as pd

st.set_page_config(page_title="CineMind — Hybrid Recommender", layout="wide")
st.title("🎬 CineMind — Hybrid Movie Recommendation Engine")

rec = HybridRecommender("sample_data/movies.csv", ratings_csv="sample_data/ratings.csv")

movie_list = rec.movies['title'].tolist()
selected_movie = st.selectbox("Choose a movie (content seed)", [""] + movie_list)
user_ids = []
if rec.user_item_matrix is not None:
    user_ids = rec.user_item_matrix.index.tolist()
selected_user = st.selectbox("Choose user (collaborative)", [""] + user_ids)

alpha = st.slider("Content / Collaborative weight (alpha)", 0.0, 1.0, 0.6)

if st.button("Get Recommendations"):
    title = selected_movie if selected_movie != "" else None
    user = selected_user if selected_user != "" else None
    results = rec.hybrid_recommend(title=title, user_id=user, top_n=10, alpha=alpha)
    if not results:
        st.write("No recommendations found. Try using a movie seed or a different user.")
    else:
        df = pd.DataFrame(results)
        st.table(df)
