import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, movies_csv, ratings_csv=None):
        self.movies = pd.read_csv(movies_csv)
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['combined'] = (self.movies['title'].fillna('') + ' ' +
                                   self.movies['genres'] + ' ' +
                                   self.movies['overview'])
        self._build_content_model()

        self.ratings = None
        self.user_item_matrix = None
        self.cf_similarity = None
        if ratings_csv is not None:
            self._load_ratings(ratings_csv)

    def _build_content_model(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies['combined'])
        self.content_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _load_ratings(self, ratings_csv):
        self.ratings = pd.read_csv(ratings_csv)
        pivot = self.ratings.pivot_table(index='userId', columns='movieId', values='rating')
        self.user_item_matrix = pivot.fillna(0)
        if not self.user_item_matrix.empty:
            self.cf_similarity = cosine_similarity(self.user_item_matrix)
            self.cf_similarity = pd.DataFrame(self.cf_similarity,
                                              index=self.user_item_matrix.index,
                                              columns=self.user_item_matrix.index)

    def _get_movie_index(self, title):
        mask = self.movies['title'].str.lower() == title.lower()
        if not mask.any():
            return None
        return int(self.movies[mask].index[0])

    def content_recs(self, title, top_n=10):
        idx = self._get_movie_index(title)
        if idx is None:
            return []
        sim_scores = list(enumerate(self.content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n+1]
        indices = [i for i, s in sim_scores]
        return self.movies.iloc[indices][['movieId', 'title']].to_dict('records')

    def collaborative_recs_for_user(self, user_id, top_n=10):
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index.tolist()
        scores = {}
        for other_user in self.user_item_matrix.index:
            if other_user == user_id:
                continue
            sim = self.cf_similarity.at[user_id, other_user]
            if sim <= 0:
                continue
            other_ratings = self.user_item_matrix.loc[other_user]
            for movie_id in unrated:
                score = sim * other_ratings.get(movie_id, 0)
                scores[movie_id] = scores.get(movie_id, 0) + score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        movie_ids = [m for m, sc in ranked]
        return self.movies[self.movies['movieId'].isin(movie_ids)][['movieId', 'title']].to_dict('records')

    def hybrid_recommend(self, title=None, user_id=None, top_n=10, alpha=0.6):
        content = []
        cf = []

        if title:
            content = self.content_recs(title, top_n=top_n*2)

        if user_id and self.user_item_matrix is not None:
            cf = self.collaborative_recs_for_user(user_id, top_n=top_n*2)

        content_df = pd.DataFrame(content)
        cf_df = pd.DataFrame(cf)

        combined = pd.concat([content_df, cf_df]).drop_duplicates(subset='movieId', keep='first')
        if combined.empty:
            return []

        combined['score'] = 0.0
        if not content_df.empty:
            combined = combined.merge(content_df[['movieId']], on='movieId', how='left', indicator='in_content')
            combined['in_content_score'] = combined['in_content'].apply(lambda x: 1.0 if x == 'both' or x == 'left_only' else 0.0)
        else:
            combined['in_content_score'] = 0.0

        if not cf_df.empty:
            combined = combined.merge(cf_df[['movieId']], on='movieId', how='left', indicator='in_cf')
            combined['in_cf_score'] = combined['in_cf'].apply(lambda x: 1.0 if x == 'both' or x == 'left_only' else 0.0)
        else:
            combined['in_cf_score'] = 0.0

        combined['score'] = alpha * combined['in_content_score'] + (1 - alpha) * combined['in_cf_score']
        combined = combined.sort_values('score', ascending=False).head(top_n)
        return combined[['movieId', 'title', 'score']].to_dict('records')
