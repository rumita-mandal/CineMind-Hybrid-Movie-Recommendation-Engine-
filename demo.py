from model import HybridRecommender
import pprint

rec = HybridRecommender("sample_data/movies.csv", ratings_csv="sample_data/ratings.csv")
print("=== Content-based recommendations for 'Inception' ===")
pprint.pprint(rec.content_recs("Inception", top_n=5))

print("\n=== Collaborative recommendations for user 1 ===")
pprint.pprint(rec.collaborative_recs_for_user(1, top_n=5))

print("\n=== Hybrid recommendations (title='Inception', user=1) ===")
pprint.pprint(rec.hybrid_recommend(title="Inception", user_id=1, top_n=6, alpha=0.6))
