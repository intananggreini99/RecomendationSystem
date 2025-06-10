import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def load_places():
    return pd.read_csv("D:/AI_KecerdasanBuatan/RecomendationSystem/data/tourism_with_id.csv")

def load_ratings():
    return pd.read_csv("D:/AI_KecerdasanBuatan/RecomendationSystem/data/tourism_rating.csv")

def preprocess_places(df):
    df['content'] = df['Category'] + " " + df['Description']
    return df

def get_content_recs(places, ratings, user_id, topn=10):
    places = preprocess_places(places)
    tfidf = TfidfVectorizer()
    M = tfidf.fit_transform(places['content'])
    user_rated = ratings[ratings['User_Id']==user_id]
    user_places = user_rated.merge(places, on='Place_Id')
    
    # Rata-rata centroid konten
    vects = tfidf.transform(user_places['content'])
    centroid = vects.mean(axis=0)
    
    sims = cosine_similarity(centroid, M).flatten()
    
    places['score'] = sims
    return places.sort_values('score', ascending=False).head(topn)

def get_collab_recs(places, ratings, user_id, topn=10):
    # user-user basic
    pivot = ratings.pivot_table(index='User_Id', columns='Place_Id', values='rating').fillna(0)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(pivot)
    dist, idx = knn.kneighbors([pivot.loc[user_id]])
    similar_uid = pivot.index[idx.flatten()[1]]
    
    sim_ratings = ratings[ratings['User_Id'].isin(similar_uid)]
    mean_ratings = sim_ratings.groupby('Place_Id')['rating'].mean()
    recs = places.merge(mean_ratings, on='Place_Id')
    recs = recs[~recs['Place_Id'].isin(ratings[ratings['User_Id']==user_id]['Place_Id'])]
    recs = recs.rename(columns={'rating':'score'})
    return recs.sort_values('score', ascending=False).head(topn)
