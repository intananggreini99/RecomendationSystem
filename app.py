import streamlit as st
st.set_page_config(layout="wide")

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

# === SIDEBAR DASHBOARD ===
st.sidebar.title("🧭 Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Tempat Serupa", "Rekomendasi User"])

# === LOAD DATA ===
@st.cache_data
def load_places():
    df = pd.read_csv("tourism_with_id.csv")
    if 'Lat' not in df.columns or 'Long' not in df.columns:
        df[['Lat', 'Long']] = df['Coordinate'].str.extract(r'\((.*),\s*(.*)\)').astype(float)
    return df

@st.cache_data
def load_ratings():
    df = pd.read_csv("tourism_rating.csv")
    df.drop_duplicates(inplace=True)
    return df

places = load_places()
ratings = load_ratings()

# === MAPPING USERNAME ===
username_to_userid = ratings.drop_duplicates(subset="username")[["username", "User_Id"]].set_index("username").to_dict()["User_Id"]
user_id_to_username = {v: k for k, v in username_to_userid.items()}

# === HALAMAN: TEMPAT SERUPA ===
if menu == "Tempat Serupa":
    st.title("""🔍 Rekomendasi Destinasi 
    Used Content Based Filtering""")
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(places['Category'])
    cos_sim = cosine_similarity(tfidf_matrix)
    cos_sim_df = pd.DataFrame(cos_sim, index=places.Place_Name, columns=places.Place_Name)

    def recommend_place(name, k=5):
        idx = cos_sim_df[name].to_numpy().argpartition(range(-1, -k, -1))
        similar = cos_sim_df.columns[idx[-1:-(k+2):-1]]
        similar = similar.drop(name, errors='ignore')
        return pd.DataFrame(similar).merge(places[['Place_Name', 'City','Category', 'Lat', 'Long']])

    selected_place = st.selectbox("Pilih Tempat Wisata", places['Place_Name'].unique())

    if st.button("Tampilkan Rekomendasi"):
        st.session_state['selected_place'] = selected_place

    if 'selected_place' in st.session_state:
        place = st.session_state['selected_place']
        st.markdown(f"**Rekomendasi berdasarkan:** {place}")
        result = recommend_place(place)
        st.dataframe(result[['Place_Name', 'City','Category']].rename(columns={
            'Place_Name': 'Nama Tempat',
            'City': 'Kota',
            'Category': 'Kategori'
        }))

        df_map = result.rename(columns={'Lat': 'latitude', 'Long': 'longitude'})
        if 'latitude' in df_map and 'longitude' in df_map:
            st.markdown("### 🗺️ Peta Lokasi Rekomendasi")
            st.map(df_map[['latitude', 'longitude']])

# === HALAMAN: REKOMENDASI USER ===
elif menu == "Rekomendasi User":

    st.title("Rekomendasi Destinasi - Collaborative Filtering")

    if 'selected_user' not in st.session_state:
        st.markdown("### 🔎 Cari Username")
        search_user = st.text_input("Masukkan nama pengguna:")

        if search_user:
            pattern = re.compile(re.escape(search_user), re.IGNORECASE)
            matched_users = [u for u in ratings['username'].unique() if pattern.search(u)]

            if matched_users:
                st.markdown("**Username yang cocok:**")
                for u in matched_users:
                    if st.button(f"👤 {u}", key=f"match_{u}"):
                        st.session_state['selected_user'] = u
                        st.rerun()
            else:
                st.warning("Tidak ada username yang cocok ditemukan.")

    if 'selected_user' in st.session_state:
        selected_username = st.session_state['selected_user']
        st.header(f"Rekomendasi untuk 👤 {selected_username}")

        # Encode user dan tempat
        user_enc = LabelEncoder()
        place_enc = LabelEncoder()
        ratings['user_id'] = user_enc.fit_transform(ratings['User_Id'])
        ratings['place_id'] = place_enc.fit_transform(ratings['Place_Id'])

        # Buat matriks user-item
        n_users = ratings['user_id'].nunique()
        n_items = ratings['place_id'].nunique()
        ratings_matrix = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            ratings_matrix[row.user_id, row.place_id] = row.Place_Ratings

        # SVD
        svd = TruncatedSVD(n_components=10, random_state=42)
        user_factors = svd.fit_transform(ratings_matrix)
        item_factors = svd.components_.T
        pred_matrix = np.dot(user_factors, item_factors.T)

        # Ambil rekomendasi
        user_idx = user_enc.transform([username_to_userid[selected_username]])[0]
        user_rated = ratings[ratings['user_id'] == user_idx]['place_id'].tolist()

        preds = pred_matrix[user_idx]
        unrated_idx = [i for i in range(n_items) if i not in user_rated]
        top_idx = sorted(unrated_idx, key=lambda x: preds[x], reverse=True)[:5]

        top_place_ids = place_enc.inverse_transform(top_idx)
        top_places = places[places['Place_Id'].isin(top_place_ids)]

        df_display = top_places[['Place_Name', 'City', 'Category']].rename(columns={
            'Place_Name': 'Nama Tempat',
            'City': 'Kota',
            'Category': 'Kategori'
        })

        st.markdown("### 📋 Daftar Destinasi")
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

        df_map = top_places.rename(columns={'Lat': 'latitude', 'Long': 'longitude'})
        if 'latitude' in df_map and 'longitude' in df_map:
            st.markdown("### 📍 Peta Lokasi Destinasi")
            st.map(df_map[['latitude', 'longitude']])

        if st.button("🔄 Ganti Pengguna"):
            del st.session_state['selected_user']
            st.rerun()
