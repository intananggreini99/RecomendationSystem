import streamlit as st
st.set_page_config(layout="wide")

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers

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
    st.title("🔍 Rekomendasi Destinasi - Content Based Filtering")
    
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

    # Jika belum ada user terpilih, tampilkan pencarian
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

    # Jika user sudah dipilih, tampilkan rekomendasi saja
    if 'selected_user' in st.session_state:
        selected_username = st.session_state['selected_user']
        st.header(f"Rekomendasi 👤 {selected_username}")

        user_id = username_to_userid.get(selected_username)
        user_ids = ratings['User_Id'].unique().tolist()
        place_ids = places['Place_Id'].unique().tolist()

        user2idx = {x: i for i, x in enumerate(user_ids)}
        place2idx = {x: i for i, x in enumerate(place_ids)}
        idx2place = {i: x for i, x in enumerate(place_ids)}

        ratings['user'] = ratings['User_Id'].map(user2idx)
        ratings['place'] = ratings['Place_Id'].map(place2idx)
        ratings['rating'] = ratings['Place_Ratings'].astype(np.float32)

        min_r, max_r = ratings['rating'].min(), ratings['rating'].max()
        x = ratings[['user', 'place']].values
        y = ratings['rating'].apply(lambda r: (r - min_r) / (max_r - min_r)).values

        split = int(0.8 * len(ratings))
        x_train, x_val = x[:split], x[split:]
        y_train, y_val = y[:split], y[split:]

        class RecommenderNet(tf.keras.Model):
            def __init__(self, n_users, n_places, embed_size=50):
                super().__init__()
                self.user_embed = layers.Embedding(n_users, embed_size)
                self.user_bias = layers.Embedding(n_users, 1)
                self.place_embed = layers.Embedding(n_places, embed_size)
                self.place_bias = layers.Embedding(n_places, 1)

            def call(self, inputs):
                u = self.user_embed(inputs[:, 0])
                p = self.place_embed(inputs[:, 1])
                dot = tf.reduce_sum(u * p, axis=1, keepdims=True)
                return tf.nn.sigmoid(dot + self.user_bias(inputs[:, 0]) + self.place_bias(inputs[:, 1]))

        model = RecommenderNet(len(user_ids), len(place_ids))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(x_train, y_train, batch_size=8, epochs=1, verbose=0)

        u_index = user2idx[user_id]
        rated_places = ratings[ratings['User_Id'] == user_id]['Place_Id'].tolist()
        not_rated = [pid for pid in place_ids if pid not in rated_places and pid in place2idx]
        test_input = np.array([[u_index, place2idx[pid]] for pid in not_rated])
        preds = model.predict(test_input).flatten()

        top_idx = preds.argsort()[-5:][::-1]
        top_place_ids = [idx2place[place2idx[not_rated[i]]] for i in top_idx]
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

        # Tombol reset (opsional)
        if st.button("🔄 Ganti Pengguna"):
            del st.session_state['selected_user']
            st.rerun()