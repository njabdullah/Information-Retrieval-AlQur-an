import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
import os

nltk_path = os.path.join(os.path.expanduser("~"), "nltk_data", "tokenizers", "punkt")

if not os.path.exists(nltk_path):
    nltk.download("punkt")

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# Load Data
df = pd.read_csv("Result/alquran_clustered.csv")

# Load Word2Vec Model and Embeddings
w2v_model = Word2Vec.load("Result/Embedding/w2v_model.model")
w2v_embeddings = np.load("Result/Embedding/w2v_embeddings.npy")

# Load BERT Embeddings and Vectorizer
bert_embeddings = np.load("Result/Embedding/bert_embeddings.npy")

# Load TF-IDF Embeddings and Vectorizers
tfidf_embeddings = joblib.load("Result/Embedding/tfidf_embeddings.pkl")
tfidf_vectorizer = joblib.load("Result/Embedding/tfidf_vectorizer.pkl")

# Load TF-IDF + LDA + PCA Embeddings and Vectorizers
tfidfldapca_embeddings = joblib.load("Result/Embedding/tfidfldapca_embeddings.pkl")
tfidfldapca_vectorizer = joblib.load("Result/Embedding/tfidfldapca_vectorizer.pkl")
lda_model = joblib.load("Result/Embedding/lda_model.pkl")
pca_model = joblib.load("Result/Embedding/pca_model.pkl")

# Load Clustering Models
kmeans_w2v = joblib.load("Result/Model/kmeans_w2v.pkl")
kmeans_bert = joblib.load("Result/Model/kmeans_bert.pkl")
kmeans_tfidf = joblib.load("Result/Model/kmeans_tfidf.pkl")
kmeans_tfidfldapca = joblib.load("Result/Model/kmeans_tfidfldapca.pkl")
ahc_w2v = joblib.load("Result/Model/ahc_w2v.pkl")
ahc_bert = joblib.load("Result/Model/ahc_bert.pkl")
ahc_tfidf = joblib.load("Result/Model/ahc_tfidf.pkl")
ahc_tfidfldapca = joblib.load("Result/Model/ahc_tfidfldapca.pkl")
dbscan_w2v = joblib.load("Result/Model/dbscan_w2v.pkl")
dbscan_bert = joblib.load("Result/Model/dbscan_bert.pkl")
dbscan_tfidf = joblib.load("Result/Model/dbscan_tfidf.pkl")
dbscan_tfidfldapca = joblib.load("Result/Model/dbscan_tfidfldapca.pkl")

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# Sidebar - Method Selection & Input
st.sidebar.title("🔍 Pengaturan Pencarian")

query = st.sidebar.text_input("Ketikkan topik atau kata kunci:")
top_n = st.sidebar.selectbox("Tampilkan berapa hasil teratas?", [5, 10, 20, "All Results in Selected Cluster"], index=0)

search_mode = st.sidebar.radio(
    "Pilih Mode Pencarian:",
    (
        "Keyword Based",
        # "Model Based (KMeans + W2V)",
        # "Model Based (KMeans + BERT)",        
        # "Model Based (KMeans + TF-IDF)",
        # "Model Based (KMeans + TF-IDF + LDA + PCA)",
        "Model Based (KMeans + Hybrid TF-IDF)",
        # "Model Based (AHC + W2V)",
        # "Model Based (AHC + BERT)",
        # "Model Based (AHC + TF-IDF)",
        # "Model Based (AHC + TF-IDF + LDA + PCA)",
        "Model Based (AHC + Hybrid TF-IDF)",
        # "Model Based (DBSCAN + W2V)",
        # "Model Based (DBSCAN + BERT)",
        # "Model Based (DBSCAN + TF-IDF)",
        # "Model Based (DBSCAN + TF-IDF + LDA + PCA)",
        "Model Based (DBSCAN + Hybrid TF-IDF)"
    )
)

# Main Title
st.markdown("<h1 style='color: #ffffff;'>📖 Al-Qur'an Information Retrieval</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# Preprocessing Function
def preprocess_query(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"[-]", ' ', text)
    text = re.sub(r"[^\w\s']", '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens_clean = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens_clean).strip()

# Search Function
def search_clustered_quran(query_vector, method_name, cluster_column, model, embeddings, raw_embeddings, query_vector_for_similarity=None, top_n=10):
    if hasattr(model, "predict"):
        query_cluster = model.predict([query_vector])[0]
    else:
        all_similarities = cosine_similarity([query_vector], embeddings)[0]
        most_similar_idx = np.argmax(all_similarities)
        query_cluster = df.loc[most_similar_idx, cluster_column]

    filtered_df = df[df[cluster_column] == query_cluster]
    filtered_embeddings = raw_embeddings[filtered_df.index]
    similarity_query_vector = query_vector_for_similarity if query_vector_for_similarity is not None else query_vector
    similarities = cosine_similarity([similarity_query_vector], filtered_embeddings)[0]

    # Tambahkan pengecekan untuk TF-IDF jika semua similarity = 0
    if method_name in ["Model Based (KMeans + TF-IDF)", "Model Based (AHC + TF-IDF)", "Model Based (DBSCAN + TF-IDF)",
                       "Model Based (KMeans + TF-IDF + LDA + PCA)", "Model Based (AHC + TF-IDF + LDA + PCA)", "Model Based (DBSCAN + TF-IDF + LDA + PCA)",
                       "Model Based (KMeans + Hybrid TF-IDF)", "Model Based (AHC + Hybrid TF-IDF)", "Model Based (DBSCAN + Hybrid TF-IDF)"]:
        if np.all(similarities == 0):
            return None

    filtered_df = filtered_df.copy()
    filtered_df['Similarity'] = similarities
    if top_n == "All Results in Selected Cluster":
        top_n = len(filtered_df)

    top_results = filtered_df.sort_values(by="Similarity", ascending=False).head(top_n)
    return top_results

# Display Verse Function
def display_verse(row):
    st.markdown(f"<div style='background-color:#808080; padding:1px; border-radius:10px;'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:#808080;'>{row['SurahNo']}:{row['AyahNo']}</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 25px; color: #ffffff; text-align: right;'>{row['ArabicText']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px; color: #ffffff;'>{row['IndonesianText']}</p>", unsafe_allow_html=True)
    st.markdown("</div><br>", unsafe_allow_html=True)

# Process Query
if query:
    st.subheader("Hasil Pencarian:")

    # Keyword Based
    if search_mode == "Keyword Based":
        results = df[df['IndonesianText'].str.contains(query, case=False, na=False)]
        if top_n == "All Results in Selected Cluster":
            top_n = 6236
        if not results.empty:
            for _, row in results.head(top_n).iterrows():
                display_verse(row)
        else:
            st.warning("Tidak ditemukan ayat yang cocok dengan kata kunci tersebut.")

    # Model Based Searches
    else:
        # Define combinations
        combinations = [
            ("Model Based (KMeans + W2V)", "Cluster_W2V_KMEANS", kmeans_w2v, w2v_embeddings, w2v_embeddings, None),
            ("Model Based (KMeans + BERT)", "Cluster_BERT_KMEANS", kmeans_bert, bert_embeddings, bert_embeddings, None),
            ("Model Based (KMeans + TF-IDF)", "Cluster_TFIDF_KMEANS", kmeans_tfidf, tfidf_embeddings, tfidf_embeddings, None),
            ("Model Based (KMeans + TF-IDF + LDA + PCA)", "Cluster_TFIDFLDAPCA_KMEANS", kmeans_tfidfldapca, tfidfldapca_embeddings, tfidfldapca_embeddings, None),
            ("Model Based (KMeans + Hybrid TF-IDF)", "Cluster_TFIDFLDAPCA_KMEANS", kmeans_tfidfldapca, tfidfldapca_embeddings, tfidf_embeddings, None),
            ("Model Based (AHC + W2V)", "Cluster_W2V_AHC", ahc_w2v, w2v_embeddings, w2v_embeddings, None),
            ("Model Based (AHC + BERT)", "Cluster_BERT_AHC", ahc_bert, bert_embeddings, bert_embeddings, None),
            ("Model Based (AHC + TF-IDF)", "Cluster_TFIDF_AHC", ahc_tfidf, tfidf_embeddings, tfidf_embeddings, None),
            ("Model Based (AHC + TF-IDF + LDA + PCA)", "Cluster_TFIDFLDAPCA_AHC", ahc_tfidfldapca, tfidfldapca_embeddings, tfidfldapca_embeddings, None),
            ("Model Based (AHC + Hybrid TF-IDF)", "Cluster_TFIDFLDAPCA_AHC", ahc_tfidfldapca, tfidfldapca_embeddings, tfidf_embeddings, None),
            ("Model Based (DBSCAN + W2V)", "Cluster_W2V_DBSCAN", dbscan_w2v, w2v_embeddings, w2v_embeddings, None),
            ("Model Based (DBSCAN + BERT)", "Cluster_BERT_DBSCAN", dbscan_bert, bert_embeddings, bert_embeddings, None),
            ("Model Based (DBSCAN + TF-IDF)", "Cluster_TFIDF_DBSCAN", dbscan_tfidf, tfidf_embeddings, tfidf_embeddings, None),
            ("Model Based (DBSCAN + TF-IDF + LDA + PCA)", "Cluster_TFIDFLDAPCA_DBSCAN", dbscan_tfidfldapca, tfidfldapca_embeddings, tfidfldapca_embeddings, None),
            ("Model Based (DBSCAN + Hybrid TF-IDF)", "Cluster_TFIDFLDAPCA_DBSCAN", dbscan_tfidfldapca, tfidfldapca_embeddings, tfidf_embeddings, None),
        ]

        # Generate Query Vectors W2V
        preprocessed_query = preprocess_query(query)
        query_tokens = [word for word in preprocessed_query.split() if word in w2v_model.wv]
        w2v_query_vector = np.mean([w2v_model.wv[word] for word in query_tokens], axis=0) if query_tokens else np.zeros(200)

        # Generate Query Vectors TF-IDF
        tfidf_query_vectors = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]
        tfidf_query_vector = tfidfldapca_vectorizer.transform([preprocessed_query]).toarray()[0]
        lda_query_vector = lda_model.transform(tfidf_query_vector.reshape(1, -1))
        combined_query_vector = np.hstack([tfidf_query_vector.reshape(1, -1), lda_query_vector])
        query_vector_for_cluster = pca_model.transform(combined_query_vector)[0]

        # Select the appropriate combination
        for method_name, cluster_column, model, embeddings, raw_embeddings, _ in combinations:
            if search_mode == method_name:
                query_vector = {
                    "Model Based (KMeans + W2V)": w2v_query_vector,
                    "Model Based (KMeans + BERT)": bert_embeddings.mean(axis=0),
                    "Model Based (KMeans + TF-IDF)": tfidf_query_vectors,
                    "Model Based (KMeans + TF-IDF + LDA + PCA)": query_vector_for_cluster,
                    "Model Based (KMeans + Hybrid TF-IDF)": query_vector_for_cluster,
                    "Model Based (AHC + W2V)": w2v_query_vector,
                    "Model Based (AHC + BERT)": bert_embeddings.mean(axis=0),
                    "Model Based (AHC + TF-IDF)": tfidf_query_vectors,
                    "Model Based (AHC + TF-IDF + LDA + PCA)": query_vector_for_cluster,
                    "Model Based (AHC + Hybrid TF-IDF)": query_vector_for_cluster,
                    "Model Based (DBSCAN + W2V)": w2v_query_vector,
                    "Model Based (DBSCAN + BERT)": bert_embeddings.mean(axis=0),
                    "Model Based (DBSCAN + TF-IDF)": tfidf_query_vectors,
                    "Model Based (DBSCAN + TF-IDF + LDA + PCA)": query_vector_for_cluster,
                    "Model Based (DBSCAN + Hybrid TF-IDF)": query_vector_for_cluster,
                }[method_name]

                query_vector_for_similarity = {
                    "Model Based (KMeans + Hybrid TF-IDF)": tfidf_query_vectors,
                    "Model Based (AHC + Hybrid TF-IDF)": tfidf_query_vectors,
                    "Model Based (DBSCAN + Hybrid TF-IDF)": tfidf_query_vectors,
                }.get(method_name)

                if method_name in ["Model Based (KMeans + W2V)", "Model Based (AHC + W2V)", "Model Based (DBSCAN + W2V)"] and not query_tokens:
                    st.warning("❌ Kata dalam query tidak dikenali oleh model Word2Vec. Gunakan kata yang muncul di Al-Qur'an.")
                    break

                try:
                    results = search_clustered_quran(
                        query_vector=query_vector,
                        method_name=method_name,
                        cluster_column=cluster_column,
                        model=model,
                        embeddings=embeddings,
                        raw_embeddings=raw_embeddings,
                        query_vector_for_similarity=query_vector_for_similarity,
                        top_n=top_n
                    )

                    if results is None or results.empty:
                        st.warning("⚠️ Tidak ditemukan ayat yang relevan dengan query ini.")
                    else:
                        for _, row in results.iterrows():
                            display_verse(row)
                except Exception as e:
                    st.error(f"❌ Error in {method_name}: {e}")
                break