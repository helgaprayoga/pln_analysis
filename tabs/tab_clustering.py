from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


CLUSTERING_FILE = Path(__file__).resolve().parent.parent / "data" / "clustering_result.csv"
EXTRA_STOPWORDS = {
    "pln", "listrik", "yg", "ga", "gak", "nya", "aja", "banget",
    "udah", "sih", "nih",
}


# Membaca dataframe hasil clustering dari file CSV.
@st.cache_data
def load_clustering_data():
    if not CLUSTERING_FILE.exists():
        return None
    return pd.read_csv(CLUSTERING_FILE)


# Menampilkan wordcloud dari teks pada setiap cluster.
# Mengambil stopword yang sama dengan proses di Google Colab.
def get_stopwords():
    try:
        words = set(stopwords.words("indonesian"))
        words.update(stopwords.words("english"))
    except LookupError:
        words = set()
    words.update(EXTRA_STOPWORDS)
    return list(words)


# Menghitung bobot kata TF-IDF berdasarkan label cluster dari CSV.
@st.cache_data(show_spinner=False)
def create_wordcloud_frequencies(dataframe):
    data = dataframe.copy()
    data["clean_text"] = data["clean_text"].fillna("").astype(str)
    data = data[data["clean_text"].str.strip() != ""]
    data = data.drop_duplicates(subset="clean_text").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words=get_stopwords(), min_df=2, max_df=0.9
    )
    matrix = vectorizer.fit_transform(data["clean_text"])
    words = vectorizer.get_feature_names_out()
    cluster_ids = sorted(data["cluster"].unique())
    frequencies = {}

    for cluster_id in cluster_ids:
        cluster_mask = (data["cluster"] == cluster_id).to_numpy()
        other_mask = (data["cluster"] != cluster_id).to_numpy()
        cluster_matrix = matrix[cluster_mask]
        cluster_center = cluster_matrix.mean(axis=0).A1
        other_centers = matrix[other_mask].mean(axis=0).A1
        scores = np.clip(cluster_center - other_centers, 0, None)
        frequencies[cluster_id] = {
            words[index]: float(scores[index])
            for index in np.nonzero(scores)[0]
            if scores[index] > 0
        }

    return frequencies


# Membuat koordinat PCA untuk memvisualisasikan posisi setiap cluster.
@st.cache_data(show_spinner=False)
def create_cluster_coordinates(dataframe):
    data = dataframe.copy()
    data["clean_text"] = data["clean_text"].fillna("").astype(str)
    data = data[data["clean_text"].str.strip() != ""]
    data = data.drop_duplicates(subset="clean_text").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words=get_stopwords(), min_df=2, max_df=0.9
    )
    matrix = vectorizer.fit_transform(data["clean_text"])

    if matrix.shape[1] >= 2:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(
            matrix.toarray()
        )
    else:
        values = matrix.toarray()
        first_component = values[:, 0] if values.shape[1] else np.zeros(len(data))
        coordinates = np.column_stack([first_component, np.zeros(len(data))])

    return coordinates, data["cluster"].to_numpy()


# Menampilkan sebaran postingan berdasarkan label cluster dari CSV.
def show_cluster_visualization(dataframe):
    coordinates, cluster_labels = create_cluster_coordinates(dataframe)
    cluster_codes = pd.Categorical(cluster_labels).codes

    figure, axis = plt.subplots(figsize=(9, 5))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cluster_codes,
        cmap="tab10",
        alpha=0.65,
    )
    axis.set_xlabel("Komponen PCA 1")
    axis.set_ylabel("Komponen PCA 2")
    axis.set_title("Sebaran Postingan Berdasarkan Cluster")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    st.pyplot(figure)
    plt.close(figure)


# Menampilkan wordcloud dari bobot TF-IDF setiap cluster.
def show_wordclouds(dataframe, cluster_values):
    columns = st.columns(len(cluster_values))
    frequencies = create_wordcloud_frequencies(dataframe)

    for cluster_id, column in zip(cluster_values, columns):
        with column:
            st.markdown(f"**Cluster {cluster_id}**")
            if not frequencies.get(cluster_id):
                st.info("Tidak ada kata dominan.")
                continue

            image = WordCloud(
                background_color="white",
                width=600,
                height=350,
                random_state=42,
            ).generate_from_frequencies(frequencies[cluster_id])
            figure, axis = plt.subplots(figsize=(5, 3))
            axis.imshow(image, interpolation="bilinear")
            axis.axis("off")
            st.pyplot(figure)
            plt.close(figure)


# Menampilkan seluruh hasil clustering dari file CSV.
def render_tab_clustering():
    st.header("Clustering Postingan")
    st.write("Hasil clustering dibaca dari `data/clustering_result.csv`.")

    dataframe = load_clustering_data()
    if dataframe is None:
        st.warning("File `data/clustering_result.csv` belum ditemukan.")
        return

    required_columns = {"clean_text", "cluster"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        st.error(
            "Kolom berikut belum ada di CSV: "
            + ", ".join(sorted(missing_columns))
        )
        return

    cluster_values = sorted(dataframe["cluster"].dropna().unique())

    col1, col2 = st.columns(2)
    col1.metric("Total data", f"{len(dataframe):,}")
    col2.metric("Jumlah cluster", len(cluster_values))

    # Menampilkan jumlah postingan pada setiap cluster.
    st.subheader("Ukuran Cluster")
    st.bar_chart(dataframe["cluster"].value_counts().sort_index())

    # Menampilkan visualisasi posisi postingan pada setiap cluster.
    st.subheader("Visualisasi Cluster")
    show_cluster_visualization(dataframe)

    # Menampilkan wordcloud berdasarkan label cluster dari CSV.
    st.subheader("Wordcloud Tiap Cluster")
    show_wordclouds(dataframe, cluster_values)

    # Menampilkan tema yang sudah dibuat di Google Colab.
    st.subheader("Tema dan Kata Dominan")
    theme_columns = st.columns(len(cluster_values))
    for cluster_id, column in zip(cluster_values, theme_columns):
        with column:
            cluster_data = dataframe[dataframe["cluster"] == cluster_id]
            st.markdown(f"**Cluster {cluster_id}**")
            st.caption(f"{len(cluster_data):,} postingan")

            if "nama_tema" in dataframe.columns:
                themes = cluster_data["nama_tema"].dropna().astype(str).unique()
                if len(themes) > 0:
                    st.write(themes[0])

    # Menampilkan beberapa contoh postingan dari setiap cluster.
    st.subheader("Contoh Postingan")
    text_column = "full_text" if "full_text" in dataframe.columns else "clean_text"
    for cluster_id in cluster_values:
        with st.expander(f"Cluster {cluster_id}"):
            examples = dataframe.loc[
                dataframe["cluster"] == cluster_id, text_column
            ].fillna("").head(3)
            for example in examples:
                st.write(f"- {str(example)[:300]}")

    # Menampilkan dataframe hasil clustering tanpa memproses ulang.
    st.subheader("Data Hasil Clustering")
    st.dataframe(dataframe, use_container_width=True)

    # Menjelaskan perbedaan tema dan kemungkinan tumpang tindih antarcluster.
    st.subheader("Penjelasan Singkat")
    st.markdown(
        "Setiap cluster dibedakan berdasarkan kata dan tema yang paling dominan "
        "pada postingan di dalamnya. Perbedaan tema dapat dilihat pada bagian "
        "Tema dan Kata Dominan.\n"
        "Kemungkinan masih ada pembahasan yang tumpang tindih. Beberapa postingan dapat menggunakan kata "
        "yang sama atau membahas isu PLN yang berdekatan, sehingga batas antarcluster "
        "tidak selalu sepenuhnya terpisah."
    )
