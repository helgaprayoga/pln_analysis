import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud


CLUSTER_COUNT = 3
EXTRA_STOPWORDS = {
    "pln", "listrik", "yg", "ga", "gak", "nya", "aja", "banget",
    "udah", "sih", "nih",
}


# Mengambil stopword bahasa Indonesia, Inggris, dan kata umum pada dataset.
def _get_stopwords():
    try:
        words = set(stopwords.words("indonesian"))
        words.update(stopwords.words("english"))
    except LookupError:
        # Stopword NLTK belum tersedia, tetapi clustering tetap dapat berjalan.
        words = set()
    words.update(EXTRA_STOPWORDS)
    return list(words)


# Menjalankan TF-IDF, K-Means, pengukuran skor, dan persiapan data visualisasi.
@st.cache_data(show_spinner=False)
def _run_clustering(dataframe):
    data = dataframe.copy()
    data["clean_text"] = data["clean_text"].fillna("").astype(str)
    data = data[data["clean_text"].str.strip() != ""]
    data = data.drop_duplicates(subset="clean_text").reset_index(drop=True)

    if len(data) < CLUSTER_COUNT:
        raise ValueError(
            f"Data bersih hanya {len(data)} baris, sedangkan jumlah cluster "
            f"yang digunakan adalah {CLUSTER_COUNT}."
        )

    vectorizer = TfidfVectorizer(
        stop_words=_get_stopwords(), min_df=2, max_df=0.9
    )
    matrix = vectorizer.fit_transform(data["clean_text"])
    if matrix.shape[1] == 0:
        raise ValueError("Tidak ada kosakata yang dapat dipakai setelah preprocessing.")

    model = KMeans(n_clusters=CLUSTER_COUNT, n_init=10, random_state=42)
    data["cluster"] = model.fit_predict(matrix)

    words = vectorizer.get_feature_names_out()
    word_order = model.cluster_centers_.argsort()[:, ::-1]
    summaries = []
    wordclouds = []

    # Menyiapkan kata dominan dan bobot wordcloud untuk setiap cluster.
    for cluster_id in range(CLUSTER_COUNT):
        cluster_data = data[data["cluster"] == cluster_id]
        dominant_words = [words[i] for i in word_order[cluster_id, :5]]

        other_centers = model.cluster_centers_[
            [i for i in range(CLUSTER_COUNT) if i != cluster_id]
        ].mean(axis=0)
        word_scores = (model.cluster_centers_[cluster_id] - other_centers).clip(min=0)
        word_frequencies = {
            words[i]: float(word_scores[i])
            for i in word_scores.nonzero()[0]
            if word_scores[i] > 0
        }
        wordclouds.append(word_frequencies)

        example_column = "full_text" if "full_text" in data.columns else "clean_text"
        examples = cluster_data[example_column].fillna("").head(3).tolist()
        summaries.append({
            "cluster": cluster_id,
            "size": len(cluster_data),
            "dominant_words": dominant_words,
            "theme": "Tema: " + ", ".join(dominant_words[:3]),
            "examples": examples,
        })

    silhouette = silhouette_score(matrix, data["cluster"])
    if matrix.shape[1] >= 2:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(matrix.toarray())
    else:
        coordinates = matrix.toarray()
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
    return data, summaries, wordclouds, coordinates, silhouette


# Menampilkan seluruh hasil clustering dalam bentuk ringkasan dan visualisasi.
def render_tab_clustering(dataframe):
    st.header("Clustering Postingan")
    st.write("Mengelompokkan postingan berdasarkan kemiripan kata pada kolom `clean_text`.")

    if dataframe is None or "clean_text" not in dataframe.columns:
        st.warning("Data belum tersedia atau kolom 'clean_text' tidak ditemukan.")
        return

    try:
        result, summaries, wordclouds, coordinates, silhouette = _run_clustering(dataframe)
    except ValueError as error:
        st.error(str(error))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Data awal", f"{len(dataframe):,}")
    col2.metric("Data dianalisis", f"{len(result):,}")
    col3.metric("Silhouette score", f"{silhouette:.3f}")

    # Menampilkan jumlah data pada masing-masing cluster.
    st.subheader("Ukuran Cluster")
    cluster_sizes = result["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_sizes)

    # Menampilkan posisi data berdasarkan dua komponen PCA.
    st.subheader("Sebaran Cluster")
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=result["cluster"], cmap="tab10", alpha=0.65
    )
    ax.set_xlabel("Komponen 1")
    ax.set_ylabel("Komponen 2")
    ax.set_title("Visualisasi PCA")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Wordcloud Tiap Cluster")
    columns = st.columns(CLUSTER_COUNT)
    for info, frequencies, column in zip(summaries, wordclouds, columns):
        with column:
            st.markdown(f"**Cluster {info['cluster']}**")
            if frequencies:
                wordcloud_image = WordCloud(
                    background_color="white", width=600, height=350
                ).generate_from_frequencies(frequencies)
                fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
                ax_wc.imshow(wordcloud_image, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            else:
                st.info("Belum ada kata dominan.")

    # Menjelaskan tema dan kata dominan setelah wordcloud ditampilkan.
    st.subheader("Tema dan Kata Dominan")
    theme_columns = st.columns(CLUSTER_COUNT)
    for info, column in zip(summaries, theme_columns):
        with column:
            st.markdown(f"**Cluster {info['cluster']}**")
            st.caption(f"{info['size']:,} postingan")
            st.write(info["theme"])
            st.write(", ".join(info["dominant_words"]))

    # Menampilkan beberapa contoh postingan agar hasil cluster mudah dipahami.
    st.subheader("Contoh Postingan")
    for info in summaries:
        with st.expander(f"Cluster {info['cluster']} - {info['theme']}"):
            for example in info["examples"]:
                st.write(f"- {example[:300]}")

    # Menampilkan semua baris hasil clustering.
    st.subheader("Data Hasil Clustering")
    display_columns = [column for column in ["full_text", "clean_text", "cluster"] if column in result.columns]
    st.dataframe(result[display_columns], use_container_width=True)
