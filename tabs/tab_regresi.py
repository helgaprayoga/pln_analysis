from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "regression_model.joblib"
PREDICTION_FILE = DATA_DIR / "regression_predictions.csv"


# Membaca model regresi dan hasil prediksi dari folder data.
@st.cache_resource
def load_regression_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


# Membaca data pengujian yang sudah diprediksi dari Google Colab.
@st.cache_data
def load_prediction_data():
    if not PREDICTION_FILE.exists():
        return None
    return pd.read_csv(PREDICTION_FILE)


# Menampilkan hasil evaluasi model regresi.
def show_regression_metrics(prediction_data):
    actual = prediction_data["actual_likes"]
    predicted = prediction_data["predicted_likes"]
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Data pengujian", f"{len(prediction_data):,}")
    metric_col2.metric("MAE", f"{mae:.2f}")
    metric_col3.metric("R²", f"{r2:.4f}")

    if r2 >= 0:
        st.info(
            f"Model menjelaskan sekitar {r2 * 100:.2f}% variasi likes "
            "pada data pengujian."
        )


# Menampilkan grafik perbandingan likes aktual dan prediksi.
def show_prediction_chart(prediction_data):
    actual = prediction_data["actual_likes"]
    predicted = prediction_data["predicted_likes"]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(actual, predicted, alpha=0.6, label="Hasil prediksi")

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Prediksi ideal",
    )
    axis.set_xlabel("Likes Aktual")
    axis.set_ylabel("Likes Prediksi")
    axis.set_title("Perbandingan Likes Aktual dan Prediksi")
    axis.legend()
    axis.grid(True, alpha=0.3)
    st.pyplot(figure)
    plt.close(figure)


# Menampilkan form prediksi likes menggunakan model yang sudah disimpan.
def show_prediction_form(model):
    word_count = st.number_input("Jumlah kata", min_value=0, value=20, step=1)
    hashtag_count = st.number_input("Jumlah hashtag", min_value=0, value=2, step=1)

    if st.button("Prediksi Likes"):
        new_data = pd.DataFrame({
            "jumlah_kata": [word_count],
            "jumlah_hashtag": [hashtag_count],
        })
        prediction = float(model.predict(new_data)[0])
        st.success(f"Estimasi likes: {max(0, prediction):.2f}")
        st.caption("Hasil ini merupakan estimasi model, bukan jumlah likes pasti.")


# Menampilkan seluruh isi tab regresi tanpa melatih model ulang.
def render_tab_regression():
    st.header("Regresi Likes")
    st.write(
        "Visualisasi menggunakan model dan hasil prediksi yang sudah dibuat "
        "di Google Colab."
    )

    model = load_regression_model()
    prediction_data = load_prediction_data()

    if model is None:
        st.warning("File `models/regression_model.joblib` belum ditemukan.")
        return

    if prediction_data is None:
        st.warning("File `data/regression_predictions.csv` belum ditemukan.")
        return

    required_columns = {
        "jumlah_kata",
        "jumlah_hashtag",
        "actual_likes",
        "predicted_likes",
    }
    missing_columns = required_columns.difference(prediction_data.columns)
    if missing_columns:
        st.error(
            "Kolom berikut belum ada di CSV: "
            + ", ".join(sorted(missing_columns))
        )
        return

    # Menampilkan nilai evaluasi dari hasil prediksi yang sudah tersimpan.
    st.subheader("Evaluasi Model")
    show_regression_metrics(prediction_data)

    # Menampilkan koefisien model regresi yang sudah disimpan.
    st.subheader("Parameter Model")
    parameter_col1, parameter_col2, parameter_col3 = st.columns(3)
    parameter_col1.metric("Intercept", f"{model.intercept_:.4f}")
    parameter_col2.metric("Koefisien jumlah kata", f"{model.coef_[0]:.4f}")
    parameter_col3.metric("Koefisien hashtag", f"{model.coef_[1]:.4f}")

    # Menampilkan grafik hasil prediksi pada data pengujian.
    st.subheader("Grafik Likes Aktual dan Prediksi")
    show_prediction_chart(prediction_data)

    # Menampilkan beberapa baris hasil prediksi.
    st.subheader("Contoh Hasil Prediksi")
    st.dataframe(prediction_data.head(10), use_container_width=True)

    # Menyediakan form prediksi likes untuk data baru.
    st.subheader("Prediksi Likes Baru")
    show_prediction_form(model)

    st.caption(
        "Model hanya menggunakan jumlah kata dan jumlah hashtag. "
        "Faktor lain seperti usia postingan dan jangkauan akun tidak digunakan."
    )
