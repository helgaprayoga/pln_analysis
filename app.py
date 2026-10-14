import streamlit as st
import pandas as pd
from pathlib import Path

from tabs import tab_clustering, tab_data, tab_regresi, tab_sentimen


DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Dashboard Analisis PLN", layout="wide")
st.title("Dashboard Analisis PLN & Pemadaman")

@st.cache_data
def load_default_data():
    file_path = DATA_DIR / "merged_clean_sentimen.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

if 'df_utama' not in st.session_state:
    st.session_state['df_utama'] = load_default_data()

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Sentimen", "Clustering", "Regresi"])

with tab1:
    df_terbaru = tab_data.render_tab_data(st.session_state['df_utama'])
    if df_terbaru is not None:
         st.session_state['df_utama'] = df_terbaru

with tab2:
    tab_sentimen.render_tab_sentimen(st.session_state['df_utama'])

with tab3:
    tab_clustering.render_tab_clustering()

with tab4:
    tab_regresi.render_tab_regression()
