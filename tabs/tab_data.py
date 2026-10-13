import streamlit as st
import pandas as pd

def render_tab_data(df):
    st.header("Data Postingan X (Twitter)")
    st.write("Analisis sentimen percakapan layanan listrik PLN dan pemadaman.")
    
    # mengunggah csv baru
    uploaded_file = st.file_uploader("Unggah Dataset CSV Baru (Opsional)", type=["csv"])
    
    # jika user mengunggah file, akan menimpa dataframe bawaan
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df_utama'] = df 
        st.success("Dataset berhasil diunggah!")

    if df is not None:
        # tampikan pratinjau dataset
        st.subheader("Pratinjau Dataset")
        st.dataframe(df.head(10))
        
        # tampilkan jumlah postingan
        st.subheader("Informasi Data")
        st.write(f"**Total Postingan:** {len(df)} baris")
        
        # pemeriksaan kolom wajib
        st.subheader("Pemeriksaan Kolom Wajib")
        kolom_wajib = ['clean_text', 'created_at', 'like_count', 'sentimen_prediksi']
        kolom_tersedia = df.columns.tolist()
        
        for kolom in kolom_wajib:
            if kolom in kolom_tersedia:
                st.success(f" Kolom '{kolom}' tersedia.")
            else:
                st.error(f" Kolom '{kolom}' TIDAK tersedia!")
                
    return df