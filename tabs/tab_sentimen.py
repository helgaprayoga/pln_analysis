import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def render_tab_sentimen(df):
    st.header("Hasil Analisis Sentimen")
    
    if df is None or 'sentimen_prediksi' not in df.columns:
        st.warning("Data belum tersedia atau kolom 'sentimen_prediksi' tidak ditemukan.")
        return

    #jumlah dan persentase sentimen
    st.subheader("Distribusi Sentimen")
    jumlah_sentimen = df['sentimen_prediksi'].value_counts()
    persen_sentimen = df['sentimen_prediksi'].value_counts(normalize=True) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Positif", f"{jumlah_sentimen.get('positif', 0)}", f"{persen_sentimen.get('positif', 0):.2f}%")
    col2.metric("Netral", f"{jumlah_sentimen.get('netral', 0)}", f"{persen_sentimen.get('netral', 0):.2f}%", delta_color="off")
    col3.metric("Negatif", f"{jumlah_sentimen.get('negatif', 0)}", f"{persen_sentimen.get('negatif', 0):.2f}%", delta_color="inverse")
    
    #grafik distribusi sentimen
    st.subheader("Grafik Distribusi Sentimen")
    fig, ax = plt.subplots(figsize=(8, 5))
    warna = ['#ff9999', '#66b3ff', '#99ff99']
    
    kategori_urut = ['negatif', 'netral', 'positif']
    jumlah_urut = [jumlah_sentimen.get(k, 0) for k in kategori_urut]
    
    ax.bar(kategori_urut, jumlah_urut, color=warna, edgecolor='black')
    ax.set_title('Grafik Distribusi Sentimen Postingan PLN', fontsize=12)
    ax.set_ylabel('Jumlah Postingan')
    ax.set_xlabel('Kategori Sentimen')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    #tabel hasil prediksi 
    st.subheader("Tabel Hasil Klasifikasi")
    if 'clean_text' in df.columns:
        st.dataframe(df[['clean_text', 'sentimen_prediksi']].head(50), use_container_width=True)
        
    #ringkasan pemeriksaan manual
    st.subheader("Ringkasan Pemeriksaan Manual")
    if 'sentimen_asli' in df.columns:
        df_evaluasi = df.dropna(subset=['sentimen_asli'])
        if not df_evaluasi.empty:
            df_evaluasi['sesuai'] = df_evaluasi['sentimen_asli'].str.lower() == df_evaluasi['sentimen_prediksi']
            jumlah_sesuai = df_evaluasi['sesuai'].sum()
            persen_sesuai = (jumlah_sesuai / len(df_evaluasi)) * 100
            
            st.info(f"**Tingkat Kesesuaian:** {jumlah_sesuai} dari {len(df_evaluasi)} sampel ({persen_sesuai:.2f}%)")
            
            tidak_sesuai = df_evaluasi[~df_evaluasi['sesuai']]
            if not tidak_sesuai.empty:
                st.write("**Contoh Analisis Ketidaksesuaian:**")
                st.dataframe(tidak_sesuai[['clean_text', 'sentimen_asli', 'sentimen_prediksi']].head(3), use_container_width=True)
                st.caption("*Catatan: Model AI mungkin menangkap konteks kalimat secara keseluruhan yang berbeda dari perspektif manusia, atau gagal memahami ironi/sarkasme khusus.*")
        else:
            st.write("Kolom 'sentimen_asli' ditemukan, tetapi tidak ada baris yang diisi.")
    else:
        st.write("Kolom 'sentimen_asli' belum dibuat di file CSV untuk dilakukan evaluasi.")