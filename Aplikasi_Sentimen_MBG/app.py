import streamlit as st
import joblib
import pandas as pd
import re

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib paling atas)
# ==========================================
st.set_page_config(page_title="Analisis Sentimen MBG", page_icon="🍽️", layout="centered")

# ==========================================
# 2. LOAD MODEL & KAMUS (Di-cache agar cepat)
# ==========================================
@st.cache_resource
def load_models():
    # Pastikan file .pkl berada di direktori yang sama dengan app.py di GitHub
    nb_model = joblib.load('model_naive_bayes_final.pkl')
    tfidf = joblib.load('tfidf_vectorizer_final.pkl')
    return nb_model, tfidf

# ==========================================
# 3. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("📌 Menu Navigasi")
pilihan_halaman = st.sidebar.radio(
    "Silakan pilih halaman:",
    ("Pengenalan MBG", "Analisis Sentimen")
)

# ==========================================
# 4. HALAMAN 1: PENGENALAN MBG
# ==========================================
if pilihan_halaman == "Pengenalan MBG":
    st.title("🍽️ Pengenalan Program MBG")
    
    # Menampilkan Gambar
    try:
        st.image("gambar_mbg.jpg", caption="Ilustrasi Program Makan Bergizi Gratis (MBG)", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ File gambar 'gambar_mbg.jpg' belum ditemukan. Silakan upload gambar ke repositori GitHub Anda.")

    # Bagian Pengenalan
    st.info("""
    **Tentang Program Makan Bergizi Gratis (MBG)**  
    Program Makan Bergizi Gratis (MBG) merupakan inisiatif pemerintah yang bertujuan untuk menyediakan makanan sehat dan bernutrisi bagi anak-anak sekolah dan ibu hamil. Program ini diharapkan dapat menekan angka stunting, meningkatkan fokus belajar siswa, serta menggerakkan ekonomi lokal melalui pelibatan UMKM penyedia bahan makanan.
    """)

    st.markdown("""
    Aplikasi ini dirancang untuk **menganalisis dan memprediksi sentimen opini publik** di media sosial X (Twitter) mengenai Program MBG. Prediksi dilakukan secara otomatis menggunakan model *Machine Learning* dengan algoritma **Multinomial Naive Bayes**.
    """)

# ==========================================
# 5. HALAMAN 2: ANALISIS SENTIMEN
# ==========================================
elif pilihan_halaman == "Analisis Sentimen":
    st.title("📊 Uji Analisis Sentimen")
    st.markdown("Silakan masukkan opini atau teks cuitan untuk mengetahui apakah sentimen publik tersebut bernada Positif, Negatif, atau Netral.")
    st.markdown("---")
    
    # Menangani kemungkinan error saat load model
    try:
        model, tfidf_vectorizer = load_models()
        model_loaded = True
    except FileNotFoundError:
        st.error("⚠️ File model ('model_naive_bayes_final.pkl' atau 'tfidf_vectorizer_final.pkl') tidak ditemukan di server. Pastikan file tersebut sudah di-push ke GitHub.")
        model_loaded = False

    st.subheader("Coba Prediksi Opini Baru")
    input_teks = st.text_area(
        "Masukkan opini atau cuitan Anda di sini:", 
        height=100, 
        placeholder="Contoh: Saya sangat mendukung program makan siang gratis ini karena sangat membantu gizi anak-anak di daerah."
    )

    # Tombol prediksi
    if st.button("Prediksi Sentimen", type="primary", disabled=not model_loaded):
        if input_teks.strip() == "":
            st.warning("⚠️ Silakan masukkan teks cuitan terlebih dahulu!")
        else:
            # Proses Pembersihan Teks (Sederhana)
            teks_bersih = input_teks.lower()
            st.caption(f"*(Teks setelah preprocessing: {teks_bersih})*") 
            
            with st.spinner('Sedang menganalisis sentimen menggunakan Naive Bayes...'):
                # Prediksi
                teks_vektor = tfidf_vectorizer.transform([teks_bersih])
                hasil_prediksi = model.predict(teks_vektor)[0]
                
                # Output Hasil
                st.markdown("### Hasil Analisis Mesin:")
                
                if hasil_prediksi == 'Positif':
                    st.success("🟢 **SENTIMEN POSITIF**")
                elif hasil_prediksi == 'Negatif':
                    st.error("🔴 **SENTIMEN NEGATIF**")
                else:
                    st.info("⚪ **SENTIMEN NETRAL**")