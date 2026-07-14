import streamlit as st
import joblib
import pandas as pd
import re

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sentimen MBG", page_icon="🍽️", layout="centered")

st.title("🍽️ Analisis Sentimen Program MBG")
st.write("Aplikasi ini memprediksi sentimen opini publik di media sosial X mengenai Program Makan Bergizi Gratis (MBG) menggunakan algoritma Multinomial Naive Bayes.")

# ==========================================
# 2. LOAD MODEL & KAMUS
# ==========================================
@st.cache_resource
def load_models():
    nb_model = joblib.load('model_naive_bayes_final.pkl')
    tfidf = joblib.load('tfidf_vectorizer_final.pkl')
    return nb_model, tfidf

model, tfidf_vectorizer = load_models()


# ==========================================
# 4. ANTARMUKA INPUT PENGGUNA
# ==========================================
st.subheader("Coba Prediksi Cuitan Baru")
input_teks = st.text_area("Masukkan opini Anda di sini:", height=100, placeholder="Contoh: Saya sangat mendukung program makan siang gratis ini karena membantu anak-anak.")

if st.button("Prediksi Sentimen", type="primary"):
    if input_teks.strip() == "":
        st.warning("⚠️ Silakan masukkan teks cuitan terlebih dahulu!")
    else:
        # Proses Pembersihan Teks seperti di Bab 3
        teks_bersih = input_teks.lower()  # Contoh pembersihan sederhana
        st.caption(f"*(Teks setelah dibersihkan mesin: {teks_bersih})*") # Menampilkan teks bersih agar bisa dianalisis
        
        # Prediksi
        teks_vektor = tfidf_vectorizer.transform([teks_bersih])
        hasil_prediksi = model.predict(teks_vektor)[0]
        
        # Output
        st.markdown("---")
        st.write("**Hasil Prediksi Mesin:**")
        
        if hasil_prediksi == 'Positif':
            st.success("🟢 SENTIMEN POSITIF")
        elif hasil_prediksi == 'Negatif':
            st.error("🔴 SENTIMEN NEGATIF")
        else:
            st.info("⚪ SENTIMEN NETRAL")