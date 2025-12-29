import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Rekomendasi Jurusan",
    page_icon="🎓"
)

# --- 2. LOAD MODEL YANG SUDAH DILATIH ---
# Kita pakai @st.cache agar model tidak diload berulang-ulang tiap klik tombol
@st.cache_resource
def load_models():
    model = joblib.load('model_kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Panggil fungsi load
try:
    kmeans, scaler = load_models()
except:
    st.error("❌ File model tidak ditemukan! Pastikan 'model_kmeans.pkl' dan 'scaler.pkl' ada di folder yang sama.")
    st.stop()

# --- 3. LOGIKA REKOMENDASI (KAMUS) ---
# Sesuaikan teks ini dengan hasil analisis Jupyter Notebook kamu
rekomendasi_dict = {
    0: "📚 Jurusan BAHASA & SENI\n(Sastra Inggris, DKV, Seni Musik, Hubungan Internasional)",
    1: "💰 Jurusan SOSIAL & EKONOMI\n(Manajemen, Akuntansi, Hukum, Administrasi Bisnis)",
    2: "🔬 Jurusan SAINS & TEKNIK\n(Teknik Informatika, Kedokteran, Sipil, Arsitektur)"
}

# --- 4. TAMPILAN WEBSITE (UI) ---
st.title("🎓 Rekomendasi Jurusan Kuliah")
st.write("Sistem cerdas berbasis AI untuk menentukan jurusan kuliah berdasarkan nilai rapor siswa.")
st.markdown("---")

# Membuat Kolom Isian (Form)
st.sidebar.header("Input Nilai Siswa")
st.sidebar.write("Masukkan nilai skala 0 - 100")

# Menggunakan Slider agar lebih keren (atau bisa ganti st.number_input)
mtk = st.sidebar.number_input("Nilai Matematika", min_value=0, max_value=100, value=70)
ing = st.sidebar.number_input("Nilai B. Inggris", min_value=0, max_value=100, value=70)
ipa = st.sidebar.number_input("Nilai IPA", min_value=0, max_value=100, value=70)
ips = st.sidebar.number_input("Nilai IPS", min_value=0, max_value=100, value=70)
seni = st.sidebar.number_input("Nilai Seni", min_value=0, max_value=100, value=70)

# Tombol Prediksi
if st.sidebar.button("🔍 Cek Rekomendasi"):
    # --- 5. PROSES PREDIKSI ---
    
    # Masukkan data ke DataFrame (Harus sama persis kolomnya dengan waktu training)
    data_input = pd.DataFrame([[mtk, ing, ipa, ips, seni]], 
                              columns=['Matematika', 'B_Inggris', 'IPA', 'IPS', 'Seni'])
    
    # Scaling data
    data_scaled = scaler.transform(data_input)
    
    # Prediksi
    cluster_hasil = kmeans.predict(data_scaled)[0]
    
    # --- 6. TAMPILKAN HASIL ---
    st.success("✅ Analisis Selesai!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediksi Cluster", f"Cluster {cluster_hasil}")
    
    with col2:
        st.info("Berdasarkan pola nilaimu, kamu paling cocok masuk ke:")
        
    st.header(rekomendasi_dict[cluster_hasil])
    
    # Tambahan visualisasi sederhana
    st.write("### 📊 Grafik Nilai Kamu:")
    chart_data = pd.DataFrame({
        'Mata Pelajaran': ['MTK', 'Inggris', 'IPA', 'IPS', 'Seni'],
        'Nilai': [mtk, ing, ipa, ips, seni]
    })
    st.bar_chart(chart_data.set_index('Mata Pelajaran'))

else:
    st.info("👈 Masukkan nilai di menu sebelah kiri, lalu tekan tombol 'Cek Rekomendasi'")