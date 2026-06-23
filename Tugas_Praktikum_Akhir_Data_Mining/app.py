import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi halaman utama web
st.set_page_config(page_title="Prediksi Kelayakan Pinjaman UMKM", layout="centered")

st.title("🏪 Aplikasi Prediksi Kelayakan Pinjaman Modal Usaha UMKM")
st.write("Proyek Akhir Praktikum Data Mining 2026")
st.write("---")

# Fungsi untuk memuat model Decision Tree yang sudah dilatih tadi
@st.cache_resource
def load_model():
    with open('model_pinjaman_umkm.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.sidebar.success("🧠 Otak Model ML Berhasil Dimuat!")

    st.subheader("📋 Silakan Isi Formulir Pengajuan Pinjaman")
    st.write("Masukkan data profil keuangan dan usaha UMKM Anda di bawah ini:")

    # 1. Membuat Form Input Interaktif untuk Pengguna Awam
    omset_bulanan = st.number_input("💵 Omset Bulanan Usaha (Rupiah):", min_value=0, value=15000000, step=1000000)
    jumlah_pinjaman = st.number_input("💰 Jumlah Pinjaman yang Diajukan (Rupiah):", min_value=0, value=50000000, step=1000000)
    lama_usaha = st.slider("📅 Lama Usaha Berjalan (Tahun):", min_value=0, max_value=20, value=3)
    skor_kredit = st.slider("📈 Skor Kredit (BI Checking / SLIK):", min_value=300, max_value=850, value=650, help="Rentang skor 300-850. Semakin tinggi semakin bersih riwayat kredit Anda.")
    rasio_utang = st.slider("📊 Rasio Pengeluaran Utang Bulanan:", min_value=0.0, max_value=1.0, value=0.3, step=0.05, help="Persentase omset yang habis untuk bayar cicilan utang lain saat ini.")

    agunan_pilihan = st.selectbox("🛡️ Apakah Memiliki Agunan/Jaminan (Sertifikat Rumah/Tanah/BPKB)?", ["Ada Jaminan", "Tidak Ada Jaminan"])
    status_agunan = 1 if agunan_pilihan == "Ada Jaminan" else 0

    st.write("---")

    # 2. Tombol Proses Prediksi
    if st.button("🚀 Cek Kelayakan Pinjaman Sekarang"):
        # Menyusun input pengguna menjadi format array yang dipahami model ML
        input_data = np.array([[omset_bulanan, jumlah_pinjaman, lama_usaha, skor_kredit, rasio_utang, status_agunan]])

        # Melakukan prediksi (0 = Ditolak, 1 = Layak)
        prediksi = model.predict(input_data)

        # Menampilkan Hasil Keputusan Instan
        st.subheader("📊 Hasil Analisis Keputusan Model AI:")
        if prediksi[0] == 1:
            st.success("🟢 **REKOMENDASI: PENGAJUAN PINJAMAN LAYAK / DITERIMA**")
            st.balloons() # Efek balon perayaan
            st.write("💡 *Catatan: Profil finansial dan rekam jejak kredit UMKM Anda dinilai sehat dan memenuhi standar kelayakan dana modal.*")
        else:
            st.error("🔴 **REKOMENDASI: PENGAJUAN PINJAMAN BELUM LAYAK / DITOLAK**")
            st.write("⚠️ *Saran Perbaikan: Coba turunkan nominal jumlah pinjaman yang diajukan, tingkatkan omset usaha bulanan, atau bersihkan riwayat tunggakan kredit Anda terlebih dahulu agar skor kredit naik.*")

except FileNotFoundError:
    st.error("File model_pinjaman_umkm.pkl tidak ditemukan di folder server!")
except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {str(e)}")
