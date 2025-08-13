import pandas as pd
import streamlit as st

def home():
    # Judul
    st.title("Deteksi Transaksi Fraud pada Sistem Pembayaran Digital")
    st.markdown("---")

    # Latar belakang
    st.markdown("## Latar Belakang")
    st.markdown("""
Fraud pada transaksi digital memiliki **prevalensi sangat rendah** (highly imbalanced) namun berisiko tinggi.
Deteksi dini diperlukan agar tim risk dapat **meminimalkan kerugian** tanpa mengganggu pengalaman pengguna.
Pada proyek ini dikembangkan **model klasifikasi** berbasis *machine learning* untuk mengidentifikasi transaksi berisiko.
""")

    # Rumusan masalah
    st.markdown("## Rumusan Masalah")
    st.markdown("""
Membangun model untuk **memprediksi transaksi fraud** dengan fokus pada **recall** kelas fraud (meminimalkan *false negative*).
Model harus siap **inference** pada data baru dan dapat di-*deploy* melalui antarmuka Streamlit.
""")

    st.markdown("---")

    # Dataset
    st.markdown("## Dataset")
    st.caption("Sumber: *Payment Fraud — Empowering Financial Security* (Kaggle). Target: `label` (0=legitimate, 1=fraud).")
    try:
        df = pd.read_csv("payment_fraud.csv")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")
    except Exception:
        st.info("File `payment_fraud.csv` belum tersedia di root repo. Letakkan file tersebut untuk preview dataset.")

    st.markdown("---")

    # Data Overview (kolom utama yang dipakai model)
    st.markdown("## Data Overview", unsafe_allow_html=True)
    st.markdown("""
<table>
<thead>
<tr><th>Kolom</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td><code>paymentMethod</code></td><td>Metode pembayaran: <code>creditcard</code>, <code>storecredit</code>, <code>paypal</code>.</td></tr>
<tr><td><code>Category</code></td><td>Kategori transaksi: <code>shopping</code>, <code>electronics</code>, <code>food</code>.</td></tr>
<tr><td><code>numItems</code></td><td>Jumlah item pada transaksi (numerik).</td></tr>
<tr><td><code>localTime</code></td><td>Waktu lokal yang sudah dinormalisasi (≈ 4.70-5.05 pada data ini).</td></tr>
<tr><td><code>hour</code></td><td>Jam transaksi (0-23).</td></tr>
<tr><td><code>risk_score</code></td><td>Skor komposit dari indikator risiko (hasil rekayasa fitur).</td></tr>
<tr><td><code>transaction_velocity</code></td><td>Indikator “kecepatan”/intensitas transaksi.</td></tr>
<tr><td><code>payment_age_ratio</code></td><td>Rasio umur metode pembayaran terhadap umur akun.</td></tr>
<tr><td><code>category_prob</code></td><td>Probabilitas kategori dari hasil EDA (mapping per kategori).</td></tr>
<tr><td><code>category_deviation</code></td><td><code>1 - category_prob</code>, deviasi dari perilaku umum kategori.</td></tr>
<tr><td><code>isHighRiskPayment</code></td><td>Indikator metode berisiko (mis. <code>paypal</code> → 1).</td></tr>
<tr><td><code>isNight</code></td><td>Indikator malam hari (jam ≥21 atau &lt;6).</td></tr>
<tr><td><code>time_bin</code></td><td>Versi ordinal dari <code>hour</code> untuk pemodelan.</td></tr>
<tr><td><code>temporal_risk_window</code></td><td>Indikator apakah berada pada time window at risk.</td></tr>
<tr><td><code>label</code></td><td>Target: 0 = legitimate, 1 = fraud.</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Model yang digunakan
    st.markdown("## Algoritma & Evaluasi")
    st.markdown("""
Model terbaik: **XGBoost** dalam **Pipeline** (preprocessing + model) dengan **threshold tuning = 0.10** untuk
meningkatkan *recall* pada data **imbalanced**. Metrik utama **ROC-AUC ~0.86** (test) dan *recall* kelas fraud ditingkatkan
dengan konsekuensi penurunan *precision* (dikompensasi via triage dan verifikasi bertahap).
""")

    st.markdown("---")


if __name__ == "__main__":
    home()