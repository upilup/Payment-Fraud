import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# -------- helpers --------
def _load_dataset(show_uploader: bool = False):
    """
    Load dataset tanpa menampilkan uploader secara default.
    Jika show_uploader=False, akan mencoba membaca 'payment_fraud.csv'.
    Bila file tidak ada, hanya menampilkan info (tanpa uploader).
    """
    df = None

    if not show_uploader:
        # baca dari file lokal saja
        try:
            df = pd.read_csv("payment_fraud.csv")
            # normalisasi ringan (sesuai sebelumnya)
            if "paymentMethod" in df.columns:
                df["paymentMethod"] = df["paymentMethod"].astype(str).str.strip().str.lower()
            if "Category" in df.columns:
                df["Category"] = df["Category"].astype(str).str.strip().str.lower()
            if "hour" in df.columns and "isNight" not in df.columns:
                df["isNight"] = ((df["hour"] >= 21) | (df["hour"] < 6)).astype(int)
            if "paymentMethod" in df.columns and "isHighRiskPayment" not in df.columns:
                df["isHighRiskPayment"] = (df["paymentMethod"] == "paypal").astype(int)
            if "hour" in df.columns and "time_bin" not in df.columns:
                df["time_bin"] = df["hour"].astype(int)
            cat_prob_map = {"shopping": 0.344749, "electronics": 0.328588, "food": 0.329321}
            if "Category" in df.columns and "category_prob" not in df.columns:
                df["category_prob"] = df["Category"].map(cat_prob_map).fillna(np.mean(list(cat_prob_map.values())))
            if "category_prob" in df.columns and "category_deviation" not in df.columns:
                df["category_deviation"] = 1.0 - df["category_prob"]
            return df
        except Exception:
            st.info("File `payment_fraud.csv` tidak ditemukan di root repo.")
            return None

    # Normalisasi dan turunkan fitur rekayasa bila belum ada
    if "paymentMethod" in df.columns:
        df["paymentMethod"] = df["paymentMethod"].astype(str).str.strip().str.lower()
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.strip().str.lower()

    if "hour" in df.columns and "isNight" not in df.columns:
        df["isNight"] = ((df["hour"] >= 21) | (df["hour"] < 6)).astype(int)
    if "paymentMethod" in df.columns and "isHighRiskPayment" not in df.columns:
        df["isHighRiskPayment"] = (df["paymentMethod"] == "paypal").astype(int)
    if "hour" in df.columns and "time_bin" not in df.columns:
        df["time_bin"] = df["hour"].astype(int)

    # Mapping category_prob default (sesuai insight notebook)
    cat_prob_map = {"shopping": 0.344749, "electronics": 0.328588, "food": 0.329321}
    if "Category" in df.columns and "category_prob" not in df.columns:
        df["category_prob"] = df["Category"].map(cat_prob_map).fillna(np.mean(list(cat_prob_map.values())))
    if "category_prob" in df.columns and "category_deviation" not in df.columns:
        df["category_deviation"] = 1.0 - df["category_prob"]

    return df


def _hist_by_label(df, col, bins):
    """Return pivoted histogram counts by label for Streamlit charts."""
    x = df[[col, "label"]].dropna().copy()
    x["bin"] = pd.cut(x[col], bins=bins, include_lowest=True)
    x["mid"] = x["bin"].apply(lambda b: b.mid if pd.notnull(b) else np.nan)
    pivot = x.groupby(["mid", "label"]).size().unstack(fill_value=0).sort_index()
    pivot.columns = ["Legitimate(0)" if c == 0 else "Fraud(1)" for c in pivot.columns]
    return pivot


# -------- page --------
def eda():
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    df = _load_dataset()
    if df is None:
        return

    st.subheader("Preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")
    st.markdown("---")

    # Bangun daftar pertanyaan sesuai kolom yang tersedia
    options = []
    if "label" in df.columns:
        options.append("Distribusi label (imbalance)")
    if {"paymentMethod", "label"}.issubset(df.columns):
        options.append("Fraud rate per paymentMethod")
    if {"Category", "label"}.issubset(df.columns):
        options.append("Fraud rate per Category")
    if {"hour", "label"}.issubset(df.columns):
        options.append("Fraud rate per jam (hour)")
    if {"isNight", "label"}.issubset(df.columns):
        options.append("Fraud rate: Night vs Non-Night")
    if {"temporal_risk_window", "label"}.issubset(df.columns):
        options.append("Fraud rate: Temporal Risk Window")
    if {"risk_score", "label"}.issubset(df.columns):
        options.append("Distribusi risk_score menurut label")
    if {"transaction_velocity", "label"}.issubset(df.columns):
        options.append("Distribusi transaction_velocity menurut label")
    if {"payment_age_ratio", "label"}.issubset(df.columns):
        options.append("Distribusi payment_age_ratio menurut label")

    if not options:
        st.info("Kolom kunci untuk EDA belum lengkap. Pastikan minimal ada kolom `label`.")
        return

    choice = st.selectbox("Pilih EDA", options)
    st.markdown("---")

    # 1) Class imbalance
    if choice == "Distribusi label (imbalance)":
        vc = df["label"].value_counts(dropna=False).rename({0: "Legitimate(0)", 1: "Fraud(1)"}).sort_index()
        fig, ax = plt.subplots(figsize=(16, 5))
        bars = ax.bar(vc.index, vc.values, color=["#1f77b4", "#ff7f0e"])
        
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Jumlah Transaksi")
        ax.set_title("Distribusi Label (Imbalance)")
        ax.set_xticks(range(len(vc.index)))
        ax.set_xticklabels(vc.index, rotation=0)  # rotation=0 bikin horizontal

        # Tambah angka di atas bar
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)
        st.caption(f"Fraud rate keseluruhan: **{df['label'].mean():.2%}** — menunjukkan dataset imbalanced.")
        st.write('''Implikasi: fokus pada **recall** kelas fraud & tuning **threshold**." \
**Insight:**
- Kelas sangat tidak seimbang, hanya 1.09% fraud (396 dari 36188 transaksi).
- Evaluasi model harus menggunakan F1-Score, bukan accuracy.
- Jika model selalu memprediksi "tidak fraud", akurasinya mencapai 99.1%, tetapi tidak berguna karena gagal mendeteksi kasus fraud.
                 ''')

    # 2) Fraud rate per paymentMethod
    elif choice == "Fraud rate per paymentMethod":
        rate = df.groupby("paymentMethod")["label"].mean()
        cnt  = df["paymentMethod"].value_counts()

        out = (
            pd.DataFrame({"fraud_rate": rate, "count": cnt})
            .fillna(0)
            .sort_values("fraud_rate", ascending=False)
        )

        # --- Plot (matplotlib) ---
        fig, ax = plt.subplots(figsize=(16, 5))
        bars = ax.bar(out.index, out["fraud_rate"], width=0.6)

        ax.set_title("Fraud Rate per Payment Method", fontsize=14)
        ax.set_xlabel("Payment Method", fontsize=12)
        ax.set_ylabel("Fraud Rate", fontsize=12)
        ax.set_xticks(range(len(out.index)))
        ax.set_xticklabels(out.index, rotation=0, fontsize=12)
        ax.set_ylim(0, out["fraud_rate"].max() * 1.25)

        # Label persentase di atas bar
        for b, v in zip(bars, out["fraud_rate"]):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2%}",
                    ha="center", va="bottom", fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)

        # --- Tabel di bawah chart ---
        disp = out.copy().rename_axis("paymentMethod").reset_index()
        disp["fraud_rate"] = disp["fraud_rate"].map(lambda x: f"{x:.2%}")
        disp["count"] = disp["count"].astype(int)
        st.dataframe(disp, use_container_width=True)
        st.write('''
**Insight Bisnis:**
- **PayPal:** Memiliki **fraud rate tertinggi (1.21%)**, mungkin karena PayPal sering digunakan dalam **transaksi besar atau internasional**, yang lebih rentan terhadap penipuan.
- **CreditCard:** Memiliki **fraud rate paling rendah (1.12%)**, mungkin karena sistem verifikasi lebih ketat dibandingkan metode lainnya.
- **StoreCredit:** Berada di tengah-tengah (1.05%), menunjukkan pola yang stabil.

**Interpretasi Bisnis:**
- **Prioritas Monitoring:** Transaksi menggunakan **PayPal harus dipantau lebih ketat**, terutama jika melibatkan jumlah besar atau transaksi internasional.
- **Relevansi Feature Engineering:** Bisa membuat fitur baru seperti `isHighRiskPaymentMethod` berdasarkan insight ini, misalnya:
  - `1` jika `paymentMethod == 'paypal'`
  - `0` jika tidak.
''')

    # 3) Fraud rate per Category
    elif choice == "Fraud rate per Category":
        rate = df.groupby("Category")["label"].mean().sort_values(ascending=False)
        cnt = df["Category"].value_counts()
        out = pd.DataFrame({"fraud_rate": rate, "count": cnt}).fillna(0)

        # Plot pakai matplotlib supaya kontrol penuh
        fig, ax = plt.subplots(figsize=(16, 5))
        out["fraud_rate"].plot(kind="bar", ax=ax, color="royalblue")

        ax.set_ylabel("Fraud Rate")
        ax.set_xlabel("Category")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # horizontal
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # biar jadi %
        plt.tight_layout()

        st.pyplot(fig)
        st.dataframe(out.style.format({"fraud_rate": "{:.2%}"}))
        st.write('''
    **Insight Bisnis:**
- **Kategori `shopping` memiliki fraud rate tertinggi (1.14%)**, diikuti oleh `electronics` (1.08%) dan `food` (1.06%).
- Semua kategori memiliki **fraud rate yang relatif rendah**, tetapi `shopping` sedikit lebih tinggi dibandingkan dua lainnya.

**Interpretasi Bisnis:**
- **Produk Shopping:**  
Transaksi belanja umum cenderung memiliki volume besar, sehingga potensi fraud juga lebih tinggi.  
Ini bisa disebabkan oleh:
- **Transaksi dalam jumlah besar:** Belanja online sering kali melibatkan banyak item atau nilai transaksi tinggi.
- **Kurangnya verifikasi tambahan:** Beberapa platform e-commerce mungkin tidak memiliki kontrol keamanan ekstra untuk transaksi shopping.

- **Produk Electronics:**  
Kategori ini memiliki **fraud rate cukup tinggi (1.08%)**, meskipun sedikit lebih rendah daripada `shopping`.  
Ini mungkin karena:
- **Harga produk elektronik yang tinggi:** Barang elektronik seperti smartphone, laptop, atau gadget mahal menjadi target pelaku fraud.
- **Pengiriman internasional:** Produk elektronik sering dikirim ke luar negeri, yang meningkatkan risiko penipuan.

- **Produk Food:**  
Kategori ini memiliki **fraud rate paling rendah (1.06%)**, tetapi masih relatif tinggi dibandingkan kategori lainnya.  
Ini mungkin karena:
- **Nilai transaksi lebih rendah:** Pembelian makanan biasanya memiliki nilai yang lebih kecil dibandingkan produk lainnya.
- **Verifikasi lebih ketat:** Platform e-commerce mungkin memiliki kontrol tambahan untuk transaksi food, seperti verifikasi alamat pengiriman atau pembatasan jumlah transaksi harian.
''')

    # 4) Fraud rate per hour
    elif choice == "Fraud rate per Category":
        dfx = df.copy()

        # 1) Normalisasi string TANPA mengubah NA asli
        s = pd.Series(dfx["Category"], dtype="string").str.strip().str.lower()

        # 2) Placeholder -> NA (hindari kategori "nan"/"null"/"")
        s = s.replace({"nan": pd.NA, "none": pd.NA, "null": pd.NA, "": pd.NA})

        # 3) Isi NA dengan modus setelah normalisasi
        if s.isna().all():
            s = s.fillna("unknown")            # fallback jika semuanya NA
        else:
            s = s.fillna(s.dropna().mode().iat[0])

        dfx["Category"] = s

        # 4) Agregasi
        rate = dfx.groupby("Category")["label"].mean()
        cnt  = dfx["Category"].value_counts()
        out = (
            pd.DataFrame({"fraud_rate": rate, "count": cnt})
            .fillna(0)
            .sort_values("fraud_rate", ascending=False)
        )

        # 5) Plot (x-axis horizontal, y dalam %)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(out.index, out["fraud_rate"], color="royalblue", width=0.6)

        ax.set_title("Fraud Rate per Category", fontsize=14)
        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Fraud Rate", fontsize=12)
        ax.set_xticklabels(out.index, rotation=0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, out["fraud_rate"].max() * 1.25 if len(out) else 1)

        # label % di atas bar
        for b, v in zip(bars, out["fraud_rate"]):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2%}", ha="center", va="bottom", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        # 6) Tabel bawah
        st.dataframe(
            out.reset_index().rename(columns={"index": "Category"})
            .style.format({"fraud_rate": "{:.2%}"}),
            use_container_width=True
        )        
        st.caption('''Pola jam membantu justifikasi **time_bin** & fitur **isNight**.
**Insight:** 
- Fraud Terjadi di Akun Baru<br>
    Semua transaksi fraud (`label = 1`) dilakukan dari akun yang berusia hanya 1 hari. Hal ini menunjukkan bahwa akun baru sangat rentan terhadap penyalahgunaan, kemungkinan besar karena dibuat khusus untuk tujuan penipuan.
- Akun Lama Lebih Aman<br>
    Transaksi legitimate (`label = 0`) memiliki usia akun rata-rata 823 hari (lebih dari 2 tahun), dengan variasi besar. Akun lama cenderung lebih aman karena telah melewati proses verifikasi dan membangun riwayat transaksi positif.
                   ''')

    # 5) Night vs Non-Night
    elif choice == "Fraud rate: Night vs Non-Night":
        rate = df.groupby("isNight")["label"].mean().rename({0: "Non-Night", 1: "Night"})
        st.bar_chart(rate)
        st.caption('''Jika malam (Night) lebih tinggi → validasi fitur **isNight**.
**Insight:**
- **Metode Pembayaran Baru pada Fraud**<br>
Median usia metode pembayaran pada transaksi fraud adalah 0,000694 hari (~60 detik) dengan maksimum 0,991667 hari (~23 jam). Pada transaksi legitimate, median-nya 0,268056 hari (~6,4 jam) dan maksimum 1999 hari. Ini menunjukkan pelaku fraud hampir selalu menggunakan metode pembayaran yang baru saja ditambahkan.
- **Akun Fraud Selalu Baru**<br>
Semua akun fraud berusia tepat 1 hari (min = max = median = 1), sementara akun legitimate memiliki median 569 hari, dengan rentang 2-2000 hari. Hal ini menunjukkan bahwa akun baru sangat rentan terhadap penipuan, karena pelaku fraud mungkin menggunakan akun palsu atau akun yang belum stabil.
- **Aturan Deteksi Potensial**<br>
Pada scatter plot, semua titik fraud berada di area Usia Akun <= 1 hari dan Usia Metode Pembayaran <= 1 hari, sedangkan titik legitimate tersebar di seluruh rentang. Kombinasi dua kondisi ini dapat digunakan sebagai aturan deteksi yang jelas dan terukur.

**Interpretasi Bisnis:**
- **Prioritas Monitoring**: Akun baru dan metode pembayaran baru harus dipantau lebih ketat, terutama dalam beberapa hari pertama setelah pembuatan akun atau pengaktifan metode pembayaran.
- **Feature Engineering**: Bisa membuat fitur baru seperti:
  - `isNewAccount` (1 jika `accountAgeDays <= 7`, 0 jika tidak).
  - `isNewPaymentMethod` (1 jika `paymentMethodAgeDays <= 7`, 0 jika tidak).
                   ''')

    # 6) Temporal risk window
    elif choice == "Fraud rate: Temporal Risk Window":
        rate = df.groupby("temporal_risk_window")["label"].mean().rename({0: "Window=0", 1: "Window=1"})
        st.bar_chart(rate)
        st.caption('''Window berisiko menaikkan probabilitas; mendukung fitur **temporal_risk_window**.
Visualisasi ini menunjukkan **fraud rate** (rata-rata label `1`) untuk transaksi pada hari kerja (`isWeekend = 0`) dan akhir pekan (`isWeekend = 1`).

**Insight Bisnis**:
- **Transaksi di akhir pekan memiliki fraud rate tertinggi (2.16%)**, sedangkan transaksi di hari kerja memiliki fraud rate rendah (0%).
- Ini menunjukkan bahwa **transaksi fraud cenderung terjadi di akhir pekan**, mungkin karena:
    - **Volume transaksi lebih tinggi**: Akhir pekan sering kali menjadi waktu puncak untuk belanja online atau transaksi besar.
    - **Pengawasan lebih lemah**: Tim keamanan atau sistem pemantauan mungkin tidak beroperasi secara penuh selama akhir pekan.
                   ''')

    # 7) risk_score distribution by label
    elif choice == "Distribusi risk_score menurut label":
        bins = np.linspace(0, 1, 21)
        pivot = _hist_by_label(df, "risk_score", bins=bins)
        st.bar_chart(pivot)
        st.caption("Histogram terpisah per label; **risk_score** tinggi di fraud menandakan sinyal yang relevan.")

    # 8) transaction_velocity distribution by label
    elif choice == "Distribusi transaction_velocity menurut label":
        # auto-bins berdasarkan IQR
        q1, q3 = df["transaction_velocity"].quantile([0.25, 0.75])
        iqr = max(q3 - q1, 1e-6)
        step = max(round(iqr / 5, 2), 0.5)
        bins = np.arange(df["transaction_velocity"].min(), df["transaction_velocity"].max() + step, step)
        pivot = _hist_by_label(df, "transaction_velocity", bins=bins)
        st.bar_chart(pivot)
        st.caption("Velocity tinggi pada fraud bisa mengindikasikan perilaku mencurigakan.")

    # 9) payment_age_ratio distribution by label
    elif choice == "Distribusi payment_age_ratio menurut label":
        bins = np.linspace(0, 1, 21)
        pivot = _hist_by_label(df, "payment_age_ratio", bins=bins)
        st.bar_chart(pivot)
        st.caption("Rasio rendah pada fraud konsisten dengan **alat bayar baru** yang berisiko.")

# jalankan lokal untuk debug manual:
if __name__ == "__main__":
    eda()