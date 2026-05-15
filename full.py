import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Cuaca DKI Jakarta",
    page_icon="🌤️",
    layout="wide",
)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
WEATHER_LABELS = ['Badai Petir', 'Berawan', 'Cerah', 'Embun', 'Hujan', 'Kabut asap']
LOCATIONS = ['Jakarta Barat', 'Jakarta Pusat', 'Jakarta Selatan',
             'Jakarta Timur', 'Jakarta Utara']
COLORS = ['#ec4899', '#94a3b8', '#fbbf24', '#a78bfa', '#3b82f6', '#78716c']

FEATURE_COLS = ['Suhu (celcius)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV', 'Lokasi', 'Waktu']

DISPLAY_COL_MAP = {
    'Suhu (celcius)': 'Suhu (°C)',
    'Kelembapan (%)': 'Kelembapan (%)',
    'Angin (km/h)': 'Angin (km/h)',
    'Intensitas UV': 'Intensitas UV',
}

ICONS = {
    'Cerah': '☀️',
    'Berawan': '⛅',
    'Hujan': '🌧️',
    'Badai Petir': '⛈️',
    'Kabut asap': '🌫️',
    'Embun': '🌁',
}


# ─── LOAD DATA & MODELS ────────────────────────────────────────────────────────
@st.cache_data
def load_main_data():
    df = pd.read_excel("data_preprocessed.xlsx")
    df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.strftime('%d/%m/%Y')
    return df


@st.cache_data
def load_folds():
    folds = []
    for i in range(1, 6):
        fold_df = pd.read_excel(f"fold_{i}.xlsx")
        fold_df['Tanggal'] = pd.to_datetime(fold_df['Tanggal']).dt.strftime('%d/%m/%Y')
        folds.append(fold_df)
    return folds


@st.cache_resource
def load_models():
    models_fold = []
    for i in range(1, 6):
        m = joblib.load(f"model_fold_{i}.pkl")
        models_fold.append(m)
    model_full = joblib.load("model_full.pkl")
    return models_fold, model_full


@st.cache_resource
def load_label_encoder():
    from sklearn.preprocessing import LabelEncoder
    # Fit encoder dengan urutan yang sama persis seperti saat training
    df = pd.read_excel("data_preprocessed.xlsx")
    le_lokasi = LabelEncoder()
    le_lokasi.fit(df['Lokasi'])
    le_target = LabelEncoder()
    le_target.fit(df['Cuaca'])
    return le_lokasi, le_target


def encode_features(df_input, le_lokasi):
    """Encode Lokasi dan pastikan Waktu numerik, kemudian pilih fitur."""
    df = df_input.copy()
    df['Lokasi'] = le_lokasi.transform(df['Lokasi'])
    df['Waktu'] = pd.to_numeric(df['Waktu'], errors='coerce')
    return df[FEATURE_COLS]


def compute_fold_results(folds, models_fold, le_lokasi, le_target):
    results = []
    for fold_idx, (fold_df, model) in enumerate(zip(folds, models_fold)):
        test_df = fold_df.copy()
        actuals_str = test_df['Cuaca'].tolist()

        X_test = encode_features(test_df, le_lokasi)
        y_pred_encoded = model.predict(X_test)
        preds_str = le_target.inverse_transform(y_pred_encoded).tolist()

        test_df = test_df.copy()
        test_df['Label Aktual'] = actuals_str
        test_df['Hasil Prediksi'] = preds_str
        test_df['Keterangan'] = ['✅ Sesuai' if a == p else '❌ Tidak Sesuai'
                                  for a, p in zip(actuals_str, preds_str)]

        labels_present = sorted(set(actuals_str + preds_str))
        all_labels = WEATHER_LABELS
        label_to_idx = {l: i for i, l in enumerate(all_labels)}

        cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        for a, p in zip(actuals_str, preds_str):
            if a in label_to_idx and p in label_to_idx:
                cm[label_to_idx[a]][label_to_idx[p]] += 1

        precision_list, recall_list, f1_list = [], [], []
        for i in range(len(all_labels)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0
            precision_list.append(p_val)
            recall_list.append(r_val)
            f1_list.append(f1_val)

        acc = sum(a == p for a, p in zip(actuals_str, preds_str)) / len(actuals_str)

        results.append({
            'fold': fold_idx + 1,
            'test_df': test_df,
            'actuals': actuals_str,
            'preds': preds_str,
            'cm': cm,
            'accuracy': acc,
            'precision': np.mean(precision_list),
            'recall': np.mean(recall_list),
            'f1': np.mean(f1_list),
        })
    return results


# ─── LOAD ──────────────────────────────────────────────────────────────────────
try:
    df = load_main_data()
    folds = load_folds()
    models_fold, model_full = load_models()
    le_lokasi, le_target = load_label_encoder()
    fold_results = compute_fold_results(folds, models_fold, le_lokasi, le_target)
    data_loaded = True
except Exception as e:
    data_loaded = False
    load_error = str(e)

if not data_loaded:
    st.error(
        f"❌ Gagal memuat data atau model. Pastikan file berikut ada di direktori yang sama dengan `full.py`:\n\n"
        f"- `data_preprocessed.xlsx`\n"
        f"- `fold_1.xlsx` s.d. `fold_5.xlsx`\n"
        f"- `model_fold_1.pkl` s.d. `model_fold_5.pkl`\n"
        f"- `model_full.pkl`\n\n"
        f"**Error:** `{load_error}`"
    )
    st.stop()

avg_acc  = np.mean([r['accuracy']  for r in fold_results])
avg_prec = np.mean([r['precision'] for r in fold_results])
avg_rec  = np.mean([r['recall']    for r in fold_results])
avg_f1   = np.mean([r['f1']        for r in fold_results])

# Pastikan semua label cuaca dari data nyata ada di WEATHER_LABELS
actual_labels_in_data = sorted(df['Cuaca'].unique().tolist())


# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.title("🌤️ Sistem Klasifikasi Cuaca DKI Jakarta")
st.write(
    "Aplikasi ini melakukan klasifikasi kondisi cuaca pada lima wilayah administratif "
    "Provinsi DKI Jakarta menggunakan algoritma **Random Forest** dengan evaluasi model "
    "menggunakan metode **K-Fold Cross Validation (k=5)**."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data", f"{len(df):,}", "observasi")
col2.metric("Wilayah", "5", "administratif DKI")
col3.metric("Kategori Cuaca", str(len(actual_labels_in_data)), "kelas target")
col4.metric("Rata-rata Akurasi", f"{avg_acc*100:.1f}%", "K-Fold CV")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CUACA
# ════════════════════════════════════════════════════════════════════════════════
st.header("📊 Data Cuaca")
st.write(
    "Dataset observasi cuaca dari 5 wilayah administratif DKI Jakarta, "
    "dikumpulkan pukul 09.00 dan 15.00 WIB setiap harinya."
)

col_f1, col_f2, _ = st.columns([1, 1, 2])
with col_f1:
    filter_loc = st.selectbox("Filter Lokasi", ["Semua"] + sorted(df['Lokasi'].unique().tolist()))
with col_f2:
    filter_cuaca = st.selectbox("Filter Cuaca", ["Semua"] + actual_labels_in_data)

display_df = df.copy()
if filter_loc != "Semua":
    display_df = display_df[display_df['Lokasi'] == filter_loc]
if filter_cuaca != "Semua":
    display_df = display_df[display_df['Cuaca'] == filter_cuaca]

# Rename kolom untuk tampilan
display_df_show = display_df.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
st.dataframe(display_df_show.reset_index(drop=True), use_container_width=True, height=300)
st.caption(f"Menampilkan {len(display_df)} dari {len(df)} data.")

# Distribusi kelas
st.subheader("Distribusi Kategori Cuaca")
cuaca_counts = df['Cuaca'].value_counts().reindex(actual_labels_in_data, fill_value=0)
cols_dist = st.columns(len(actual_labels_in_data))
for col, cat in zip(cols_dist, actual_labels_in_data):
    cnt = cuaca_counts[cat]
    pct = cnt / len(df) * 100
    icon = ICONS.get(cat, '🌡️')
    col.metric(f"{icon} {cat}", cnt, f"{pct:.1f}%")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EVALUASI MODEL (K-FOLD)
# ════════════════════════════════════════════════════════════════════════════════
st.header("🔁 Evaluasi Model — K-Fold Cross Validation (k=5)")
st.write(
    "Setiap fold menampilkan hasil evaluasi model pada data uji yang berbeda. "
    "Dataset dibagi menjadi 5 bagian; setiap bagian bergantian menjadi data uji."
)
st.info("💡 Pilih tab fold untuk melihat hasil evaluasi, confusion matrix, dan metrik masing-masing fold.")

fold_tabs = st.tabs([f"Fold {i+1}" for i in range(5)])

for fi, tab in enumerate(fold_tabs):
    r = fold_results[fi]
    with tab:
        st.subheader(f"Fold {r['fold']} — Hasil Evaluasi")
        st.caption(f"Jumlah data uji: {len(r['actuals'])} sampel")

        # Tabel prediksi
        col_tbl, col_met = st.columns([3, 1.5])
        with col_tbl:
            tdf = r['test_df'].copy()
            tdf = tdf.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
            preview = tdf[[
                'Tanggal', 'Waktu', 'Lokasi',
                'Suhu (°C)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV',
                'Label Aktual', 'Hasil Prediksi', 'Keterangan'
            ]].head(20).reset_index(drop=True)
            st.dataframe(preview, use_container_width=True, height=280)

        # Metrik
        with col_met:
            st.metric("Akurasi",  f"{r['accuracy']*100:.2f}%")
            st.metric("Presisi",  f"{r['precision']*100:.2f}%")
            st.metric("Recall",   f"{r['recall']*100:.2f}%")
            st.metric("F1-Score", f"{r['f1']*100:.2f}%")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = r['cm']
        cmap = LinearSegmentedColormap.from_list('blues', ['#f0f9ff', '#0369a1'])
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(cm, cmap=cmap, aspect='auto')

        ax.set_xticks(range(len(WEATHER_LABELS)))
        ax.set_yticks(range(len(WEATHER_LABELS)))
        short = [c.replace(' ', '\n') for c in WEATHER_LABELS]
        ax.set_xticklabels(short, fontsize=8)
        ax.set_yticklabels(WEATHER_LABELS, fontsize=8)
        ax.set_xlabel('Prediksi', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_ylabel('Aktual', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title(f'Confusion Matrix — Fold {r["fold"]}', fontsize=11, fontweight='bold', pad=12)

        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(len(WEATHER_LABELS)):
            for j in range(len(WEATHER_LABELS)):
                val = cm[i, j]
                color = 'white' if val > max_val * 0.6 else '#0f172a'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RINGKASAN EVALUASI
# ════════════════════════════════════════════════════════════════════════════════
st.header("📋 Ringkasan Evaluasi")
st.write("Nilai rata-rata dari seluruh fold K-Fold Cross Validation (k=5).")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Rata-rata Akurasi",  f"{avg_acc*100:.2f}%")
col_s2.metric("Rata-rata Presisi",  f"{avg_prec*100:.2f}%")
col_s3.metric("Rata-rata Recall",   f"{avg_rec*100:.2f}%")
col_s4.metric("Rata-rata F1-Score", f"{avg_f1*100:.2f}%")

st.write("**Tabel Metrik per Fold**")
fold_table = pd.DataFrame({
    'Fold':         [f"Fold {r['fold']}" for r in fold_results],
    'Akurasi (%)':  [round(r['accuracy']  * 100, 2) for r in fold_results],
    'Presisi (%)':  [round(r['precision'] * 100, 2) for r in fold_results],
    'Recall (%)':   [round(r['recall']    * 100, 2) for r in fold_results],
    'F1-Score (%)': [round(r['f1']        * 100, 2) for r in fold_results],
})
avg_row = pd.DataFrame([{
    'Fold': 'Rata-rata',
    'Akurasi (%)':  round(avg_acc  * 100, 2),
    'Presisi (%)':  round(avg_prec * 100, 2),
    'Recall (%)':   round(avg_rec  * 100, 2),
    'F1-Score (%)': round(avg_f1   * 100, 2),
}])
fold_table_full = pd.concat([fold_table, avg_row], ignore_index=True)
st.dataframe(fold_table_full, use_container_width=True, hide_index=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISASI
# ════════════════════════════════════════════════════════════════════════════════
st.header("📈 Visualisasi Distribusi Cuaca")
st.write("Distribusi kategori cuaca pada data uji per fold dan per wilayah.")

viz_tabs = st.tabs([f"Fold {i+1}" for i in range(5)] + ["Semua Wilayah"])

for fi, tab in enumerate(viz_tabs[:-1]):
    r = fold_results[fi]
    with tab:
        col_v1, col_v2 = st.columns(2)

        actual_counts = pd.Series(r['actuals']).value_counts().reindex(WEATHER_LABELS, fill_value=0)
        pred_counts   = pd.Series(r['preds']).value_counts().reindex(WEATHER_LABELS, fill_value=0)

        with col_v1:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(WEATHER_LABELS, actual_counts.values,
                          color=COLORS, edgecolor='white', linewidth=1.5, width=0.65)
            ax.set_title(f'Distribusi Label Aktual — Fold {r["fold"]}',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Jumlah', fontsize=9)
            ax.set_xticklabels([c.replace(' ', '\n') for c in WEATHER_LABELS], fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, actual_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(val),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_v2:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(WEATHER_LABELS, pred_counts.values,
                          color=COLORS, edgecolor='white', linewidth=1.5,
                          width=0.65, alpha=0.85)
            ax.set_title(f'Distribusi Hasil Prediksi — Fold {r["fold"]}',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Jumlah', fontsize=9)
            ax.set_xticklabels([c.replace(' ', '\n') for c in WEATHER_LABELS], fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, pred_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(val),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

with viz_tabs[-1]:
    st.subheader("Distribusi Cuaca per Wilayah Administratif DKI Jakarta")
    fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=False)
    for loc, ax in zip(sorted(df['Lokasi'].unique()), axes):
        loc_df = df[df['Lokasi'] == loc]
        counts = loc_df['Cuaca'].value_counts().reindex(WEATHER_LABELS, fill_value=0)
        ax.bar(range(len(WEATHER_LABELS)), counts.values,
               color=COLORS, edgecolor='white', linewidth=1.2, width=0.7)
        ax.set_title(loc.replace('Jakarta ', 'Jkt\n'), fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, WEATHER_LABELS)]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    fig.suptitle('Distribusi Cuaca per Wilayah DKI Jakarta',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PERBANDINGAN
# ════════════════════════════════════════════════════════════════════════════════
st.header("⚖️ Perbandingan Hasil Klasifikasi")
st.write(
    "Perbandingan antara label aktual (data Weather Channel) "
    "dengan hasil prediksi model Random Forest, per fold."
)

comp_tabs = st.tabs([f"Fold {i+1}" for i in range(5)])

for fi, tab in enumerate(comp_tabs):
    r = fold_results[fi]
    with tab:
        test_df_show = r['test_df'].copy()
        test_df_show = test_df_show.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
        match_count    = (test_df_show['Keterangan'] == '✅ Sesuai').sum()
        mismatch_count = (test_df_show['Keterangan'] == '❌ Tidak Sesuai').sum()
        total_count    = len(test_df_show)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("✅ Sesuai",        match_count,
                     f"{match_count/total_count*100:.1f}% dari data uji")
        col_b.metric("❌ Tidak Sesuai",  mismatch_count,
                     f"{mismatch_count/total_count*100:.1f}% dari data uji")
        col_c.metric("Total Data Uji",   total_count,
                     f"Fold {r['fold']}")

        comp_show = test_df_show[[
            'Tanggal', 'Waktu', 'Lokasi',
            'Suhu (°C)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV',
            'Label Aktual', 'Hasil Prediksi', 'Keterangan'
        ]].reset_index(drop=True)
        st.dataframe(comp_show, use_container_width=True, height=350)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — KLASIFIKASI DATA BARU
# ════════════════════════════════════════════════════════════════════════════════
st.header("🔮 Klasifikasi Data Baru")
st.write(
    "Masukkan data observasi cuaca baru untuk diklasifikasikan menggunakan "
    "**model_full.pkl** (model yang dilatih dengan seluruh data). "
    "Pastikan semua kolom terisi dengan benar."
)

with st.form("form_klasifikasi"):
    st.subheader("Input Data Observasi")

    col_i1, col_i2, col_i3 = st.columns(3)

    with col_i1:
        input_lokasi = st.selectbox(
            "Lokasi",
            options=sorted(df['Lokasi'].unique().tolist()),
            help="Pilih wilayah administratif DKI Jakarta"
        )
        input_waktu = st.selectbox(
            "Waktu Pengamatan",
            options=[9, 15],
            format_func=lambda x: "09.00 WIB" if x == 9 else "15.00 WIB",
            help="Waktu pengamatan cuaca"
        )

    with col_i2:
        input_suhu = st.number_input(
            "Suhu (°C)",
            min_value=15.0, max_value=45.0,
            value=float(df['Suhu (celcius)'].median()),
            step=0.1,
            help="Suhu udara dalam derajat Celsius"
        )
        input_kelembapan = st.number_input(
            "Kelembapan (%)",
            min_value=0.0, max_value=100.0,
            value=float(df['Kelembapan (%)'].median()),
            step=0.1,
            help="Kelembapan relatif udara dalam persen"
        )

    with col_i3:
        input_angin = st.number_input(
            "Kecepatan Angin (km/h)",
            min_value=0.0, max_value=150.0,
            value=float(df['Angin (km/h)'].median()),
            step=0.1,
            help="Kecepatan angin dalam km/jam"
        )
        input_uv = st.number_input(
            "Intensitas UV",
            min_value=0.0, max_value=15.0,
            value=float(df['Intensitas UV'].median()),
            step=0.1,
            help="Indeks UV (0–15)"
        )

    submitted = st.form_submit_button("🔍 Klasifikasikan", use_container_width=True, type="primary")

if submitted:
    try:
        # Buat DataFrame input dengan nama kolom yang sama persis seperti saat training
        input_data = pd.DataFrame([{
            'Suhu (celcius)': input_suhu,
            'Kelembapan (%)': input_kelembapan,
            'Angin (km/h)': input_angin,
            'Intensitas UV': input_uv,
            'Lokasi': input_lokasi,
            'Waktu': input_waktu,
        }])

        X_input = encode_features(input_data, le_lokasi)
        pred_encoded = model_full.predict(X_input)
        pred_proba = model_full.predict_proba(X_input)[0]
        pred_label = le_target.inverse_transform(pred_encoded)[0]

        # Tampilkan hasil
        st.success(f"### Hasil Klasifikasi: {ICONS.get(pred_label, '🌡️')} **{pred_label}**")

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])

        with col_r1:
            st.subheader("Ringkasan Input")
            st.markdown(f"""
| Parameter | Nilai |
|-----------|-------|
| Lokasi | {input_lokasi} |
| Waktu | {'09.00 WIB' if input_waktu == 9 else '15.00 WIB'} |
| Suhu | {input_suhu:.1f} °C |
| Kelembapan | {input_kelembapan:.1f} % |
| Angin | {input_angin:.1f} km/h |
| Intensitas UV | {input_uv:.1f} |
""")

        with col_r2:
            st.subheader("Probabilitas per Kelas")
            classes = le_target.classes_
            proba_df = pd.DataFrame({
                'Kelas Cuaca': classes,
                'Probabilitas (%)': (pred_proba * 100).round(2)
            }).sort_values('Probabilitas (%)', ascending=False).reset_index(drop=True)

            fig_proba, ax_proba = plt.subplots(figsize=(6, 3.5))
            bar_colors = [COLORS[WEATHER_LABELS.index(c)] if c in WEATHER_LABELS else '#888'
                          for c in proba_df['Kelas Cuaca']]
            bars_proba = ax_proba.barh(
                proba_df['Kelas Cuaca'], proba_df['Probabilitas (%)'],
                color=bar_colors, edgecolor='white', linewidth=1.2
            )
            ax_proba.set_xlabel('Probabilitas (%)', fontsize=9)
            ax_proba.set_title('Distribusi Probabilitas Prediksi', fontsize=10, fontweight='bold')
            ax_proba.spines['top'].set_visible(False)
            ax_proba.spines['right'].set_visible(False)
            ax_proba.set_xlim(0, 105)
            for bar, val in zip(bars_proba, proba_df['Probabilitas (%)']):
                ax_proba.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                              f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')
            fig_proba.tight_layout()
            st.pyplot(fig_proba, use_container_width=True)
            plt.close()

        with col_r3:
            st.subheader("Prediksi")
            st.metric(
                label="Kelas Prediksi",
                value=f"{ICONS.get(pred_label, '🌡️')} {pred_label}",
                delta=f"Kepercayaan: {pred_proba.max()*100:.1f}%"
            )
            st.dataframe(
                proba_df.rename(columns={'Probabilitas (%)': 'Prob (%)'}),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"❌ Terjadi error saat klasifikasi: `{str(e)}`")

st.divider()
st.caption(
    "Sistem Klasifikasi Cuaca DKI Jakarta · Random Forest + K-Fold Cross Validation · "
    "Yoga Ramadhani Kabakora · NIM 535220247 · UNTAR 2026"
)