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


def apply_filters(df, loc_filter, tanggal_filter, waktu_filter, cuaca_filter, variasi_filter=False):
    """Terapkan filter ke DataFrame."""
    filtered = df.copy()
    if loc_filter and loc_filter != "Semua":
        filtered = filtered[filtered['Lokasi'] == loc_filter]
    if tanggal_filter and tanggal_filter != "Semua":
        filtered = filtered[filtered['Tanggal'] == tanggal_filter]
    if waktu_filter and waktu_filter != "Semua":
        waktu_val = int(waktu_filter.split('.')[0])
        filtered = filtered[filtered['Waktu'] == waktu_val]
    if cuaca_filter and cuaca_filter != "Semua":
        cuaca_col = 'Cuaca' if 'Cuaca' in filtered.columns else 'Label Aktual'
        if cuaca_col in filtered.columns:
            filtered = filtered[filtered[cuaca_col] == cuaca_filter]
    if variasi_filter:
        # Tampilkan baris di mana pada tanggal+waktu yang sama ada perbedaan kategori cuaca antar lokasi
        cuaca_col = 'Cuaca' if 'Cuaca' in filtered.columns else 'Label Aktual'
        if cuaca_col in filtered.columns and 'Tanggal' in filtered.columns and 'Waktu' in filtered.columns:
            group_cols = ['Tanggal', 'Waktu']
            variasi = filtered.groupby(group_cols)[cuaca_col].nunique()
            variasi_idx = variasi[variasi > 1].index
            filtered = filtered[pd.MultiIndex.from_arrays([filtered['Tanggal'], filtered['Waktu']]).isin(variasi_idx)]
    return filtered


def render_section_filters(prefix, df, include_cuaca=True, cuaca_col='Cuaca',
                           include_status=False, include_variasi=False):
    """Render filter per section. Kembalikan nilai filter yang dipilih."""
    lokasi_list = sorted(df['Lokasi'].unique().tolist()) if 'Lokasi' in df.columns else []
    waktu_list  = sorted(df['Waktu'].unique().tolist())  if 'Waktu'  in df.columns else []
    waktu_options = ["Semua"] + [f"{w:02d}.00 WIB" for w in waktu_list]

    # Hitung min/max tanggal untuk date_input
    if 'Tanggal' in df.columns:
        tgl_series = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
        tgl_min = tgl_series.min().date()
        tgl_max = tgl_series.max().date()
    else:
        import datetime
        tgl_min = tgl_max = datetime.date.today()

    variasi_f = False
    if include_variasi:
        variasi_f = st.checkbox(
            "🗺️ Variasi Antar Wilayah",
            key=f"{prefix}_variasi",
            help="Tampilkan hanya baris di mana pada tanggal dan waktu yang sama terdapat perbedaan kategori cuaca antar lokasi. Filter lain dinonaktifkan saat ini dicentang."
        )

    disabled = variasi_f

    # Baris 1: Lokasi (multiselect) + Waktu + Cuaca (+ Status)
    n_right = 2 + (1 if include_status else 0)
    col_loc, *cols_right = st.columns([2] + [1] * n_right)

    with col_loc:
        loc_f = st.multiselect(
            "📍 Lokasi",
            options=lokasi_list,
            default=lokasi_list if not disabled else [],
            key=f"{prefix}_loc",
            disabled=disabled,
            placeholder="Semua lokasi"
        )

    with cols_right[0]:
        wkt_f = st.selectbox("🕐 Waktu", waktu_options,
                             key=f"{prefix}_wkt", disabled=disabled)
    with cols_right[1]:
        cuaca_options = ["Semua"] + sorted(df[cuaca_col].dropna().unique().tolist()) \
            if (include_cuaca and cuaca_col in df.columns) else ["Semua"]
        cuaca_f = st.selectbox("🌤️ Cuaca", cuaca_options,
                               key=f"{prefix}_cuaca", disabled=disabled)

    status_f = "Semua"
    if include_status and len(cols_right) > 2:
        with cols_right[2]:
            status_f = st.selectbox("✅ Status", ["Semua", "✅ Sesuai", "❌ Tidak Sesuai"],
                                    key=f"{prefix}_status", disabled=disabled)

    # Baris 2: Date range (tanggal awal & akhir)
    col_d1, col_d2, col_pad = st.columns([1, 1, 2])
    with col_d1:
        tgl_awal = st.date_input("📅 Tanggal Awal", value=tgl_min,
                                  min_value=tgl_min, max_value=tgl_max,
                                  key=f"{prefix}_tgl_awal", disabled=disabled)
    with col_d2:
        tgl_akhir = st.date_input("📅 Tanggal Akhir", value=tgl_max,
                                   min_value=tgl_min, max_value=tgl_max,
                                   key=f"{prefix}_tgl_akhir", disabled=disabled)

    # Jika variasi aktif, reset semua filter
    if variasi_f:
        loc_f    = lokasi_list  # semua lokasi
        tgl_awal = tgl_min
        tgl_akhir= tgl_max
        wkt_f    = "Semua"
        cuaca_f  = "Semua"
        status_f = "Semua"

    # Konversi waktu filter
    wkt_val = "Semua"
    if wkt_f != "Semua":
        wkt_val = wkt_f.split('.')[0]

    return loc_f, tgl_awal, tgl_akhir, wkt_val, cuaca_f, variasi_f, status_f


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

actual_labels_in_data = sorted(df['Cuaca'].unique().tolist())

# Gabungkan semua hasil fold ke satu DataFrame untuk Perbandingan
all_fold_df = []
for r in fold_results:
    tmp = r['test_df'].copy()
    tmp['Fold'] = f"Fold {r['fold']}"
    all_fold_df.append(tmp)
all_results_df = pd.concat(all_fold_df, ignore_index=True)
all_results_df = all_results_df.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})


# ════════════════════════════════════════════════════════════════════════════════
# HEADER / DASHBOARD
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

with st.expander("🔍 Filter Data Cuaca", expanded=True):
    dc_loc, dc_tgl_awal, dc_tgl_akhir, dc_wkt, dc_cuaca, dc_variasi, _ = render_section_filters(
        "datacuaca", df, include_cuaca=True, cuaca_col='Cuaca', include_status=False, include_variasi=True
    )

display_df = df.copy()
if dc_loc:
    display_df = display_df[display_df['Lokasi'].isin(dc_loc)]
_dc_tgl = pd.to_datetime(display_df['Tanggal'], format='%d/%m/%Y')
display_df = display_df[(_dc_tgl.dt.date >= dc_tgl_awal) & (_dc_tgl.dt.date <= dc_tgl_akhir)]
if dc_wkt != "Semua":
    display_df = display_df[display_df['Waktu'] == int(dc_wkt)]
if dc_cuaca != "Semua":
    display_df = display_df[display_df['Cuaca'] == dc_cuaca]
if dc_variasi:
    group_cols = ['Tanggal', 'Waktu']
    variasi = df.groupby(group_cols)['Cuaca'].nunique()
    variasi_idx = variasi[variasi > 1].index
    if len(variasi_idx) > 0:
        display_df = df[pd.MultiIndex.from_arrays([df['Tanggal'], df['Waktu']]).isin(variasi_idx)]
        display_df = display_df.sort_values('No').reset_index(drop=True)
    else:
        st.info("Tidak ditemukan variasi cuaca antar wilayah.")

display_df_show = display_df.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
st.dataframe(display_df_show, use_container_width=True, height=300)
st.caption(f"Menampilkan {len(display_df)} dari {len(df)} data.")

# Tabel pivot + pie chart variasi (hanya muncul saat filter dicentang)
if dc_variasi and len(display_df) > 0:
    st.subheader("🗺️ Analisis Variasi Cuaca Antar Wilayah")
    st.caption("Slot tanggal+waktu di mana cuaca aktual berbeda antar lokasi.")

    # Tabel pivot: Tanggal+Waktu × Lokasi, isi = cuaca aktual
    pivot_dc = display_df.pivot_table(
        index=['Tanggal', 'Waktu'], columns='Lokasi', values='Cuaca', aggfunc='first'
    )
    pivot_dc.columns.name = None
    # Sort kronologis: konversi Tanggal ke datetime sementara untuk sorting
    pivot_dc = pivot_dc.reset_index()
    pivot_dc['_sort'] = pd.to_datetime(pivot_dc['Tanggal'], format='%d/%m/%Y')
    pivot_dc = pivot_dc.sort_values(['_sort', 'Waktu']).drop(columns='_sort').reset_index(drop=True)
    st.dataframe(pivot_dc, use_container_width=True, height=300)
    st.caption(f"Total slot waktu dengan variasi: {len(pivot_dc)} slot")

    # Pie chart: slot semua sama vs ada yang berbeda (dari SELURUH data df)
    st.subheader("Proporsi Slot Waktu: Cuaca Seragam vs Bervariasi")
    all_slots = df.groupby(['Tanggal', 'Waktu'])['Cuaca'].nunique()
    total_slots = len(all_slots)
    variasi_slots = int((all_slots > 1).sum())
    sama_slots = total_slots - variasi_slots

    col_pie, col_info = st.columns([1, 1])
    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        sizes = [sama_slots, variasi_slots]
        labels = ['Semua Sama\n(' + str(sama_slots) + ' slot)', 'Ada Variasi\n(' + str(variasi_slots) + ' slot)']
        colors_pie = ['#94a3b8', '#f59e0b']
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9}, pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax_pie.set_title('Distribusi Slot Waktu Cuaca\nSeluruh Dataset', fontsize=10, fontweight='bold')
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close()
    with col_info:
        st.metric("Total Slot Waktu", total_slots)
        st.metric("Slot Cuaca Seragam", sama_slots, f"{sama_slots/total_slots*100:.1f}%")
        st.metric("Slot Ada Variasi", variasi_slots, f"{variasi_slots/total_slots*100:.1f}%")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALISIS DISTRIBUSI DAN PERBANDINGAN CUACA
# ════════════════════════════════════════════════════════════════════════════════
st.header("📈 Analisis Distribusi dan Perbandingan Cuaca")
st.write("Distribusi dan perbandingan kategori cuaca pada data uji per fold dan per wilayah.")

with st.expander("🔍 Filter Visualisasi", expanded=True):
    viz_loc, viz_tgl_awal, viz_tgl_akhir, viz_wkt, viz_cuaca, _, _ = render_section_filters(
        "viz", df, include_cuaca=True, cuaca_col='Cuaca', include_status=False, include_variasi=False
    )

viz_tabs = st.tabs([f"Fold {i+1}" for i in range(5)] + ["Semua Fold"])

for fi, tab in enumerate(viz_tabs[:-1]):
    r = fold_results[fi]
    with tab:
        # Terapkan filter ke test_df fold ini
        fold_test = r['test_df'].copy()
        if viz_loc:
            fold_test = fold_test[fold_test['Lokasi'].isin(viz_loc)]
        _vz_tgl = pd.to_datetime(fold_test['Tanggal'], format='%d/%m/%Y')
        fold_test = fold_test[(_vz_tgl.dt.date >= viz_tgl_awal) & (_vz_tgl.dt.date <= viz_tgl_akhir)]
        if viz_wkt != "Semua":
            fold_test = fold_test[fold_test['Waktu'] == int(viz_wkt)]
        if viz_cuaca != "Semua":
            fold_test = fold_test[fold_test['Cuaca'] == viz_cuaca]

        if len(fold_test) == 0:
            st.warning("Tidak ada data yang sesuai dengan filter yang dipilih untuk fold ini.")
            continue

        actuals_filt = fold_test['Label Aktual'].tolist()
        preds_filt = fold_test['Hasil Prediksi'].tolist()

        actual_counts = pd.Series(actuals_filt).value_counts().reindex(WEATHER_LABELS, fill_value=0)
        pred_counts   = pd.Series(preds_filt).value_counts().reindex(WEATHER_LABELS, fill_value=0)

        st.subheader(f"Aktual vs Prediksi — Fold {r['fold']}")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(WEATHER_LABELS))
        width = 0.35
        bars1 = ax.bar(x - width/2, actual_counts.values, width,
                        label='Aktual', color='#3b82f6', edgecolor='white', linewidth=1.2)
        bars2 = ax.bar(x + width/2, pred_counts.values, width,
                        label='Prediksi', color='#f59e0b', edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' ', '\n') for c in WEATHER_LABELS], fontsize=8)
        ax.set_ylabel('Jumlah', fontsize=9)
        ax.set_title(f'Aktual vs Prediksi per Kategori Cuaca — Fold {r["fold"]}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

with viz_tabs[-1]:
    # Gabungkan semua fold, terapkan filter
    df_semua_fold = pd.concat([r['test_df'] for r in fold_results], ignore_index=True)
    if viz_loc:
        df_semua_fold = df_semua_fold[df_semua_fold['Lokasi'].isin(viz_loc)]
    _vsf_tgl = pd.to_datetime(df_semua_fold['Tanggal'], format='%d/%m/%Y')
    df_semua_fold = df_semua_fold[(_vsf_tgl.dt.date >= viz_tgl_awal) & (_vsf_tgl.dt.date <= viz_tgl_akhir)]
    if viz_wkt != "Semua":
        df_semua_fold = df_semua_fold[df_semua_fold['Waktu'] == int(viz_wkt)]
    if viz_cuaca != "Semua":
        df_semua_fold = df_semua_fold[df_semua_fold['Cuaca'] == viz_cuaca]

    if len(df_semua_fold) == 0:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        st.subheader("Aktual vs Prediksi — Semua Fold")
        actual_all = pd.Series(df_semua_fold['Label Aktual']).value_counts().reindex(WEATHER_LABELS, fill_value=0)
        pred_all   = pd.Series(df_semua_fold['Hasil Prediksi']).value_counts().reindex(WEATHER_LABELS, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(WEATHER_LABELS))
        width = 0.35
        bars1 = ax.bar(x - width/2, actual_all.values, width,
                       label='Aktual', color='#3b82f6', edgecolor='white', linewidth=1.2)
        bars2 = ax.bar(x + width/2, pred_all.values, width,
                       label='Prediksi', color='#f59e0b', edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' ', '\n') for c in WEATHER_LABELS], fontsize=8)
        ax.set_ylabel('Jumlah', fontsize=9)
        ax.set_title('Aktual vs Prediksi per Kategori Cuaca — Semua Fold',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUASI MODEL (K-FOLD)
# ════════════════════════════════════════════════════════════════════════════════
st.header("🔁 Evaluasi Model — K-Fold Cross Validation (k=5)")
st.write(
    "Setiap fold menampilkan hasil evaluasi model pada data uji yang berbeda. "
    "Dataset dibagi menjadi 5 bagian; setiap bagian bergantian menjadi data uji."
)
st.info("💡 Pilih tab fold untuk melihat hasil evaluasi, confusion matrix, dan metrik masing-masing fold.")

with st.expander("🔍 Filter Evaluasi Model", expanded=False):
    ev_loc, ev_tgl_awal, ev_tgl_akhir, ev_wkt, ev_cuaca, ev_variasi, _ = render_section_filters(
        "eval", df, include_cuaca=True, cuaca_col='Cuaca', include_status=False, include_variasi=True
    )

fold_tabs = st.tabs([f"Fold {i+1}" for i in range(5)] + ["Semua Fold"])

for fi, tab in enumerate(fold_tabs[:-1]):
    r = fold_results[fi]
    with tab:
        # Terapkan filter ke data fold
        fold_test_ev = r['test_df'].copy()
        if ev_loc:
            fold_test_ev = fold_test_ev[fold_test_ev['Lokasi'].isin(ev_loc)]
        _ev_tgl = pd.to_datetime(fold_test_ev['Tanggal'], format='%d/%m/%Y')
        fold_test_ev = fold_test_ev[(_ev_tgl.dt.date >= ev_tgl_awal) & (_ev_tgl.dt.date <= ev_tgl_akhir)]
        if ev_wkt != "Semua":
            fold_test_ev = fold_test_ev[fold_test_ev['Waktu'] == int(ev_wkt)]
        if ev_cuaca != "Semua":
            fold_test_ev = fold_test_ev[fold_test_ev['Cuaca'] == ev_cuaca]
        if ev_variasi:
            group_cols = ['Tanggal', 'Waktu']
            # Variasi berbasis Hasil Prediksi: slot waktu di mana model memprediksi berbeda antar lokasi
            variasi_pred = fold_test_ev.groupby(group_cols)['Hasil Prediksi'].nunique()
            variasi_idx = variasi_pred[variasi_pred > 1].index
            if len(variasi_idx) > 0:
                fold_test_ev = fold_test_ev[pd.MultiIndex.from_arrays([fold_test_ev['Tanggal'], fold_test_ev['Waktu']]).isin(variasi_idx)]

        if len(fold_test_ev) == 0:
            st.warning("Tidak ada data yang sesuai dengan filter untuk fold ini.")
            continue

        # Hitung metrik ulang berdasarkan data terfilter
        act_filt = fold_test_ev['Label Aktual'].tolist()
        pred_filt = fold_test_ev['Hasil Prediksi'].tolist()
        acc_filt = sum(a == p for a, p in zip(act_filt, pred_filt)) / len(act_filt) if act_filt else 0

        label_to_idx = {l: i for i, l in enumerate(WEATHER_LABELS)}
        cm_filt = np.zeros((len(WEATHER_LABELS), len(WEATHER_LABELS)), dtype=int)
        for a, p in zip(act_filt, pred_filt):
            if a in label_to_idx and p in label_to_idx:
                cm_filt[label_to_idx[a]][label_to_idx[p]] += 1

        prec_list, rec_list, f1_list_filt = [], [], []
        for i in range(len(WEATHER_LABELS)):
            tp = cm_filt[i, i]
            fp = cm_filt[:, i].sum() - tp
            fn = cm_filt[i, :].sum() - tp
            p_v = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_v = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0
            prec_list.append(p_v)
            rec_list.append(r_v)
            f1_list_filt.append(f1_v)

        st.subheader(f"Fold {r['fold']} — Hasil Evaluasi")
        st.caption(f"Jumlah data uji (setelah filter): {len(act_filt)} sampel")

        col_tbl, col_met = st.columns([3, 1.5])
        with col_tbl:
            tdf = fold_test_ev.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
            preview = tdf[[
                'Tanggal', 'Waktu', 'Lokasi',
                'Suhu (°C)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV',
                'Label Aktual', 'Hasil Prediksi', 'Keterangan'
            ]].reset_index(drop=True)
            st.dataframe(preview, use_container_width=True, height=280)

        with col_met:
            st.metric("Akurasi",  f"{acc_filt*100:.2f}%")
            st.metric("Presisi",  f"{np.mean(prec_list)*100:.2f}%")
            st.metric("Recall",   f"{np.mean(rec_list)*100:.2f}%")
            st.metric("F1-Score", f"{np.mean(f1_list_filt)*100:.2f}%")

        st.subheader("Confusion Matrix")
        cmap = LinearSegmentedColormap.from_list('blues', ['#f0f9ff', '#0369a1'])
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(cm_filt, cmap=cmap, aspect='auto')

        ax.set_xticks(range(len(WEATHER_LABELS)))
        ax.set_yticks(range(len(WEATHER_LABELS)))
        short = [c.replace(' ', '\n') for c in WEATHER_LABELS]
        ax.set_xticklabels(short, fontsize=8)
        ax.set_yticklabels(WEATHER_LABELS, fontsize=8)
        ax.set_xlabel('Prediksi', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_ylabel('Aktual', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title(f'Confusion Matrix — Fold {r["fold"]}', fontsize=11, fontweight='bold', pad=12)

        max_val = cm_filt.max() if cm_filt.max() > 0 else 1
        for i in range(len(WEATHER_LABELS)):
            for j in range(len(WEATHER_LABELS)):
                val = cm_filt[i, j]
                color = 'white' if val > max_val * 0.6 else '#0f172a'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Visualisasi variasi prediksi (hanya muncul saat filter variasi dicentang)
        if ev_variasi and len(fold_test_ev) > 0:
            st.subheader(f"🗺️ Analisis Variasi Prediksi Antar Wilayah — Fold {r['fold']}")
            st.caption("Slot tanggal+waktu di mana model memprediksi cuaca berbeda antar lokasi.")

            # Tabel pivot prediksi: Tanggal+Waktu × Lokasi, isi = Hasil Prediksi, sort kronologis
            pivot_pred = fold_test_ev.pivot_table(
                index=['Tanggal', 'Waktu'], columns='Lokasi',
                values='Hasil Prediksi', aggfunc='first'
            )
            pivot_pred.columns.name = None
            pivot_pred = pivot_pred.reset_index()
            pivot_pred['_sort'] = pd.to_datetime(pivot_pred['Tanggal'], format='%d/%m/%Y')
            pivot_pred = pivot_pred.sort_values(['_sort', 'Waktu']).drop(columns='_sort').reset_index(drop=True)
            st.dataframe(pivot_pred, use_container_width=True, height=280)
            st.caption(f"Total slot waktu prediksi bervariasi: {len(pivot_pred)} slot")

            # Pie chart: slot prediksi semua sama vs ada yang berbeda (dari fold ini)
            st.subheader(f"Proporsi Slot Waktu: Prediksi Seragam vs Bervariasi — Fold {r['fold']}")
            fold_all_slots = r['test_df'].groupby(['Tanggal', 'Waktu'])['Hasil Prediksi'].nunique()
            total_slots_ev = len(fold_all_slots)
            variasi_slots_ev = int((fold_all_slots > 1).sum())
            sama_slots_ev = total_slots_ev - variasi_slots_ev

            col_pie_ev, col_info_ev = st.columns([1, 1])
            with col_pie_ev:
                fig_pie2, ax_pie2 = plt.subplots(figsize=(5, 4))
                sizes_ev = [sama_slots_ev, variasi_slots_ev]
                labels_ev = ['Prediksi Sama\n(' + str(sama_slots_ev) + ' slot)', 'Ada Variasi\n(' + str(variasi_slots_ev) + ' slot)']
                colors_ev = ['#94a3b8', '#3b82f6']
                wedges2, texts2, autotexts2 = ax_pie2.pie(
                    sizes_ev, labels=labels_ev, colors=colors_ev,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 9}, pctdistance=0.75
                )
                for at2 in autotexts2:
                    at2.set_fontweight('bold')
                ax_pie2.set_title('Distribusi Slot Waktu Prediksi\nFold ' + str(r['fold']),
                                  fontsize=10, fontweight='bold')
                fig_pie2.tight_layout()
                st.pyplot(fig_pie2, use_container_width=True)
                plt.close()
            with col_info_ev:
                st.metric("Total Slot Waktu", total_slots_ev)
                st.metric("Prediksi Seragam", sama_slots_ev, f"{sama_slots_ev/total_slots_ev*100:.1f}%" if total_slots_ev > 0 else "0%")
                st.metric("Ada Variasi Prediksi", variasi_slots_ev, f"{variasi_slots_ev/total_slots_ev*100:.1f}%" if total_slots_ev > 0 else "0%")

with fold_tabs[-1]:
    # Gabungkan semua fold, terapkan filter yang sama
    all_ev_df = pd.concat([r['test_df'] for r in fold_results], ignore_index=True)
    if ev_loc:
        all_ev_df = all_ev_df[all_ev_df['Lokasi'].isin(ev_loc)]
    _evs_tgl = pd.to_datetime(all_ev_df['Tanggal'], format='%d/%m/%Y')
    all_ev_df = all_ev_df[(_evs_tgl.dt.date >= ev_tgl_awal) & (_evs_tgl.dt.date <= ev_tgl_akhir)]
    if ev_wkt != "Semua":
        all_ev_df = all_ev_df[all_ev_df['Waktu'] == int(ev_wkt)]
    if ev_cuaca != "Semua":
        all_ev_df = all_ev_df[all_ev_df['Cuaca'] == ev_cuaca]
    if ev_variasi:
        group_cols = ['Tanggal', 'Waktu']
        variasi_pred_all = all_ev_df.groupby(group_cols)['Hasil Prediksi'].nunique()
        variasi_idx_all = variasi_pred_all[variasi_pred_all > 1].index
        if len(variasi_idx_all) > 0:
            all_ev_df = all_ev_df[pd.MultiIndex.from_arrays(
                [all_ev_df['Tanggal'], all_ev_df['Waktu']]).isin(variasi_idx_all)]

    if len(all_ev_df) == 0:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        act_all = all_ev_df['Label Aktual'].tolist()
        pred_all = all_ev_df['Hasil Prediksi'].tolist()
        acc_all = sum(a == p for a, p in zip(act_all, pred_all)) / len(act_all)

        label_to_idx = {l: i for i, l in enumerate(WEATHER_LABELS)}
        cm_all_ev = np.zeros((len(WEATHER_LABELS), len(WEATHER_LABELS)), dtype=int)
        for a, p in zip(act_all, pred_all):
            if a in label_to_idx and p in label_to_idx:
                cm_all_ev[label_to_idx[a]][label_to_idx[p]] += 1

        prec_all, rec_all, f1_all = [], [], []
        for i in range(len(WEATHER_LABELS)):
            tp = cm_all_ev[i, i]
            fp = cm_all_ev[:, i].sum() - tp
            fn = cm_all_ev[i, :].sum() - tp
            p_v = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_v = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0
            prec_all.append(p_v)
            rec_all.append(r_v)
            f1_all.append(f1_v)

        st.subheader("Semua Fold — Hasil Evaluasi Gabungan")
        st.caption(f"Jumlah data uji (setelah filter): {len(act_all)} sampel")

        col_tbl_all, col_met_all = st.columns([3, 1.5])
        with col_tbl_all:
            tdf_all = all_ev_df.rename(columns={'Suhu (celcius)': 'Suhu (°C)'})
            preview_all = tdf_all[[
                'Tanggal', 'Waktu', 'Lokasi',
                'Suhu (°C)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV',
                'Label Aktual', 'Hasil Prediksi', 'Keterangan'
            ]].reset_index(drop=True)
            st.dataframe(preview_all, use_container_width=True, height=280)
        with col_met_all:
            st.metric("Akurasi",  f"{acc_all*100:.2f}%")
            st.metric("Presisi",  f"{np.mean(prec_all)*100:.2f}%")
            st.metric("Recall",   f"{np.mean(rec_all)*100:.2f}%")
            st.metric("F1-Score", f"{np.mean(f1_all)*100:.2f}%")

        st.subheader("Confusion Matrix — Semua Fold")
        cmap_all = LinearSegmentedColormap.from_list('blues', ['#f0f9ff', '#0369a1'])
        fig_all, ax_all = plt.subplots(figsize=(8, 5.5))
        im_all = ax_all.imshow(cm_all_ev, cmap=cmap_all, aspect='auto')
        ax_all.set_xticks(range(len(WEATHER_LABELS)))
        ax_all.set_yticks(range(len(WEATHER_LABELS)))
        short_all = [c.replace(' ', '\n') for c in WEATHER_LABELS]
        ax_all.set_xticklabels(short_all, fontsize=8)
        ax_all.set_yticklabels(WEATHER_LABELS, fontsize=8)
        ax_all.set_xlabel('Prediksi', fontsize=10, fontweight='bold', labelpad=8)
        ax_all.set_ylabel('Aktual', fontsize=10, fontweight='bold', labelpad=8)
        ax_all.set_title('Confusion Matrix — Semua Fold', fontsize=11, fontweight='bold', pad=12)
        max_val_all = cm_all_ev.max() if cm_all_ev.max() > 0 else 1
        for i in range(len(WEATHER_LABELS)):
            for j in range(len(WEATHER_LABELS)):
                val = cm_all_ev[i, j]
                color = 'white' if val > max_val_all * 0.6 else '#0f172a'
                ax_all.text(j, i, str(val), ha='center', va='center',
                            fontsize=9, fontweight='bold', color=color)
        fig_all.colorbar(im_all, ax=ax_all, fraction=0.04, pad=0.03)
        fig_all.tight_layout()
        st.pyplot(fig_all, use_container_width=True)
        plt.close()

        if ev_variasi:
            st.subheader("🗺️ Analisis Variasi Prediksi Antar Wilayah — Semua Fold")
            st.caption("Slot tanggal+waktu di mana model memprediksi cuaca berbeda antar lokasi.")
            pivot_all = all_ev_df.pivot_table(
                index=['Tanggal', 'Waktu'], columns='Lokasi',
                values='Hasil Prediksi', aggfunc='first'
            )
            pivot_all.columns.name = None
            pivot_all = pivot_all.reset_index()
            pivot_all['_sort'] = pd.to_datetime(pivot_all['Tanggal'], format='%d/%m/%Y')
            pivot_all = pivot_all.sort_values(['_sort', 'Waktu']).drop(columns='_sort').reset_index(drop=True)
            st.dataframe(pivot_all, use_container_width=True, height=280)
            st.caption(f"Total slot waktu prediksi bervariasi: {len(pivot_all)} slot")

            all_slots_all = pd.concat([r['test_df'] for r in fold_results]).groupby(
                ['Tanggal', 'Waktu'])['Hasil Prediksi'].nunique()
            total_slots_all = len(all_slots_all)
            variasi_slots_all = int((all_slots_all > 1).sum())
            sama_slots_all = total_slots_all - variasi_slots_all

            col_pie_all, col_info_all = st.columns([1, 1])
            with col_pie_all:
                fig_pie_all, ax_pie_all = plt.subplots(figsize=(5, 4))
                sizes_all = [sama_slots_all, variasi_slots_all]
                labels_all = ['Prediksi Sama\n(' + str(sama_slots_all) + ' slot)', 'Ada Variasi\n(' + str(variasi_slots_all) + ' slot)']
                wedges_all, _, autotexts_all = ax_pie_all.pie(
                    sizes_all, labels=labels_all, colors=['#94a3b8', '#3b82f6'],
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 9}, pctdistance=0.75
                )
                for at_all in autotexts_all:
                    at_all.set_fontweight('bold')
                ax_pie_all.set_title('Distribusi Slot Waktu Prediksi\nSemua Fold',
                                     fontsize=10, fontweight='bold')
                fig_pie_all.tight_layout()
                st.pyplot(fig_pie_all, use_container_width=True)
                plt.close()
            with col_info_all:
                st.metric("Total Slot Waktu", total_slots_all)
                st.metric("Prediksi Seragam", sama_slots_all,
                          f"{sama_slots_all/total_slots_all*100:.1f}%" if total_slots_all > 0 else "0%")
                st.metric("Ada Variasi Prediksi", variasi_slots_all,
                          f"{variasi_slots_all/total_slots_all*100:.1f}%" if total_slots_all > 0 else "0%")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RINGKASAN EVALUASI
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
# SECTION 5 — PERBANDINGAN HASIL KLASIFIKASI (SEMUA FOLD TERINTEGRASI)
# ════════════════════════════════════════════════════════════════════════════════
st.header("⚖️ Perbandingan Hasil Klasifikasi")
st.write(
    "Perbandingan antara label aktual (data Weather Channel) "
    "dengan hasil prediksi model Random Forest. Data dari seluruh fold digabungkan "
    "dan dapat difilter per fold, lokasi, tanggal, waktu, kategori cuaca, maupun status kesesuaian."
)

with st.expander("🔍 Filter Perbandingan Hasil", expanded=True):
    # Filter fold tambahan
    fold_options = ["Semua Fold"] + [f"Fold {i+1}" for i in range(5)]
    pb_fold = st.selectbox("📂 Fold", fold_options, key="pb_fold")
    pb_loc, pb_tgl_awal, pb_tgl_akhir, pb_wkt, pb_cuaca, _, pb_status = render_section_filters(
        "perbandingan", all_results_df, include_cuaca=True,
        cuaca_col='Label Aktual', include_status=True, include_variasi=False
    )

# Terapkan filter pada all_results_df
comp_df = all_results_df.copy()

if pb_fold != "Semua Fold":
    comp_df = comp_df[comp_df['Fold'] == pb_fold]
if pb_loc:
    comp_df = comp_df[comp_df['Lokasi'].isin(pb_loc)]
comp_df_tgl = pd.to_datetime(comp_df['Tanggal'], format='%d/%m/%Y')
comp_df = comp_df[(comp_df_tgl.dt.date >= pb_tgl_awal) & (comp_df_tgl.dt.date <= pb_tgl_akhir)]
if pb_wkt != "Semua":
    comp_df = comp_df[comp_df['Waktu'] == int(pb_wkt)]
if pb_cuaca != "Semua":
    comp_df = comp_df[comp_df['Label Aktual'] == pb_cuaca]
if pb_status != "Semua":
    comp_df = comp_df[comp_df['Keterangan'] == pb_status]

# Ringkasan metrik
total_c = len(comp_df)
match_c = (comp_df['Keterangan'] == '✅ Sesuai').sum()
mismatch_c = (comp_df['Keterangan'] == '❌ Tidak Sesuai').sum()

col_a, col_b, col_c = st.columns(3)
col_a.metric("✅ Sesuai", match_c, f"{match_c/total_c*100:.1f}% dari data" if total_c > 0 else "0%")
col_b.metric("❌ Tidak Sesuai", mismatch_c, f"{mismatch_c/total_c*100:.1f}% dari data" if total_c > 0 else "0%")
col_c.metric("Total Data", total_c, "baris terfilter")

# Tabel hasil klasifikasi terfilter
comp_show = comp_df[[
    'Fold', 'Tanggal', 'Waktu', 'Lokasi',
    'Suhu (°C)', 'Kelembapan (%)', 'Angin (km/h)', 'Intensitas UV',
    'Label Aktual', 'Hasil Prediksi', 'Keterangan'
]].reset_index(drop=True)
st.dataframe(comp_show, use_container_width=True, height=350)
st.caption(f"Menampilkan {len(comp_df)} baris hasil klasifikasi.")

# ── Visualisasi Aktual vs Prediksi ──────────────────────────────────────────
if len(comp_df) > 0:
    st.subheader("📊 Heatmap Distribusi Aktual vs Prediksi")

    # Heatmap Aktual vs Prediksi (confusion-style)
    if True:
        label_to_idx = {l: i for i, l in enumerate(WEATHER_LABELS)}
        cm_all = np.zeros((len(WEATHER_LABELS), len(WEATHER_LABELS)), dtype=int)
        for a, p in zip(comp_df['Label Aktual'], comp_df['Hasil Prediksi']):
            if a in label_to_idx and p in label_to_idx:
                cm_all[label_to_idx[a]][label_to_idx[p]] += 1

        cmap = LinearSegmentedColormap.from_list('heatmap', ['#f0f9ff', '#1e40af'])
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm_all, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(WEATHER_LABELS)))
        ax.set_yticks(range(len(WEATHER_LABELS)))
        short = [c.replace(' ', '\n') for c in WEATHER_LABELS]
        ax.set_xticklabels(short, fontsize=7)
        ax.set_yticklabels(WEATHER_LABELS, fontsize=7)
        ax.set_xlabel('Prediksi', fontsize=9, fontweight='bold')
        ax.set_ylabel('Aktual', fontsize=9, fontweight='bold')
        ax.set_title('Heatmap Distribusi Aktual vs Prediksi', fontsize=10, fontweight='bold')
        max_val = cm_all.max() if cm_all.max() > 0 else 1
        for i in range(len(WEATHER_LABELS)):
            for j in range(len(WEATHER_LABELS)):
                val = cm_all[i, j]
                color = 'white' if val > max_val * 0.6 else '#0f172a'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=7, fontweight='bold', color=color)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


else:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")

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
            min_value=0.0, max_value=50.0,
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
            min_value=0.0, max_value=410.0,
            value=float(df['Angin (km/h)'].median()),
            step=0.1,
            help="Kecepatan angin dalam km/jam"
        )
        input_uv = st.number_input(
            "Intensitas UV",
            min_value=0.0, max_value=12.0,
            value=float(df['Intensitas UV'].median()),
            step=0.1,
            help="Indeks UV (0–11) (pilih 12 jika 11+)"
        )

    submitted = st.form_submit_button("🔍 Klasifikasikan", use_container_width=True, type="primary")

if submitted:
    try:
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