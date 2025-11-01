import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dalex as dx

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Additional libraries
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind as stats_ttest_ind 
import warnings
warnings.filterwarnings('ignore')
import os

# Page configuration
st.set_page_config(
    page_title="Pemodelan Prediksi Diagnosis Penyakit Alzheimer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F77B4;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🧠 Pemodelan Klasifikasi Prediksi Penyakit Alzheimer")

st.markdown("---")

# Sidebar Configuration
st.sidebar.title("⚙️ Eksplore Data")
st.sidebar.markdown("---")

# Ganti dengan URL RAW GitHub yang BENAR
DEFAULT_FILE_PATH = "https://raw.githubusercontent.com/Herdvair/Final_Project_Streamlit/main/alzheimers_disease_data.csv"

# Load Data Function
@st.cache_data
def load_data(uploaded=None, default_path=None):
    """
    Load dataset: Prioritizes uploaded file, then default URL file.
    """
    try:
        #  File yang Diunggah di Dashboard
        if uploaded is not None:
            return pd.read_csv(uploaded)

        #  File Default (URL RAW)
        elif default_path is not None:
            # pd.read_csv bisa membaca URL secara langsung
            return pd.read_csv(default_path)
        
        #  Tidak ada data
        else:
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
        
def data_understanding(df_eda: pd.DataFrame):
    """Display comprehensive data understanding for EDA."""
    
    # Perhitungan Metrik
    total_rows = len(df_eda)
    total_duplicates = total_rows - len(df_eda.drop_duplicates())
    duplicate_ratio = total_duplicates / total_rows if total_rows > 0 else 0
    total_missing = df_eda.isna().sum().sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Baris & Kolom", f"{total_rows} & {df_eda.shape[1]}")
        st.metric("Total Nilai yang Hilang", f"{total_missing}") 
        st.metric("Rasio Duplikasi Data", f"{duplicate_ratio:.2%}")
    
    with col2:
        st.metric("Fitur Numerik", f"{len(df_eda.select_dtypes(include=np.number).columns)}")
        st.metric("Fitur Non-Numerik", f"{len(df_eda.select_dtypes(exclude=np.number).columns)}")


# -------------------------------------------------------------------
# 2. INISIALISASI SESSION STATE
# -------------------------------------------------------------------

default_state = {
    'df': None,
    'df_cleaned': None,
    'models_trained': False
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------------------------------------------
# 3. LOAD DATA
# -------------------------------------------------------------------

df = load_data(uploaded=DEFAULT_FILE_PATH)

# Simpan ke session state
if df is not None:
    st.session_state.df = df

# --- 1. SETUP NAVIGASI SIDEBAR ---

# Daftar opsi navigasi
tab_list = [
    "Eksplorasi Analisis Data",
    "Preprocessing & VIF",
    "Model Dasar",
    "LightGBM",
    "Evaluasi Model",
    "Fitur Penting",
    "Kesimpulan & Rekomendasi"
]

# Navigasi utama di Sidebar
selected_tab = st.sidebar.radio(
    "Menu",
    tab_list,
    index=0 # Set EDA sebagai tab default
)

st.markdown("""
<style>
/* HILANGKAN styling stTabs utama, tetapi pertahankan styling sub-tabs dan multiselect */

/* Style untuk sub-tabs EDA */
[data-testid="stVerticalBlock"] > [data-testid="stTabs"] [data-baseweb="tab"] {
    color: black; /* Warna teks sub-tab diatur ke Hitam */
    font-weight: bold;
}

/* Mengatur tinggi multiselect */
.stMultiSelect div[data-baseweb="select"] div[role="listbox"] {
    max-height: 200px; /* Menampilkan sekitar 8 opsi */
}
</style>
""", unsafe_allow_html=True)
# --- AKHIR STYLING ---


# --- 3. IMPLEMENTASI KONTEN BERDASARKAN SIDEBAR ---

## 🎯 TAB 1: EXPLORATORY DATA ANALYSIS (PERBAIKAN
# ==============================================================================
if selected_tab == "Eksplorasi Analisis Data":
    
    st.header(selected_tab)
    
    # Ambil data dari session state
    df = st.session_state.get('df')
    df_eda = st.session_state.get('df_cleaned') if st.session_state.get('df_cleaned') is not None else df
    
    if df_eda is None:
        st.error("Data tidak tersedia untuk EDA. Mohon unggah atau muat data terlebih dahulu.")
    
    else:
        # ===================================================================
        # DROP KOLOM SEBELUM DATA UNDERSTANDING
        # ===================================================================
        cols_to_drop = [
            'Ethnicity', 'DietQuality', 'PatientID', 'SystolicBP', 'DiastolicBP',
            'DoctorInCharge', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
            'CholesterolTriglycerides', 'Smoking', 'Forgetfulness'
        ]
        
        # Filter kolom yang ada
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_eda.columns]
        
        if existing_cols_to_drop:
            df_eda = df_eda.drop(columns=existing_cols_to_drop, errors='ignore')
            st.info(f"Kolom yang di-drop: {', '.join(existing_cols_to_drop)}")
            
            # PENTING: Simpan data yang sudah bersih ke session state
            # untuk digunakan di tab Preprocessing dan Modeling
            st.session_state.df_cleaned = df_eda.copy()
        
        # ===================================================================
        # DATA UNDERSTANDING (SETELAH DROP KOLOM)
        # ===================================================================
        st.subheader("Memahami Data")
        data_understanding(df_eda)
        st.markdown("---")
        # ===================================================================
        
        # --- FITUR FILTER GLOBAL (Non-AD vs Alzheimer) ---
        diagnosis_filter = st.selectbox(
            "Filter Data Berdasarkan Diagnosis:",
            options=["Semua", "Non-AD (Diagnosis=0)", "Alzheimer (Diagnosis=1)"],
            index=0,
            help="Filter ini hanya bisa diterapkan di semua visualisasi pada tab EDA ini."
        )
        st.markdown("---")
        
        # Terapkan Filter Global
        if diagnosis_filter == "Non-AD (Diagnosis=0)":
            filtered_df_eda = df_eda[df_eda['Diagnosis'] == 0]
        elif diagnosis_filter == "Alzheimer (Diagnosis=1)":
            filtered_df_eda = df_eda[df_eda['Diagnosis'] == 1]
        else:
            filtered_df_eda = df_eda
            
        df_visual = filtered_df_eda
        
        # Kolom Numerik
        numerical_cols = df_visual.select_dtypes(include=['int64', 'float64']).columns
        if 'Diagnosis' in numerical_cols:
            numerical_cols = numerical_cols.drop('Diagnosis')
        # --- AKHIR FILTER GLOBAL ---
        
        # Sub-tabs for different analyses
        eda_tabs = st.tabs(["Distribusi Diagnosis", "Analisis Korelasi", "Statistical Test Chi Square", "Statistical Test T-test"])
        
        # KONTEN eda_tabs[0]: Distribution
        with eda_tabs[0]:
            st.markdown("### Distribusi Fitur berdasarkan Diagnosis")
            
            cols_to_plot = st.multiselect(
                "Pilih Fitur Numerik",
                numerical_cols.tolist(),
                default=numerical_cols[:8].tolist() if len(numerical_cols) > 8 else numerical_cols.tolist(),
                key='eda_multiselect_dist'
            )
            
            if cols_to_plot:
                N_COLS = 4
                N_ROWS = int(np.ceil(len(cols_to_plot) / N_COLS))

                fig = make_subplots(rows=N_ROWS, cols=N_COLS, subplot_titles=cols_to_plot)

                active_diagnoses = df_visual['Diagnosis'].unique()

                for index, col in enumerate(cols_to_plot):
                    row = index // N_COLS + 1
                    col_idx = index % N_COLS + 1
                    
                    for diagnosis in active_diagnoses:
                        data = df_visual[df_visual['Diagnosis'] == diagnosis][col]
                        
                        fig.add_trace(
                            go.Histogram(
                                x=data,
                                name=f"{'Alzheimer' if diagnosis == 1 else 'Non-AD'}",
                                opacity=0.7,
                                showlegend=(index == 0)
                            ),
                            row=row, col=col_idx
                        )

                fig.update_layout(
                    height=250 * N_ROWS,
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Boxplots
                st.markdown("### Boxplot berdasarkan Diagnosis")
                
                fig2 = make_subplots(rows=N_ROWS, cols=N_COLS, subplot_titles=cols_to_plot)
                
                for index, col in enumerate(cols_to_plot):
                    row = index // N_COLS + 1
                    col_idx = index % N_COLS + 1
                    
                    for diagnosis in active_diagnoses:
                        data = df_visual[df_visual['Diagnosis'] == diagnosis][col]
                        fig2.add_trace(
                            go.Box(
                                y=data,
                                name=f"{'Alzheimer' if diagnosis == 1 else 'Non-AD'}",
                                showlegend=(index == 0)
                            ),
                            row=row, col=col_idx
                        )
                
                fig2.update_layout(
                    height=250 * N_ROWS,
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # KONTEN eda_tabs[1]: Correlation Analysis
        with eda_tabs[1]:
            st.markdown("### Analisis Korelasi")
            
            corr_features = st.multiselect(
                "Pilih Fitur Numerik",
                numerical_cols.tolist(),
                default=numerical_cols[:15].tolist() if len(numerical_cols) > 15 else numerical_cols.tolist(),
                key='eda_multiselect_corr'
            )
            
            if corr_features and 'Diagnosis' in df_visual.columns:
                corr_matrix = df_visual[corr_features + ['Diagnosis']].corr()
                
                # Heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 8},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    height=700,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation with target
                st.markdown("### Korelasi dengan Diagnosis")
                
                if 'Diagnosis' in corr_matrix.columns and 'Diagnosis' in corr_matrix.index:
                    target_corr = corr_matrix['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)
                    
                    fig2 = go.Figure(go.Bar(
                        x=target_corr.values,
                        y=target_corr.index,
                        orientation='h',
                        marker_color=['green' if x > 0 else 'red' for x in target_corr.values]
                    ))
                    
                    fig2.update_layout(
                        xaxis_title="Koefisien Korelasi",
                        height=500
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                     st.info("Korelasi dengan Diagnosis tidak tersedia karena hanya satu kelas Diagnosis yang dipilih.")
                     
        # KONTEN eda_tabs[2]: Statistical Tests by Chi Square
        with eda_tabs[2]:
            st.markdown("### Uji Statistik Menggunakan Chi-Square")
            
            categorical_cols = ['BehavioralProblems', 'MemoryComplaints', 'Confusion', 
                              'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                              'Gender', 'FamilyHistoryAlzheimers',
                              'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
            
            categorical_cols = [col for col in categorical_cols if col in df_visual.columns]
            
            if 'Diagnosis' in df_visual.columns and len(df_visual['Diagnosis'].unique()) > 1:
                if categorical_cols:
                    chi_results = []
                    
                    for col in categorical_cols:
                        table = pd.crosstab(df_visual[col], df_visual['Diagnosis'])
                        
                        if table.shape[0] > 1 and table.shape[1] > 1:
                            chi2, p_value, dof, expected = chi2_contingency(table)
                            
                            chi_results.append({
                                'Variable': col,
                                'Chi2 Statistic': chi2,
                                'P-Value': p_value,
                                'Signifikansi': 'Signifikan' if p_value < 0.05 else 'Tidak Signifikan'
                            })
                    
                    if chi_results:
                        chi_df = pd.DataFrame(chi_results).sort_values('P-Value')
                        
                        st.dataframe(
                            chi_df.style.format({
                                'Chi2 Statistic': '{:.3f}',
                                'P-Value': '{:.4f}'
                            }).applymap(
                                lambda x: 'background-color: #0000FF' if x == 'Signifikan' else 'background-color: #FF0000',
                                subset=['Signifikansi']
                            ),
                            use_container_width=True
                        )
                        
                else:
                    st.warning("Tidak ada kolom yang tersedia dalam data yang difilter.")
            else:
                st.info("Uji Statistik Chi-Square tidak dapat dilakukan karena hanya satu kelas Diagnosis (Non-AD atau Alzheimer) yang dipilih.")
        
        # KONTEN eda_tabs[3]: Statistical Tests by T-Test
        with eda_tabs[3]:
            st.markdown("### Uji Statistik Menggunakan T-Test")
            
            ttest_cols = ['MMSE', 'ADL', 'FunctionalAssessment']
            ttest_cols = [col for col in ttest_cols if col in df_visual.columns]
            
            if 'Diagnosis' in df_visual.columns and len(df_visual['Diagnosis'].unique()) > 1:
                if ttest_cols:
                    ttest_results = []
                    
                    for col in ttest_cols:
                        group_non_ad = df_visual[df_visual['Diagnosis'] == 0][col].dropna()
                        group_alzheimer = df_visual[df_visual['Diagnosis'] == 1][col].dropna()
                        
                        if len(group_non_ad) > 1 and len(group_alzheimer) > 1:
                            t_stat, p_value = stats_ttest_ind(
                                group_non_ad, group_alzheimer, equal_var=False
                            )
                            
                            ttest_results.append({
                                'Variable': col,
                                't-statistic': t_stat,
                                'p-value': p_value,
                                'Signifikansi': 'Signifikan' if p_value < 0.05 else 'Tidak Signifikan'
                            })
                    
                    if ttest_results:
                        df_ttest = pd.DataFrame(ttest_results).sort_values('p-value')
                        
                        st.dataframe(
                            df_ttest.style.format({'t-statistic': '{:.3f}', 'p-value': '{:.4f}'})
                            .applymap(
                                lambda x: 'background-color: #0000FF' if x == 'Signifikan' else 'background-color: #FF0000',
                                subset=['Signifikansi']
                            ),
                            use_container_width=True
                        )
                else:
                    st.warning("Kolom T-Test ('MMSE', 'ADL', 'FunctionalAssessment') tidak ditemukan dalam data.")
            else:
                st.info("Uji Statistik T-Test tidak dapat dilakukan karena hanya satu kelas Diagnosis (Non-AD atau Alzheimer) yang dipilih.")


# =================================================================
# 🎯 TAB 2: PREPROCESSING & VIF 
# =================================================================
elif selected_tab == "Preprocessing & VIF":
    
    st.header(selected_tab)
    
    # GUNAKAN DATA YANG SUDAH DIBERSIHKAN DARI EDA
    df_processed = st.session_state.get('df_cleaned')

    if df_processed is None:
         st.error("Data tidak tersedia. Mohon lakukan EDA terlebih dahulu untuk membersihkan data.")
    else:
        st.success(f"Lakukan Split Data.")
        
        # Train-Test Split
        st.markdown("### Train-Test Split")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.2, 0.4, 0.3, 0.05)
        with col2:
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        if st.button("Split Data"):
            # Prepare features and target
            try:
                feature = df_processed.drop(['Diagnosis'], axis=1)
                target = df_processed['Diagnosis']
                
                # Validasi: Cek kolom non-numerik
                non_numeric = feature.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
                if non_numeric:
                    st.error(f"❌ Masih ada kolom non-numerik: {', '.join(non_numeric)}")
                    st.warning("Silakan drop atau encode kolom tersebut di EDA!")
                else:
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        feature, target, 
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=target
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success(f"Split Data Sukses! ({100*test_size:.0f}% Test Set)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Set", f"{len(X_train)} sample")
                    with col2:
                        st.metric("Test Set", f"{len(X_test)} sample")
                        st.metric("Non-AD (Test)", (y_test == 0).sum())
                        st.metric("Alzheimer (Test)", (y_test == 1).sum())
                    
            except KeyError:
                st.error("❌ Kolom 'Diagnosis' tidak ditemukan. Pastikan data sudah bersih dari EDA.")
        
        st.markdown("---")
        
        # VIF Check
        st.markdown("### Variance Inflation Factor (VIF) - Multicollinearity Check")
        
        if 'X_train' in st.session_state:
            if st.button("Hitung VIF"):
                with st.spinner("Sedang menghitung VIF..."):
                    X_train_num = st.session_state.X_train.select_dtypes(include=['int64', 'float64'])
                    
                    if X_train_num.empty or X_train_num.isna().all().all():
                        st.warning("Tidak ada fitur numerik untuk dihitung VIF.")
                    else:
                        # Membersihkan NaN
                        X_vif = add_constant(X_train_num.fillna(X_train_num.mean())) 
                        
                        vif_data = pd.DataFrame()
                        vif_data["Feature"] = X_vif.columns
                        vif_data["VIF"] = [vif(X_vif.values, i) for i in range(X_vif.shape[1])]
                        vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values('VIF', ascending=False)
                        
                        st.session_state.vif_results = vif_data
                        
                        st.markdown("#### Hasil VIF")
                        
                        def color_vif(val):
                            if val < 3:
                                color = 'background-color: #FFFFFF'
                            elif val < 10:
                                color = 'background-color: #0000FF'
                            else:
                                color = 'background-color: #FF0000'
                            return color
                        
                        st.dataframe(
                            vif_data.style.format({'VIF': '{:.2f}'})
                                        .applymap(color_vif, subset=['VIF']),
                            use_container_width=True
                        )
                        
                        # Summary
                        high_vif = vif_data[vif_data['VIF'] > 10]
                        moderate_vif = vif_data[(vif_data['VIF'] >= 3) & (vif_data['VIF'] <= 10)]
                        low_vif = vif_data[vif_data['VIF'] < 3]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Low VIF (< 3)", len(low_vif), help="No multicollinearity")
                        with col2:
                            st.metric("Moderate VIF (3-10)", len(moderate_vif), help="Moderate multicollinearity")
                        with col3:
                            st.metric("High VIF (> 10)", len(high_vif), help="High multicollinearity")
                        
                        if len(high_vif) > 0:
                            st.warning(f"Ditemukan {len(high_vif)} fitur dengan multikolinearitas tinggi (VIF > 10)")
                            st.write("Fitur dengan VIF tinggi:", high_vif['Feature'].tolist())
                        else:
                            st.success("Semua fitur memiliki VIF yang dapat diterima.")

        else:
            st.warning("⚠️ Mohon split data terlebih dahulu!")


# ===========================================================================
# 🎯 TAB 3: MODEL DASAR 
# ===========================================================================
elif selected_tab == "Model Dasar":
    st.header(selected_tab)
    
    if 'X_train' not in st.session_state:
        st.warning("Selesaikan preprocessing data terlebih dahulu!")
    else:
        # Define columns to scale
        cols_to_scale = ['Age', 'EducationLevel', 'BMI', 'AlcoholConsumption', 
                        'PhysicalActivity', 'SleepQuality', 'MMSE', 
                        'FunctionalAssessment', 'ADL']
        
        # Filter cols_to_scale to only include existing columns
        cols_to_scale = [col for col in cols_to_scale if col in st.session_state.X_train.columns]
        
        if st.button("🚀 Latih Seluruh Model Dasar"):
            with st.spinner("Tunggu Sebentar..."):
                
                # Create scaler
                scaler = ColumnTransformer(
                    transformers=[
                        ('scale', MinMaxScaler(), cols_to_scale)
                    ], 
                    remainder='passthrough'
                )
                
                # Define models
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
                    'LightGBM': LGBMClassifier(random_state=42),
                    'KNN': KNeighborsClassifier(),
                    'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
                    'MLP': MLPClassifier(random_state=1000, hidden_layer_sizes=(8,), solver='sgd')
                }
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, (name, clf) in enumerate(models.items()):
                    progress_bar.progress((idx + 1) / len(models))
                    
                    pipe = Pipeline([
                        ('scaler', scaler),
                        ('model', clf)
                    ])
                    
                    pipe.fit(st.session_state.X_train, st.session_state.y_train)
                    
                    y_pred_train = pipe.predict(st.session_state.X_train)
                    y_pred_test = pipe.predict(st.session_state.X_test)
                    
                    y_pred_train_probs = pipe.predict_proba(st.session_state.X_train)[:, 1]
                    y_pred_test_probs = pipe.predict_proba(st.session_state.X_test)[:, 1]
                    
                    precision_train = precision_score(st.session_state.y_train, y_pred_train, pos_label=1)
                    recall_train = recall_score(st.session_state.y_train, y_pred_train, pos_label=1)
                    f1_train = f1_score(st.session_state.y_train, y_pred_train, pos_label=1)
                    roc_auc_train = roc_auc_score(st.session_state.y_train, y_pred_train_probs)
                    
                    precision_test = precision_score(st.session_state.y_test, y_pred_test, pos_label=1)
                    recall_test = recall_score(st.session_state.y_test, y_pred_test, pos_label=1)
                    f1_test = f1_score(st.session_state.y_test, y_pred_test, pos_label=1)
                    roc_auc_test = roc_auc_score(st.session_state.y_test, y_pred_test_probs)
                    
                    results.append({
                        'Model': name,
                        'Precision (train)': precision_train,
                        'Recall (train)': recall_train,
                        'F1-Score (train)': f1_train,
                        'ROC AUC (train)': roc_auc_train,
                        'Precision (test)': precision_test,
                        'Recall (test)': recall_test,
                        'F1-Score (test)': f1_test,
                        'ROC AUC (test)': roc_auc_test
                    })
                
                results_df = pd.DataFrame(results)
                st.session_state.base_model_results = results_df
                st.session_state.scaler = scaler
        
        if 'base_model_results' in st.session_state:
            st.markdown("### Hasil Performa Model")
            
            st.dataframe(
                st.session_state.base_model_results.style.format({
                    'Precision (train)': '{:.4f}',
                    'Recall (train)': '{:.4f}',
                    'F1-Score (train)': '{:.4f}',
                    'ROC AUC (train)': '{:.4f}',
                    'Precision (test)': '{:.4f}',
                    'Recall (test)': '{:.4f}',
                    'F1-Score (test)': '{:.4f}',
                    'ROC AUC (test)': '{:.4f}'
                }).background_gradient(cmap='YlGn', subset=['Recall (test)', 'F1-Score (test)']),
                use_container_width=True
            )
            
            fig = go.Figure()
            
            for _, row in st.session_state.base_model_results.iterrows():
                fig.add_trace(go.Bar(
                    name=row['Model'],
                    x=['Precision (test)', 'Recall (test)', 'F1-Score (test)', 'ROC AUC (test)'],
                    y=[row['Precision (test)'], row['Recall (test)'], row['F1-Score (test)'], row['ROC AUC (test)']],
                    text=[f"{val:.3f}" for val in [row['Precision (test)'], row['Recall (test)'], 
                                                    row['F1-Score (test)'], row['ROC AUC (test)']]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Perbandingan Performa Model Dasar",
                xaxis_title="Metrics",
                yaxis_title="Score",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            best_models = st.session_state.base_model_results.nlargest(2, 'F1-Score (test)')
            st.success(f"""
            1. **{best_models.iloc[0]['Model']}** - F1-Score (test): {best_models.iloc[0]['F1-Score (test)']:.4f}
            2. **{best_models.iloc[1]['Model']}** - F1-Score (test): {best_models.iloc[1]['F1-Score (test)']:.4f}
            
            LightGBM akan dipilih untuk dituning selanjutnya.
            """)
elif selected_tab == "LightGBM":
    st.header(selected_tab)
    
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first!")
    else:
        # Kolom untuk scaling (sama seperti di Colab)
        cols_to_scale = ['Age', 'EducationLevel', 'BMI', 'AlcoholConsumption', 
                        'PhysicalActivity', 'SleepQuality', 'MMSE', 
                        'FunctionalAssessment', 'ADL']
        cols_to_scale = [col for col in cols_to_scale if col in st.session_state.X_train.columns]
        
        # Tab untuk metode tuning
        tuning_method = st.tabs(["Tanpa SMOTEENN", "Dengan SMOTEENN"])
        
        # ===================================================================
        # METODE 1: TANPA SMOTEENN 
        # ===================================================================
        with tuning_method[0]:
            
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.number_input("Jumlah Iterasi (n_iter)", min_value=10, max_value=100, value=20, step=5, key='lgbm_iter1')
                cv_splits = st.number_input("CV Splits", min_value=3, max_value=10, value=5, key='lgbm_cv1')
            with col2:
                scoring_metric = st.selectbox("Scoring Metric", ['recall', 'f1', 'precision', 'roc_auc'], key='lgbm_scoring1')
                random_state = st.number_input("Random State", value=42, key='lgbm_rs1')
            
            if st.button("Mulai tuning", key='lgbm_tune1'):
                with st.spinner("Sedang memuat model..."):
                    from sklearn.pipeline import Pipeline
                    
                    
                    pipe_tuned = Pipeline([
                        ('tuned_model', LGBMClassifier(random_state=random_state, is_unbalance=True))
                    ])
                    
                   
                    param_grid = {
                        'tuned_model__max_depth': [3, 5, 7, 10],
                        'tuned_model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'tuned_model__n_estimators': [100, 200, 300, 495],
                        'tuned_model__subsample': [0.6, 0.8, 1.0],
                        'tuned_model__colsample_bytree': [0.6, 0.8, 1.0],
                        'tuned_model__reg_alpha': [0, 0.1, 1],
                        'tuned_model__reg_lambda': [0, 0.1, 1],
                        'tuned_model__num_leaves': [15, 31, 63],
                        'tuned_model__min_child_samples': [10, 20, 30],
                        'tuned_model__boosting_type': ['gbdt', 'dart']
                    }
                    
                  
                    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
                    
          
                    random_search_lgbm = RandomizedSearchCV(
                        estimator=pipe_tuned,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=scoring_metric,
                        verbose=1,
                        random_state=random_state,
                        n_jobs=-1
                    )
                    
                    # Fit model
                    random_search_lgbm.fit(st.session_state.X_train, st.session_state.y_train)
                    
                  
                    tuned_lgbm = random_search_lgbm.best_estimator_
                    
                    # Store
                    st.session_state.tuned_lgbm = tuned_lgbm
                    st.session_state.lgbm_best_params = random_search_lgbm.best_params_
                    st.session_state.lgbm_best_score = random_search_lgbm.best_score_
                    
                    # Prediksi pada data train
                    tuned_lgbm_pred_train = tuned_lgbm.predict(st.session_state.X_train)
                    tuned_lgbm_pred_train_probs = tuned_lgbm.predict_proba(st.session_state.X_train)[:, 1]
                    
                    # Prediksi pada data test
                    tuned_lgbm_pred_test = tuned_lgbm.predict(st.session_state.X_test)
                    tuned_lgbm_pred_test_probs = tuned_lgbm.predict_proba(st.session_state.X_test)[:, 1]
                    
                    # Evaluasi metrik untuk data train
                    tuned_train_lgbm_metrics = {
                        'Precision': precision_score(st.session_state.y_train, tuned_lgbm_pred_train, pos_label=1),
                        'Recall': recall_score(st.session_state.y_train, tuned_lgbm_pred_train, pos_label=1),
                        'F1-Score': f1_score(st.session_state.y_train, tuned_lgbm_pred_train, pos_label=1),
                        'ROC AUC': roc_auc_score(st.session_state.y_train, tuned_lgbm_pred_train_probs)
                    }
                    
                    # Evaluasi metrik untuk data test
                    tuned_test_lgbm_metrics = {
                        'Precision': precision_score(st.session_state.y_test, tuned_lgbm_pred_test, pos_label=1),
                        'Recall': recall_score(st.session_state.y_test, tuned_lgbm_pred_test, pos_label=1),
                        'F1-Score': f1_score(st.session_state.y_test, tuned_lgbm_pred_test, pos_label=1),
                        'ROC AUC': roc_auc_score(st.session_state.y_test, tuned_lgbm_pred_test_probs)
                    }
                    
                    # Buat DataFrame evaluasi
                    tuned_lgbm_evaluation_df = pd.DataFrame(
                        [tuned_train_lgbm_metrics, tuned_test_lgbm_metrics], 
                        index=['Train', 'Test']
                    )
                    st.session_state.lgbm_results = tuned_lgbm_evaluation_df

            # Display results
            if 'lgbm_best_params' in st.session_state:   
                st.markdown("#### Best Hyperparameters")
                params_df = pd.DataFrame(
                    list(st.session_state.lgbm_best_params.items()),
                    columns=['Parameter', 'Value']
                )
                st.dataframe(params_df, use_container_width=True)
                
                st.markdown("#### Model Performance")
                st.dataframe(
                    st.session_state.lgbm_results.style.format('{:.4f}')
                    .background_gradient(cmap='YlGn'),
                    use_container_width=True
                )
                
                # Visualization
                fig = go.Figure()
                metrics = st.session_state.lgbm_results.columns
                
                fig.add_trace(go.Bar(
                    name='Train',
                    x=metrics,
                    y=st.session_state.lgbm_results.loc['Train'],
                    text=[f"{val:.3f}" for val in st.session_state.lgbm_results.loc['Train']],
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    name='Test',
                    x=metrics,
                    y=st.session_state.lgbm_results.loc['Test'],
                    text=[f"{val:.3f}" for val in st.session_state.lgbm_results.loc['Test']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="LightGBM Performance (Train vs Test)",
                    xaxis_title="Metrics",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ================================================================
        # METODE 2: DENGAN SMOTEENN 
        # ================================================================
        with tuning_method[1]:
            col1, col2 = st.columns(2)
            with col1:
                n_iter_smote = st.number_input("Jumlah Iterasi (n_iter)", min_value=10, max_value=100, value=20, step=5, key='lgbm_iter2')
                cv_splits_smote = st.number_input("CV Splits", min_value=3, max_value=10, value=5, key='lgbm_cv2')
            with col2:
                scoring_metric_smote = st.selectbox("Scoring Metric", ['recall', 'f1', 'precision', 'roc_auc'], key='lgbm_scoring2')
                random_state_smote = st.number_input("Random State", value=42, key='lgbm_rs2')
            
            if st.button("Mulai tuning", key='lgbm_tune2'):
                with st.spinner("Memuat model..."):
                  
                    from imblearn.pipeline import Pipeline
                    from imblearn.combine import SMOTEENN
                    
            
                    pipe_imb = Pipeline([
                        ('scaler', ColumnTransformer([
                            ('scale', MinMaxScaler(), cols_to_scale)
                        ], remainder='passthrough')),
                        ('sampling', SMOTEENN(random_state=random_state_smote)),
                        ('model', LGBMClassifier(random_state=random_state_smote))
                    ])
                    
                  
                    param_grid = {
                        'model__max_depth': [3, 5, 7, 10],
                        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'model__n_estimators': [100, 200, 300, 495],
                        'model__subsample': [0.6, 0.8, 1.0],
                        'model__colsample_bytree': [0.6, 0.8, 1.0],
                        'model__reg_alpha': [0, 0.1, 1],
                        'model__reg_lambda': [0, 0.1, 1],
                        'model__num_leaves': [15, 31, 63]
                    }
                    
                  
                    cv = StratifiedKFold(n_splits=cv_splits_smote, shuffle=True, random_state=random_state_smote)
                    
               
                    random_search = RandomizedSearchCV(
                        estimator=pipe_imb,
                        param_distributions=param_grid,
                        n_iter=n_iter_smote,
                        cv=cv,
                        scoring=scoring_metric_smote,
                        random_state=random_state_smote,
                        n_jobs=-1
                    )
                    
                    # Fit model
                    random_search.fit(st.session_state.X_train, st.session_state.y_train)
            
                    tuned_model_lgbm_imb = random_search.best_estimator_
                    
                    # Store
                    st.session_state.tuned_lgbm_smote = tuned_model_lgbm_imb
                    st.session_state.lgbm_smote_best_params = random_search.best_params_
                    st.session_state.lgbm_smote_best_score = random_search.best_score_
                    
                    # Predict 
                    pred_lgbm_imb_train = tuned_model_lgbm_imb.predict(st.session_state.X_train)
                    pred_lgbm_imb_train_probs = tuned_model_lgbm_imb.predict_proba(st.session_state.X_train)[:, 1]
                    pred_lgbm_imb_test = tuned_model_lgbm_imb.predict(st.session_state.X_test)
                    pred_lgbm_imb_test_probs = tuned_model_lgbm_imb.predict_proba(st.session_state.X_test)[:, 1]
                    
                    # Evaluate 
                    lgbm_imb_train_metrics = {
                        'Precision': precision_score(st.session_state.y_train, pred_lgbm_imb_train),
                        'Recall': recall_score(st.session_state.y_train, pred_lgbm_imb_train),
                        'F1-Score': f1_score(st.session_state.y_train, pred_lgbm_imb_train),
                        'ROC AUC': roc_auc_score(st.session_state.y_train, pred_lgbm_imb_train_probs)
                    }
                    
                    lgbm_imb_test_metrics = {
                        'Precision': precision_score(st.session_state.y_test, pred_lgbm_imb_test),
                        'Recall': recall_score(st.session_state.y_test, pred_lgbm_imb_test),
                        'F1-Score': f1_score(st.session_state.y_test, pred_lgbm_imb_test),
                        'ROC AUC': roc_auc_score(st.session_state.y_test, pred_lgbm_imb_test_probs)
                    }
                    
                    # DataFrame evaluasi
                    tuned_lgbm_evaluation_df = pd.DataFrame(
                        [lgbm_imb_train_metrics, lgbm_imb_test_metrics], 
                        index=['Train', 'Test']
                    )
                    st.session_state.lgbm_smote_results = tuned_lgbm_evaluation_df
            
            # Display results
            if 'lgbm_smote_best_params' in st.session_state:                
                st.markdown("#### Best Hyperparameters")
                params_df = pd.DataFrame(
                    list(st.session_state.lgbm_smote_best_params.items()),
                    columns=['Parameter', 'Value']
                )
                st.dataframe(params_df, use_container_width=True)
                
                st.markdown("#### Model Performance")
                st.dataframe(
                    st.session_state.lgbm_smote_results.style.format('{:.4f}')
                    .background_gradient(cmap='YlGn'),
                    use_container_width=True
                )
                
                # Visualization
                fig = go.Figure()
                metrics = st.session_state.lgbm_smote_results.columns
                
                fig.add_trace(go.Bar(
                    name='Train',
                    x=metrics,
                    y=st.session_state.lgbm_smote_results.loc['Train'],
                    text=[f"{val:.3f}" for val in st.session_state.lgbm_smote_results.loc['Train']],
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    name='Test',
                    x=metrics,
                    y=st.session_state.lgbm_smote_results.loc['Test'],
                    text=[f"{val:.3f}" for val in st.session_state.lgbm_smote_results.loc['Test']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="LightGBM + SMOTEENN Performance (Train vs Test)",
                    xaxis_title="Metrics",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Evaluasi Model":
    st.header(selected_tab)
    
    # Cek apakah ada model yang sudah di-tuning
    if 'tuned_lgbm' not in st.session_state and 'tuned_xgb' not in st.session_state:
        st.warning("Belum ada model yang di-tuning. Silakan lakukan hyperparameter tuning terlebih dahulu!")
    else:     
        # ===================================================================
        # EVALUASI LIGHTGBM
        # ===================================================================
        if 'tuned_lgbm' in st.session_state or 'tuned_lgbm_smote' in st.session_state:            
            # Kumpulkan semua hasil LightGBM
            lgbm_data = []
            lgbm_indices = []
            
            if 'lgbm_results' in st.session_state:
                lgbm_data.extend([
                    st.session_state.lgbm_results.loc['Train'].to_dict(),
                    st.session_state.lgbm_results.loc['Test'].to_dict()
                ])
                lgbm_indices.extend([
                    'Hyperparameter Tuning LightGBM (Train)',
                    'Hyperparameter Tuning LightGBM (Test)'
                ])
            
            if 'lgbm_smote_results' in st.session_state:
                lgbm_data.extend([
                    st.session_state.lgbm_smote_results.loc['Train'].to_dict(),
                    st.session_state.lgbm_smote_results.loc['Test'].to_dict()
                ])
                lgbm_indices.extend([
                    'Hyperparameter Tuning LightGBM menggunakan SMOTEENN (Train)',
                    'Hyperparameter Tuning LightGBM menggunakan SMOTEENN (Test)'
                ])
            
            if lgbm_data:
                evaluation_df_lgbm = pd.DataFrame(lgbm_data, index=lgbm_indices)
                st.dataframe(
                    evaluation_df_lgbm.style.format('{:.4f}')
                    .background_gradient(cmap='YlGn', subset=['F1-Score', 'ROC AUC']),
                    use_container_width=True
                )
                
                # Visualization
                # fig = go.Figure()
                
                # for idx, row_name in enumerate(lgbm_indices):
                #     if 'Test' in row_name:
                #         fig.add_trace(go.Bar(
                #             name=row_name,
                #             x=list(evaluation_df_lgbm.columns),
                #             y=evaluation_df_lgbm.loc[row_name],
                #             text=[f"{val:.3f}" for val in evaluation_df_lgbm.loc[row_name]],
                #             textposition='auto'
                #         ))
                
                # fig.update_layout(
                #     title="Perbandingan Performa LightGBM (Test Set)",
                #     xaxis_title="Metrics",
                #     yaxis_title="Score",
                #     barmode='group',
                #     height=500
                # )
                # st.plotly_chart(fig, use_container_width=True)
                
                # ===================================================================
                # CONFUSION MATRIX LIGHTGBM
                # ===================================================================
                st.markdown("### Confusion Matrix")
                
                cm_cols = st.columns(2)
                
                # Confusion Matrix: Tuned LightGBM (tanpa SMOTEENN)
                if 'tuned_lgbm' in st.session_state:
                    with cm_cols[0]:
                        tuned_lgbm_pred_test = st.session_state.tuned_lgbm.predict(st.session_state.X_test)
                        cm_lgbm = confusion_matrix(st.session_state.y_test, tuned_lgbm_pred_test)
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm_lgbm,
                            x=['Non AD', 'Alzheimer'],
                            y=['Non AD', 'Alzheimer'],
                            text=cm_lgbm,
                            texttemplate='%{text}',
                            colorscale='Reds',
                            showscale=True
                        ))
                        
                        fig_cm.update_layout(
                            title='Confusion Matrix - Tuned LightGBM',
                            xaxis_title='Predicted',
                            yaxis_title='Actual',
                            height=400
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                
                # Confusion Matrix: Tuned LightGBM dengan SMOTEENN
                if 'tuned_lgbm_smote' in st.session_state:
                    with cm_cols[1]:
                        pred_lgbm_imb_test = st.session_state.tuned_lgbm_smote.predict(st.session_state.X_test)
                        cm_lgbm_smote = confusion_matrix(st.session_state.y_test, pred_lgbm_imb_test)
                        
                        fig_cm_smote = go.Figure(data=go.Heatmap(
                            z=cm_lgbm_smote,
                            x=['Non AD', 'Alzheimer'],
                            y=['Non AD', 'Alzheimer'],
                            text=cm_lgbm_smote,
                            texttemplate='%{text}',
                            colorscale='Reds',
                            showscale=True
                        ))
                        
                        fig_cm_smote.update_layout(
                            title='Confusion Matrix - LightGBM + SMOTEENN',
                            xaxis_title='Predicted',
                            yaxis_title='Actual',
                            height=400
                        )
                        st.plotly_chart(fig_cm_smote, use_container_width=True)

        st.markdown("---")
        st.markdown("## Perbandingan Model Terbaik")
        
        # Kumpulkan semua model untuk perbandingan
        all_models = []
        
        if 'lgbm_results' in st.session_state:
            all_models.append({
                'Model': 'LightGBM',
                **st.session_state.lgbm_results.loc['Test'].to_dict()
            })
        
        if 'lgbm_smote_results' in st.session_state:
            all_models.append({
                'Model': 'LightGBM + SMOTEENN',
                **st.session_state.lgbm_smote_results.loc['Test'].to_dict()
            })
        
        if all_models:
            comparison_df = pd.DataFrame(all_models)
            comparison_df = comparison_df.set_index('Model')
            
            st.dataframe(
                comparison_df.style.format('{:.4f}')
                .background_gradient(cmap='RdYlGn', subset=comparison_df.columns),
                use_container_width=True
            )
            
            # Bar chart comparison
            fig_compare = go.Figure()
            
            for metric in comparison_df.columns:
                fig_compare.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df.index,
                    y=comparison_df[metric],
                    text=[f"{val:.3f}" for val in comparison_df[metric]],
                    textposition='auto'
                ))
            
            fig_compare.update_layout(
                title="Perbandingan Model (Test Set)",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            


elif selected_tab == "Fitur Penting":
    st.header(selected_tab)
    
    # Cek apakah model LightGBM (Tanpa SMOTEENN) sudah di-tune
    if 'tuned_lgbm' not in st.session_state:
        st.warning("⚠️ Model LightGBM belum di-tune. Silakan lakukan tuning terlebih dahulu di tab 'LightGBM'!")
    else:
        try:
            # Get model yang sudah di-train
            selected_model = st.session_state.tuned_lgbm
            feature_names = st.session_state.X_train.columns
            
            # Get feature importance dari model yang sudah di-train
            importances = None
            
            # Cek apakah model adalah pipeline
            if hasattr(selected_model, 'named_steps'):
                # Model dalam pipeline - ambil dari step 'tuned_model'
                if 'tuned_model' in selected_model.named_steps:
                    model_step = selected_model.named_steps['tuned_model']
                    if hasattr(model_step, 'feature_importances_'):
                        importances = model_step.feature_importances_
            elif hasattr(selected_model, 'feature_importances_'):
                # Model langsung (bukan pipeline)
                importances = selected_model.feature_importances_
                st.success("Feature importance berhasil diambil dari model LightGBM")
            
            if importances is None:
                st.error("❌ Tidak dapat mengambil feature importance dari model")
            else:
                # Create DataFrame
                fi_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Display top N
                top_n = st.slider("Jumlah Feature Importance", 5, len(fi_df), min(15, len(fi_df)), key='fi_topn')
                
                fi_top = fi_df.head(top_n)
                
                # Metrics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(fi_df))
                with col2:
                    st.metric("Top Feature", fi_top.iloc[0]['Feature'])
                with col3:
                    st.metric("Top Importance", f"{fi_top.iloc[0]['Importance']:.4f}")
                
                st.markdown("---")
                
                # Table
                st.markdown("#### Feature Importance Table")
                st.dataframe(
                    fi_top.style.format({'Importance': '{:.4f}'})
                    .background_gradient(cmap='YlGn', subset=['Importance']),
                    use_container_width=True
                )
                
                # Bar chart
                st.markdown("#### Feature Importance Chart")
                fig = go.Figure(go.Bar(
                    x=fi_top['Importance'],
                    y=fi_top['Feature'],
                    orientation='h',
                    marker=dict(
                        color=fi_top['Importance'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f"{val:.4f}" for val in fi_top['Importance']],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=max(500, top_n * 30),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif selected_tab == "Kesimpulan & Rekomendasi":
    st.header(selected_tab)
    
    # ===================================================================
    # KESIMPULAN
    # ===================================================================
    st.markdown("## Kesimpulan")
    
    st.markdown("""
    <div style="background-color: #f0f9ff; 
                padding: 25px; 
                border-radius: 10px; 
                border-left: 5px solid #3b82f6;
                margin-bottom: 20px;">
        <p style="font-size: 16px; line-height: 1.8; color: #1e40af; text-align: justify;">
        Model <strong>LightGBM Hyperparameter Tuning tanpa SMOTEENN </strong> menjadi pilihan terbaik dalam memprediksi diagnosis Alzheimer 
        karena menunjukkan performa sebagian besar lebih baik di beberapa metrik evaluasi. Dilihat dari performa 
        F-1 Score sebesar 98% menunjukkan keseimbangan performa model lebih baik dibandingan LightGBM + SMOTEEN. Recall sebesar 92% sedikit lebih rendah dibandingkan LightGBM + SMOTEENN dalam memprediksi pasien tersebut mengalami Alzheimer. Namun dalam hal presicion, Light GBM Tuned lebih baik yaitu 86% dalam memprediksi pasien tersebut benar-benar mengalami Alzheimer 
        Selain itu, model dapat membedakan pasien Alzheimer dan Non-AD sebesar 94% berdasarkan metrik ROC-AUC.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===================================================================
    # REKOMENDASI
    # ===================================================================
    st.markdown("## Rekomendasi")
    
    # Rekomendasi 1
    st.markdown("""
    <div style="background-color: #fff7ed; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #f97316;
                margin-bottom: 15px;">
        <h4 style="color: #c2410c; margin-top: 0;">
            1️. Deteksi Dini dengan Validasi Dokter Spesialis
        </h4>
        <p style="font-size: 15px; line-height: 1.7; color: #431407; text-align: justify; margin-bottom: 0;">
        Model ini dapat digunakan sebagai deteksi dini diagnosis Alzheimer pada pasien namun harus selalu divalidasi oleh dokter spesialis neurologi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rekomendasi 2
    st.markdown("""
    <div style="background-color: #fef2f2; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #ef4444;
                margin-bottom: 15px;">
        <h4 style="color: #991b1b; margin-top: 0;">
            2️. Jangan Abaikan Gejala Klinis Meski Hasil Negatif
        </h4>
        <p style="font-size: 15px; line-height: 1.7; color: #450a0a; text-align: justify; margin-bottom: 0;">
        Untuk pasien yang tidak terdeteksi oleh model, diharapkan tidak diabaikan gejala klinis jika hasil prediksi negatif dan lakukan pemeriksaan lanjutan ke dokter untuk mengkonfirmasi hasil diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rekomendasi 3
    st.markdown("""
    <div style="background-color: #f0fdf4; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #22c55e;
                margin-bottom: 15px;">
        <h4 style="color: #15803d; margin-top: 0;">
            3️. Komunikasikan Hasil dengan Hati-Hati kepada Pasien
        </h4>
        <p style="font-size: 15px; line-height: 1.7; color: #14532d; text-align: justify; margin-bottom: 0;">
        Jika hasil model ini disampaikan ke pasien, maka penyampaiannya harus hati-hati karena ini bukan hasil final namun hanya sebagai screening awal atau deteksi dini.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rekomendasi 4
    st.markdown("""
    <div style="background-color: #f5f3ff; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #8b5cf6;
                margin-bottom: 15px;">
        <h4 style="color: #5b21b6; margin-top: 0;">
            4️. Evaluasi dan Tingkatkan Performa Model Secara Berkala
        </h4>
        <p style="font-size: 15px; line-height: 1.7; color: #2e1065; text-align: justify; margin-bottom: 0;">
        Melakukan evaluasi performa model secara berkala agar dapat mengidentifikasi pola kesalahan prediksi agar meningkatkan akurasi model atau mempertimbangkan untuk upgrade ke model yang lebih canggih.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rekomendasi 5
    st.markdown("""
    <div style="background-color: #ecfeff; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 5px solid #06b6d4;
                margin-bottom: 15px;">
        <h4 style="color: #155e75; margin-top: 0;">
            5️. Berikan Pelatihan kepada Pihak yang Terlibat
        </h4>
        <p style="font-size: 15px; line-height: 1.7; color: #164e63; text-align: justify; margin-bottom: 0;">
        Berikan pelatihan atau edukasi ke pihak yang terlibat terkait keunggulan dan kelemahan model ini.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

