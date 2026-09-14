"""
app.py
--------------------------------------------------------------------------------
Production-Grade Streamlit Multi-Page Web Application for Rossmann Store Sales
& Customer Volume Prediction & Executive Analytics.
Includes Dark Mode & Light Mode Theme Switcher!
--------------------------------------------------------------------------------
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from inference import load_model, predict_single, predict_batch
from eda import render_eda_module

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rossmann Sales & Customer Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Default Sample Data Helper
# -----------------------------------------------------------------------------
@st.cache_data
def get_sample_data():
    """Loads default sample dataset from workspace if available."""
    train_path = os.path.join(os.path.dirname(__file__), 'train.csv')
    store_path = os.path.join(os.path.dirname(__file__), 'store.csv')
    
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path, nrows=5000, parse_dates=['Date'])
        if os.path.exists(store_path):
            df_store = pd.read_csv(store_path)
            df = df_train.merge(df_store, on='Store', how='left')
        else:
            df = df_train
        return df
    else:
        dates = pd.date_range('2015-01-01', periods=100)
        return pd.DataFrame({
            'Store': 1,
            'Date': dates,
            'Sales': np.random.randint(3000, 9000, 100),
            'Customers': np.random.randint(400, 1000, 100),
            'Open': 1,
            'Promo': np.random.choice([0, 1], 100),
            'StateHoliday': '0',
            'SchoolHoliday': 0,
            'StoreType': 'a',
            'Assortment': 'a',
            'CompetitionDistance': 1270
        })

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS, THEME TOGGLE & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shop.png", width=70)
    st.markdown("### 🏬 Control Panel")
    
    # 🎨 Theme Switcher (Dark Mode / Light Mode)
    theme_choice = st.radio(
        "🎨 UI Theme Mode:",
        ["Dark Mode 🌙", "Light Mode ☀️"],
        index=0
    )
    
    st.divider()
    
    # Navigation Mode
    app_mode = st.radio(
        "Select Application Module:",
        [
            "📊 Module A: Exploratory Data Analysis (EDA)",
            "🤖 Module B: Model Architecture Specs",
            "🏪 Module C: Store Manager Forecast Center",
            "📈 Module D: Analytics & Prediction Exporter"
        ]
    )
    
    st.divider()
    
    # Model Selector
    st.markdown("### 🤖 Trained Model Architecture")
    model_choice = st.selectbox(
        "Select Model Pipeline:",
        [
            "Random Forest",
            "XGBoost",
            "Deep Learning (PyTorch LSTM)"
        ]
    )
    
    st.divider()
    
    # CSV Upload for EDA/Batch
    st.markdown("### 📁 Dataset Upload (Optional)")
    uploaded_file = st.file_uploader("Upload Rossmann CSV Dataset:", type=['csv'])
    
    st.caption("🚀 Rossmann Store Sales & Customer Intelligence System v2.0")

# -----------------------------------------------------------------------------
# DYNAMIC THEME INJECTION (DARK vs LIGHT MODE CSS)
# -----------------------------------------------------------------------------
if "Dark Mode" in theme_choice:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #e6edf3;
        }
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d2ff 0%, #00e676 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #8b949e;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #161b22;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            border: 1px solid #30363d;
            border-left: 6px solid #00d2ff;
            margin-bottom: 15px;
        }
        .metric-title {
            font-size: 0.85rem;
            color: #8b949e;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #58a6ff;
        }
        div[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_dark"
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #2c3e50;
        }
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #7f8c8d;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
            border-left: 6px solid #1f77b4;
            margin-bottom: 15px;
        }
        .metric-title {
            font-size: 0.85rem;
            color: #7f8c8d;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #2c3e50;
        }
        div[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_white"

# Load Dataset for App
if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'])
        st.sidebar.success("✅ Custom Dataset Successfully Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        data_df = get_sample_data()
else:
    data_df = get_sample_data()

# Initialize session state for batch predictions
if 'batch_predictions' not in st.session_state:
    st.session_state['batch_predictions'] = None

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Rossmann Store Sales & Customer Intelligence System</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Production-Grade Demand Forecasting Engine — Theme: <b>{theme_choice}</b></div>", unsafe_allow_html=True)

# =============================================================================
# MODULE A: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
if app_mode == "📊 Module A: Exploratory Data Analysis (EDA)":
    render_eda_module(data_df)

# =============================================================================
# MODULE B: MODEL ARCHITECTURE SPECS
# =============================================================================
elif app_mode == "🤖 Module B: Model Architecture Specs":
    st.markdown("## 🤖 Model Architecture Specifications & Benchmarks")
    
    st.markdown("""
    This application integrates **three high-performance machine learning & deep learning model architectures** evaluated on 1,115 Rossmann stores:
    """)
    
    m_tab1, m_tab2, m_tab3 = st.tabs([
        "🌲 Random Forest Regressor",
        "⚡ XGBoost Regressor",
        "🧠 PyTorch 2-Layer LSTM"
    ])
    
    with m_tab1:
        st.markdown("### Random Forest Regressor (Tuned)")
        st.markdown("""
        - **Architecture**: Ensemble Bagging tree model with 100 parallel estimator trees.
        - **Hyperparameters**: `max_depth=22`, `min_samples_split=5`, `min_samples_leaf=2`.
        - **Validation Performance**:
          - **Accuracy**: `90.67%`
          - **RMSPE**: `12.59%`
          - **$R^2$ Score**: `0.8941`
        """)
        
    with m_tab2:
        st.markdown("### XGBoost Regressor (Tuned & Optimized)")
        st.markdown("""
        - **Architecture**: Gradient Boosted Decision Tree minimizing residual log-loss.
        - **Hyperparameters**: `n_estimators=500`, `learning_rate=0.04`, `max_depth=10`, `subsample=0.8`, `colsample_bytree=0.8`.
        - **Validation Performance**:
          - **Accuracy**: **`91.80%`** (Winning Individual Architecture)
          - **RMSPE**: **`11.29%`**
          - **$R^2$ Score**: **`0.9235`**
        """)
        
    with m_tab3:
        st.markdown("### PyTorch 2-Layer LSTM Recurrent Neural Network")
        st.markdown("""
        - **Architecture**: 2-Layer Deep LSTM with Dropout Regularization.
        - **Specification**:
          - Layer 1: `LSTM(input_size=30, hidden_size=64)` + `Dropout(0.2)`
          - Layer 2: `LSTM(input_size=64, hidden_size=32)` + `Dropout(0.2)`
          - Output: `Linear(32, 1)`
        - **Performance**:
          - **MAPE**: `9.33%`
          - **RMSPE**: `12.84%`
        """)

# =============================================================================
# MODULE C: STORE MANAGER PREDICTION INTERFACE
# =============================================================================
elif app_mode == "🏪 Module C: Store Manager Forecast Center":
    st.markdown("## 🏪 Store Manager Forecast Center")
    st.markdown("Input store-specific parameters or upload a batch CSV file to generate high-accuracy daily sales and customer volume predictions.")
    
    mode_tab1, mode_tab2 = st.tabs([
        "👤 Single Store Manual Entry",
        "📂 Batch Upload (CSV Mode)"
    ])
    
    # -------------------------------------------------------------------------
    # MODE 1: SINGLE STORE MANUAL ENTRY
    # -------------------------------------------------------------------------
    with mode_tab1:
        st.subheader("📝 Input Single Store Parameters")
        
        with st.form("single_store_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                store_id = st.number_input("Store ID Number:", min_value=1, max_value=1115, value=1, step=1)
                pred_date = st.date_input("Target Forecast Date:", datetime.date(2015, 8, 1))
                store_type = st.selectbox("Store Type:", ['a', 'b', 'c', 'd'], help="a: Standard, b: City Center, c: Mall, d: Large")
                
            with col2:
                is_promo = st.radio("Active Promotion (IsPromo)?", ["No", "Yes"], index=1, horizontal=True)
                is_weekend = st.radio("Weekend Operation (IsWeekend)?", ["No", "Yes"], index=0, horizontal=True)
                assortment = st.selectbox("Assortment Level:", ['a', 'b', 'c'], help="a: Basic, b: Extra, c: Extended")
                
            with col3:
                is_holiday = st.radio("State Holiday (IsHoliday)?", ["No", "Yes"], index=0, horizontal=True)
                competitor_dist = st.number_input("Competitor Distance (meters):", min_value=10, max_value=100000, value=1270, step=100)
                
            submit_btn = st.form_submit_button("🔮 Predict Daily Sales & Customers", use_container_width=True)
            
        if submit_btn:
            with st.spinner(f"Executing {model_choice} prediction engine..."):
                res = predict_single(
                    store_id=store_id,
                    date=str(pred_date),
                    is_holiday=(is_holiday == "Yes"),
                    is_weekend=(is_weekend == "Yes"),
                    is_promo=(is_promo == "Yes"),
                    competitor_dist=competitor_dist,
                    store_type=store_type,
                    assortment=assortment,
                    model_type=model_choice
                )
                
            st.success(f"✅ Prediction generated successfully using **{model_choice}** Architecture!")
            
            # Metric Display Cards
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>💵 Predicted Daily Sales</div>
                    <div class='metric-value'>${res['predicted_sales']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #ff7f0e;'>
                    <div class='metric-title'>👥 Predicted Customer Count</div>
                    <div class='metric-value' style='color: #ff7f0e;'>{res['predicted_customers']:,} Customers</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                sales_per_cust = res['predicted_sales'] / max(1, res['predicted_customers'])
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #2ca02c;'>
                    <div class='metric-title'>📊 Average Basket Value</div>
                    <div class='metric-value' style='color: #2ca02c;'>${sales_per_cust:,.2f} / Cust</div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 2: BATCH UPLOAD (CSV MODE)
    # -------------------------------------------------------------------------
    with mode_tab2:
        st.subheader("📂 Batch CSV Store Forecasting")
        st.markdown("Upload a batch CSV file containing store operations data to generate predictions for multiple dates and stores simultaneously.")
        
        batch_file = st.file_uploader("Upload Batch CSV File:", type=['csv'], key="batch_uploader")
        
        if batch_file is not None:
            try:
                raw_batch_df = pd.read_csv(batch_file)
                st.markdown("### Batch File Preview")
                st.dataframe(raw_batch_df.head(5), use_container_width=True)
                
                if st.button("🚀 Run Batch Forecasting Engine", use_container_width=True):
                    with st.spinner(f"Running batch predictions using {model_choice}..."):
                        preds_df = predict_batch(raw_batch_df, model_type=model_choice)
                        st.session_state['batch_predictions'] = preds_df
                        st.success(f"✅ Batch prediction completed for {len(preds_df)} records!")
            except Exception as e:
                st.error(f"Error processing batch CSV file: {e}")
        else:
            st.info("💡 Need a sample batch template? Click below to load a sample batch test dataset.")
            if st.button("Load Sample Rossmann Batch Dataset"):
                sample_batch = data_df.head(50).copy()
                st.session_state['batch_predictions'] = predict_batch(sample_batch, model_type=model_choice)
                st.success("✅ Sample batch dataset processed!")
                
        # Display Batch Results if available
        if st.session_state['batch_predictions'] is not None:
            preds_df = st.session_state['batch_predictions']
            st.markdown("### 📋 Batch Prediction Results")
            st.dataframe(preds_df, use_container_width=True)
            
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Total Batch Records", f"{len(preds_df):,}")
            b_col2.metric("Total Predicted Sales Volume", f"${preds_df['Predicted_Sales'].sum():,.2f}")
            b_col3.metric("Total Predicted Customers", f"{preds_df['Predicted_Customers'].sum():,} Customers")

# =============================================================================
# MODULE D: VISUALIZATIONS & EXPORTER
# =============================================================================
elif app_mode == "📈 Module D: Analytics & Prediction Exporter":
    st.markdown("## 📈 Analytics & Prediction Exporter")
    st.markdown("Interactive time series trend visualization of predicted sales vs. customer volume with CSV export capabilities.")
    
    if st.session_state['batch_predictions'] is None:
        st.warning("⚠️ No batch prediction results found. Generating analytics on default dataset...")
        st.session_state['batch_predictions'] = predict_batch(data_df.head(100), model_type=model_choice)
        
    preds_df = st.session_state['batch_predictions'].copy()
    
    if 'Date' in preds_df.columns:
        preds_df['Date'] = pd.to_datetime(preds_df['Date'])
        preds_df.sort_values('Date', inplace=True)
        
    # --- Interactive Plotly Line Chart ---
    st.subheader("📉 Time Series Forecast Trend: Predicted Sales vs. Customer Count")
    
    if 'Date' in preds_df.columns:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        daily_preds = preds_df.groupby('Date').agg({
            'Predicted_Sales': 'sum',
            'Predicted_Customers': 'sum'
        }).reset_index()
        
        fig_trend.add_trace(
            go.Scatter(x=daily_preds['Date'], y=daily_preds['Predicted_Sales'], name="Predicted Sales ($)", line=dict(color='#00d2ff' if "Dark" in theme_choice else '#1f77b4', width=2.5)),
            secondary_y=False
        )
        fig_trend.add_trace(
            go.Scatter(x=daily_preds['Date'], y=daily_preds['Predicted_Customers'], name="Predicted Customers", line=dict(color='#ff7f0e', width=2, dash='dot')),
            secondary_y=True
        )
        
        fig_trend.update_layout(
            title="Aggregated Predicted Daily Sales & Customer Trend",
            xaxis_title="Date",
            template=plotly_template,
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True), type="date")
        )
        fig_trend.update_yaxes(title_text="Predicted Sales ($)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Predicted Customer Count", secondary_y=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
    # --- Filterable Data Table ---
    st.subheader("📋 Detailed Prediction Output Data Table")
    st.dataframe(preds_df, use_container_width=True)
    
    # --- CSV Download Handler ---
    st.subheader("📥 Export Prediction Results")
    csv_data = preds_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="💾 Download Prediction Results as CSV",
        data=csv_data,
        file_name=f"rossmann_sales_predictions_{datetime.date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

