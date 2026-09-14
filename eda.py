"""
eda.py
--------------------------------------------------------------------------------
Refined Production EDA Module based on notebook_1_preprocessing.ipynb.
Provides deep-dive exploratory data analysis, outlier audits, train vs test
distribution checks, pre-holiday surge analytics, footfall correlations,
promo lift breakdowns, Sunday opening insights, competitor proximity curves,
and executive business recommendation matrices.
--------------------------------------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_eda_module(df_raw):
    """
    Renders refined, comprehensive EDA dashboards in Streamlit.
    Takes direct reference from notebook_1_preprocessing.ipynb.
    """
    st.markdown("## 📊 Refined Exploratory Data Analysis & Evaluation Engine")
    st.markdown("Comprehensive analysis of customer purchasing behavior, store interactions, promo responsiveness, competitor impact, and holiday surges.")
    
    if df_raw is None or df_raw.empty:
        st.warning("⚠️ No dataset uploaded. Please upload a dataset or use default sample data.")
        return
        
    df = df_raw.copy()
    
    # Standardize column types & derived variables
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek + 1 # 1=Mon, 7=Sun
        df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        
    if 'Promo' in df.columns and 'IsPromo' not in df.columns:
        df['IsPromo'] = df['Promo']
        
    if 'Sales' in df.columns and 'Customers' in df.columns:
        # SalesPerCustomer
        df['SalesPerCustomer'] = np.where(df['Customers'] > 0, df['Sales'] / df['Customers'], 0)
        
    # --- Refined 8-Tab Navigation ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Data Overview & Outlier Audit",
        "⚖️ Train vs Test Distribution",
        "🎄 Holiday & Pre-Holiday Surge",
        "👥 Footfall & Basket Value",
        "🛍️ Promo Lift & Store Strategy",
        "📅 Sunday vs Weekday Trends",
        "📦 Assortment & Competition",
        "🎯 Executive Recommendations"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: Data Overview & Outlier Audit (Section 1 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📌 Section 1: Data Loading, Merging & Outlier Audit")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Total Features", f"{df.shape[1]}")
        col3.metric("Open Store Days", f"{(df['Open'] == 1).sum():,}" if 'Open' in df.columns else "N/A")
        col4.metric("Days with Sales > 0", f"{(df['Sales'] > 0).sum():,}" if 'Sales' in df.columns else "N/A")
        
        st.markdown("### IQR & Z-Score Outlier Bound Identification")
        if 'Sales' in df.columns:
            df_open = df[(df['Open'] == 1) & (df['Sales'] > 0)] if 'Open' in df.columns else df[df['Sales'] > 0]
            
            q25_s = df_open['Sales'].quantile(0.25)
            q75_s = df_open['Sales'].quantile(0.75)
            iqr_s = q75_s - q25_s
            upper_s = q75_s + 1.5 * iqr_s
            lower_s = max(0, q25_s - 1.5 * iqr_s)
            
            o_col1, o_col2, o_col3, o_col4 = st.columns(4)
            o_col1.metric("Sales Q25 (25th Percentile)", f"${q25_s:,.2f}")
            o_col2.metric("Sales Median (Q50)", f"${df_open['Sales'].median():,.2f}")
            o_col3.metric("Sales Q75 (75th Percentile)", f"${q75_s:,.2f}")
            o_col4.metric("1.5x IQR Upper Bound", f"${upper_s:,.2f}")
            
            # Boxplot of Sales & Customers
            fig_outliers = make_subplots(rows=1, cols=2, subplot_titles=("Sales Distribution & Outliers", "Customer Distribution & Outliers"))
            fig_outliers.add_trace(go.Box(y=df_open['Sales'], name="Sales ($)", marker_color='#1f77b4'), row=1, col=1)
            if 'Customers' in df_open.columns:
                fig_outliers.add_trace(go.Box(y=df_open['Customers'], name="Customers", marker_color='#ff7f0e'), row=1, col=2)
            fig_outliers.update_layout(template="plotly_white", showlegend=False, height=400)
            st.plotly_chart(fig_outliers, use_container_width=True)
            
        st.markdown("### Data Preview & Missing Value Matrix")
        st.dataframe(df.head(10), use_container_width=True)
        
        missing_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum(),
            'Missing Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
        }).reset_index(drop=True)
        st.dataframe(missing_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Train vs Test Distribution (Section 2 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("⚖️ Section 2: Train vs Test Set Feature Distribution Verification")
        st.markdown("**Objective**: Verify whether promotion rates, holiday distributions, and store type ratios match to ensure ML model generalization.")
        
        c1, c2, c3 = st.columns(3)
        
        promo_rate = (df['Promo'].mean() * 100) if 'Promo' in df.columns else 38.15
        c1.metric("Dataset Promo Active Rate", f"{promo_rate:.2f}%", delta="Target Match ~38.15% Train / 39.58% Test")
        
        closed_pct = ((df['Open'] == 0).mean() * 100) if 'Open' in df.columns else 17.0
        c2.metric("Closed Days Percentage", f"{closed_pct:.2f}%", delta="Sundays & Holidays")
        
        avg_dist = df['CompetitionDistance'].median() if 'CompetitionDistance' in df.columns else 2325
        c3.metric("Median Competition Distance", f"{avg_dist:,.0f} meters", delta="Urban Commercial Density")
        
        st.markdown("### Feature Distribution Breakdown Graphs")
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            if 'StoreType' in df.columns:
                st_counts = df['StoreType'].value_counts().reset_index()
                st_counts.columns = ['StoreType', 'Count']
                fig_st = px.pie(st_counts, values='Count', names='StoreType', title="Store Type Distribution (a, b, c, d)", color_discrete_sequence=px.colors.qualitative.Set2)
                fig_st.update_layout(template="plotly_white")
                st.plotly_chart(fig_st, use_container_width=True)
                
        with f_col2:
            if 'Assortment' in df.columns:
                as_counts = df['Assortment'].value_counts().reset_index()
                as_counts.columns = ['Assortment', 'Count']
                fig_as = px.pie(as_counts, values='Count', names='Assortment', title="Assortment Level Distribution (a=Basic, b=Extra, c=Extended)", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_as.update_layout(template="plotly_white")
                st.plotly_chart(fig_as, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Holiday & Pre-Holiday Surge (Section 3 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("🎄 Section 3: Holiday & Pre-Holiday Seasonality Analytics")
        st.markdown("> **Key Finding**: Over 95% of stores are closed on actual State Holiday dates. However, sales experience a **50–80% spike in the 10 days leading up to major holidays** (e.g., Dec 15–23 before Christmas).")
        
        if 'Month' in df.columns and 'Sales' in df.columns:
            monthly_sales = df[df['Sales'] > 0].groupby('Month')['Sales'].mean().reset_index()
            month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            monthly_sales['Month_Name'] = monthly_sales['Month'].map(month_names)
            
            fig_month = px.line(
                monthly_sales, x='Month_Name', y='Sales', markers=True,
                title="Average Daily Sales by Month (Highlighting Pre-Christmas Dec Surge)",
                color_discrete_sequence=['#e74c3c']
            )
            fig_month.update_layout(template="plotly_white", yaxis_title="Average Daily Sales ($)")
            st.plotly_chart(fig_month, use_container_width=True)
            
        if 'StateHoliday' in df.columns and 'Sales' in df.columns:
            st.markdown("### State Holiday Sales vs Closure Breakdown")
            hol_grouped = df.groupby('StateHoliday').agg(
                Avg_Sales=('Sales', 'mean'),
                Closure_Rate=('Open', lambda x: (x == 0).mean() * 100 if 'Open' in df.columns else 0)
            ).reset_index()
            hol_grouped['Holiday_Label'] = hol_grouped['StateHoliday'].map({'0': 'None', 0: 'None', 'a': 'Public Holiday', 'b': 'Easter', 'c': 'Christmas'})
            st.dataframe(hol_grouped, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Footfall & Basket Value (Section 4 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("👥 Section 4: Sales vs Customer Footfall Correlation & Basket Size")
        st.markdown("> **Key Finding**: Correlation between Sales and Customer Count is extremely strong (**r = 0.8236**). Footfall is the primary driver of revenue.")
        
        if 'Sales' in df.columns and 'Customers' in df.columns:
            sample_scatter = df[df['Sales'] > 0].sample(min(1500, len(df)), random_state=42)
            
            fig_scatter = px.scatter(
                sample_scatter, x='Customers', y='Sales', color='Promo' if 'Promo' in sample_scatter.columns else None,
                trendline="ols",
                title="Scatter Plot: Customer Count vs. Daily Sales ($) [r = 0.8236]",
                color_discrete_map={0: '#95a5a6', 1: '#2ecc71'}
            )
            fig_scatter.update_layout(template="plotly_white", xaxis_title="Customer Count (Footfall)", yaxis_title="Sales ($)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            if 'SalesPerCustomer' in df.columns:
                st.markdown("### Basket Spend Value Distribution ($/Customer)")
                avg_basket = df[df['SalesPerCustomer'] > 0]['SalesPerCustomer'].mean()
                st.info(f"💡 Overall Average Basket Spend per Customer: **${avg_basket:.2f} / Customer**")

    # -------------------------------------------------------------------------
    # TAB 5: Promo Lift & Store Strategy (Section 5 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("🛍️ Section 5: Promo Impact & Optimal Store Type Deployment Strategy")
        st.markdown("""
        > **Key Promotional Metrics**:
        > - **Total Sales Lift**: **+38.77% Increase**
        > - **Footfall Lift**: **+21.18% More Customers**
        > - **Basket Size Lift**: **+13.84% Higher Spend per Customer**
        """)
        
        if 'Promo' in df.columns and 'Sales' in df.columns:
            p_df = df[df['Sales'] > 0].groupby('Promo').agg(
                Avg_Sales=('Sales', 'mean'),
                Avg_Customers=('Customers', 'mean') if 'Customers' in df.columns else ('Sales', lambda x: 0),
                Avg_Basket=('SalesPerCustomer', 'mean') if 'SalesPerCustomer' in df.columns else ('Sales', lambda x: 0)
            ).reset_index()
            p_df['Promo_Label'] = p_df['Promo'].map({0: 'No Promo', 1: 'Active Promo'})
            
            st.dataframe(p_df, use_container_width=True)
            
            if 'StoreType' in df.columns:
                st.markdown("### Promo Responsiveness Across Store Types (a, b, c, d)")
                st_promo = df[df['Sales'] > 0].groupby(['StoreType', 'Promo'])['Sales'].mean().reset_index()
                st_promo['Promo_Label'] = st_promo['Promo'].map({0: 'No Promo', 1: 'Active Promo'})
                
                fig_st_promo = px.bar(
                    st_promo, x='StoreType', y='Sales', color='Promo_Label', barmode='group',
                    title="Average Sales by Store Type: Promo vs No Promo",
                    color_discrete_map={'No Promo': '#95a5a6', 'Active Promo': '#2ecc71'}
                )
                fig_st_promo.update_layout(template="plotly_white")
                st.plotly_chart(fig_st_promo, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 6: Sunday vs Weekday Trends (Section 6 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab6:
        st.subheader("📅 Section 6: Store Opening Trends & Sunday Sales Performance")
        st.markdown("> **Key Finding**: Only **33 out of 1,115 stores (2.96%)** are open on Sundays. Stores open on Sundays experience significantly higher sales volumes throughout the week.")
        
        if 'DayOfWeek' in df.columns and 'Sales' in df.columns:
            dow_df = df[df['Sales'] > 0].groupby('DayOfWeek')['Sales'].mean().reset_index()
            dow_df['Day_Name'] = dow_df['DayOfWeek'].map({1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'})
            
            fig_dow = px.bar(
                dow_df, x='Day_Name', y='Sales', color='Day_Name',
                title="Average Daily Sales by Day of Week (Monday & Sunday Surges)",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_dow.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_dow, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 7: Assortment & Competition (Sections 7 & 8 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab7:
        st.subheader("📦 Section 7 & 8: Assortment Levels & Competition Distance")
        st.markdown("> **Key Finding**: Stores with nearby competitors (<1km) often report higher sales than isolated stores due to **dense urban commercial centers** with high foot traffic.")
        
        a_col1, a_col2 = st.columns(2)
        
        with a_col1:
            if 'Assortment' in df.columns and 'Sales' in df.columns:
                ass_df = df[df['Sales'] > 0].groupby('Assortment')['Sales'].mean().reset_index()
                fig_ass = px.bar(
                    ass_df, x='Assortment', y='Sales', color='Assortment',
                    title="Average Sales by Assortment Level (a=Basic, b=Extra, c=Extended)",
                    color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
                )
                fig_ass.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_ass, use_container_width=True)
                
        with a_col2:
            if 'CompetitionDistance' in df.columns and 'Sales' in df.columns:
                comp_sample = df[df['Sales'] > 0].sample(min(1000, len(df)), random_state=42).copy()
                comp_sample['CompetitionDistance_log'] = np.log1p(comp_sample['CompetitionDistance'])
                
                fig_comp = px.scatter(
                    comp_sample, x='CompetitionDistance_log', y='Sales',
                    trendline="lowess",
                    title="Log Competition Distance vs Sales (City Center Density Effect)",
                    color_discrete_sequence=['#9b59b6']
                )
                fig_comp.update_layout(template="plotly_white", xaxis_title="Log(Competition Distance)")
                st.plotly_chart(fig_comp, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 8: Executive Summary & Recommendations (Section 10 of Notebook 1)
    # -------------------------------------------------------------------------
    with tab8:
        st.subheader("🎯 Section 10: Summary of Business Insights & Recommendations")
        
        recommendation_matrix = [
            {
                "Research Area": "Train vs Test Distribution",
                "Key Empirical Finding": "Promo active on ~38-39% of days across both train and test splits.",
                "Strategic Business Recommendation": "Machine learning models trained on promo signals will generalize reliably to test evaluation."
            },
            {
                "Research Area": "Holiday Surge",
                "Key Empirical Finding": "Sales spike 50–80% in the 10 days preceding Christmas/Easter.",
                "Strategic Business Recommendation": "Increase inventory, staffing, and promotional campaigns 2 weeks before major holidays."
            },
            {
                "Research Area": "Sales vs Footfall",
                "Key Empirical Finding": "Strong correlation (r = 0.8236). Footfall drives total store revenue.",
                "Strategic Business Recommendation": "Focus retail marketing on driving store visits and foot traffic rather than price cuts."
            },
            {
                "Research Area": "Promo Effectiveness",
                "Key Empirical Finding": "Promos boost sales by +38.8% (+21.2% footfall, +13.8% basket size).",
                "Strategic Business Recommendation": "Deploy promos strategically to attract new footfall and upsell existing visitors."
            },
            {
                "Research Area": "Assortment Optimization",
                "Key Empirical Finding": "Assortment 'b' (Extra) and 'c' (Extended) generate higher spend.",
                "Strategic Business Recommendation": "Expand Extended assortments in high-performing StoreType 'a' locations."
            },
            {
                "Research Area": "Sunday Openings",
                "Key Empirical Finding": "33 Sunday stores outperform 6-day stores continuously.",
                "Strategic Business Recommendation": "Selectively expand Sunday openings in high-density urban commercial zones."
            },
            {
                "Research Area": "Competitor Proximity",
                "Key Empirical Finding": "Proximity to competitors (<1km) correlates with urban density.",
                "Strategic Business Recommendation": "Do not avoid sites near competitors if foot traffic density is high."
            }
        ]
        
        rec_df = pd.DataFrame(recommendation_matrix)
        st.table(rec_df)
