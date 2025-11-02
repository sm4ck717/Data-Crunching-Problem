import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="E-Invoicing Impact Analysis",
    page_icon="📊"
)

# --- Data Loading & Caching ---

@st.cache_data
def load_data(file_path):
    """Loads and performs initial cleaning on the raw data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the app.")
        return None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df.rename(columns={'Pmt delay': 'PD_from_Due_Date'}, inplace=True)
    return df

@st.cache_data
def preprocess_data(_df):
    """Applies all preprocessing and feature engineering steps."""
    if _df is None:
        return None
        
    df = _df.copy()
    
    # 1. Convert date columns
    date_cols = ['Inv date', 'Due date', 'Settled date', 'Invdisputedate', 'Startdate', 'Resdate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        
    # 2. Convert 'Inv tot amt' to numeric
    if df['Inv tot amt'].dtype == 'object':
        df['Inv tot amt'] = df['Inv tot amt'].str.replace(',', '', regex=False).astype(float)
        
    # 3. Clean 'Paper / Electronic' column
    df['Paper / Electronic'] = df['Paper / Electronic'].str.strip()
    
    # 4. Drop rows with nulls in key analytical columns
    key_cols = ['Customer', 'Paper / Electronic', 'PD_from_Due_Date', 'Settled date', 'Inv date', 'Due date']
    df_processed = df.dropna(subset=key_cols)
    
    # 5. Feature Engineering
    df_processed['PD_from_Inv_Date'] = (df_processed['Settled date'] - df_processed['Inv date']).dt.days
    df_processed['Is_Disputed'] = df_processed['Disputenumber'].notnull()
    df_processed['Identification_Time'] = (df_processed['Invdisputedate'] - df_processed['Inv date']).dt.days
    df_processed['Resolution_Time'] = (df_processed['Resdate'] - df_processed['Startdate']).dt.days
    df_processed['Is_Late'] = df_processed['PD_from_Due_Date'] > 0
    
    return df_processed

@st.cache_data
def get_switcher_clients(_df):
    """Filters the DataFrame to include only 'switcher' clients."""
    if _df is None:
        return None, []
        
    client_invoice_types = _df.groupby('Customer')['Paper / Electronic'].nunique()
    switcher_clients_list = client_invoice_types[client_invoice_types > 1].index
    switcher_df = _df[_df['Customer'].isin(switcher_clients_list)].copy()
    return switcher_df, switcher_clients_list.tolist()

# --- Main App ---

# Load and process data
raw_df = load_data('Spain Raw data.csv')
if raw_df is not None:
    df_processed = preprocess_data(raw_df)
    switcher_df, switcher_list = get_switcher_clients(df_processed)

    st.title("📊 Electronic Invoicing Impact Analysis")
    # st.markdown("An interactive report on revenue collection and dispute resolution based on the `analysis.ipynb` findings.")

    # --- 1.0 Background and Problem Statement ---
    with st.expander("1.0 Background and Problem Statement", expanded=True):
        st.subheader("1.1 Background")
        st.markdown("""
        An increasing number of ABC clients and local governments require ABC (and all suppliers) to provide an electronic invoice for goods and services. Clients require this to improve business process efficiencies (e.g., 3-way match), while governments are mandating it for transparency and to minimize corruption.
        
        In 2013, over $10B of ABC’s revenue was electronically invoiced, representing around 400 clients and 100K transactions.
        """)
        
        st.subheader("1.2 Problem Statement")
        st.markdown("""
        ABC has invested in e-invoicing capabilities largely in response to client demand. This analysis seeks to determine if there is also a measurable benefit to ABC, specifically in improving revenue collection and overall cash flow.
        """)

    # --- 2.0 Methodology & Data Preparation ---
    st.header("2.0 Methodology & Data Preparation")
    st.info(f"""
    **Core Methodology:** To conduct a valid 'pre vs. post' analysis, we isolated **'switcher clients'**—clients who used *both* paper and e-invoicing. 
    This ensures we compare the same groups and control for client-specific payment behaviors.
    """)
    
    col1, col2 = st.columns(2)
    col1.metric("Total 'Switcher' Clients", len(switcher_list))
    col2.metric("Total Invoices from Switchers", f"{len(switcher_df):,}")

    with st.expander("View 'Switcher' Client List and Data Cleaning Steps"):
        st.subheader("Switcher Clients Analyzed:")
        st.write(switcher_list)
        
        st.subheader("Data Cleaning & Preprocessing Steps:")
        st.markdown("""
        1.  **Loaded Data:** `Spain Raw data.csv`
        2.  **Cleaned Columns:** Removed leading/trailing whitespace.
        3.  **Converted Dates:** All date columns (Inv date, Due date, etc.) converted to datetime objects.
        4.  **Converted Numerics:** `Inv tot amt` cleaned of commas and converted to a float.
        5.  **Filtered Nulls:** Removed rows with missing key dates or customer info.
        6.  **Engineered Features:** Created new columns for the analysis:
            * `PD_from_Inv_Date`: (Settled date - Inv date)
            * `Is_Disputed`: (True/False)
            * `Identification_Time`: (Invdisputedate - Inv date)
            * `Resolution_Time`: (Resdate - Startdate)
            * `Is_Late`: (PD_from_Due_Date > 0)
        """)
        st.dataframe(switcher_df)

    # --- 3.0 Q1: Payment Delay Analysis ---
    st.header("3.0 Q1: Are clients paying faster with e-invoicing?")
    
    # Calculate Q1 metrics
    pd_comparison = switcher_df.groupby('Paper / Electronic')[['PD_from_Due_Date', 'PD_from_Inv_Date']].agg(['mean', 'median', 'count'])
    late_invoice_pct = switcher_df.groupby('Paper / Electronic')['Is_Late'].mean()

    st.subheader("Finding 1: Payment Delay (Median vs. Mean)")
    st.markdown("""
    The analysis shows that while the **mean** (average) is skewed by extreme outliers, the **median** (50th percentile) provides a more accurate view of typical payment behavior.
    """)
    st.dataframe(pd_comparison)
    
    st.success("**Insight:** E-invoicing shows a clear improvement in median payment speed. The typical cash cycle is **7 days shorter**.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Median PD from Due Date (Paper)", value="10.0 days")
        st.metric(label="Median PD from Due Date (E-invoice)", value="7.0 days", delta="-3.0 days (Faster)")
    with col2:
        st.metric(label="Median PD from Invoice Date (Paper)", value="56.0 days")
        st.metric(label="Median PD from Invoice Date (E-invoice)", value="49.0 days", delta="-7.0 days (Faster)")

    st.subheader("Finding 2: Distribution of Payment Delays")
    # Melt data for Plotly
    plot_df_q1 = switcher_df.melt(
        id_vars=['Customer', 'Paper / Electronic'], 
        value_vars=['PD_from_Due_Date', 'PD_from_Inv_Date'], 
        var_name='Delay_Type', 
        value_name='Days_Delayed'
    )
    plot_df_q1['Delay_Type'] = plot_df_q1['Delay_Type'].replace({
        'PD_from_Due_Date': 'PD from Due Date',
        'PD_from_Inv_Date': 'PD from Invoice Date'
    })
    
    fig1 = px.box(plot_df_q1, 
                  x='Delay_Type', 
                  y='Days_Delayed', 
                  color='Paper / Electronic', 
                  title="Distribution of Payment Delays (Box Plot)",
                  labels={'Days_Delayed': 'Days', 'Delay_Type': 'Payment Delay Metric'},
                  color_discrete_map={'Paper invoice': 'skyblue', 'E-invoice': 'lightgreen'})
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Insight:** The box plot confirms the median (the line in the box) for e-invoices is lower (faster). However, e-invoices also have several large outliers that skew the *mean* higher, making the median a more reliable metric for this analysis.")

    st.subheader("Finding 3: Percentage of Late Invoices")
    col1, col2 = st.columns(2)
    fig_gauge_paper = go.Figure(go.Indicator(
        mode="gauge+number",
        value=late_invoice_pct['Paper invoice'] * 100,
        title={'text': "% of Paper Invoices Paid Late"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "skyblue"}}
    ))
    fig_gauge_e = go.Figure(go.Indicator(
        mode="gauge+number",
        value=late_invoice_pct['E-invoice'] * 100,
        title={'text': "% of E-Invoices Paid Late"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "lightgreen"}}
    ))
    col1.plotly_chart(fig_gauge_paper, use_container_width=True)
    col2.plotly_chart(fig_gauge_e, use_container_width=True)
    
    st.success(f"**Insight:** The percentage of invoices paid *after* the due date dropped significantly from **{late_invoice_pct['Paper invoice']:.2%}** for paper to **{late_invoice_pct['E-invoice']:.2%}** for e-invoices.")


    # --- 4.0 Q2: Dispute Analysis ---
    st.header("4.0 Q2: Is there a positive impact on disputes?")
    
    st.subheader("Finding 1: Dispute Volume and Value")
    st.warning("**Insight:** No. The dispute *rate* by volume more than doubled, though the rate by *value* (as a % of total value) saw a slight improvement.")

    # Calculate Q2 metrics
    total_agg = switcher_df.groupby('Paper / Electronic').agg(
        Total_Volume=('Inv nbr', 'count'),
        Total_Value=('Inv tot amt', 'sum')
    )
    disputed_agg = switcher_df[switcher_df['Is_Disputed'] == True].groupby('Paper / Electronic').agg(
        Disputed_Volume=('Inv nbr', 'count'),
        Disputed_Value=('Inv tot amt', 'sum')
    )
    volume_value_summary = total_agg.join(disputed_agg).fillna(0)
    volume_value_summary['Dispute_Rate_Volume_%'] = (volume_value_summary['Disputed_Volume'] / volume_value_summary['Total_Volume']) * 100
    volume_value_summary['Dispute_Rate_Value_%'] = (volume_value_summary['Disputed_Value'] / volume_value_summary['Total_Value']) * 100
    
    st.dataframe(volume_value_summary.style.format({
        'Total_Value': '{:,.2f}',
        'Disputed_Value': '{:,.2f}',
        'Dispute_Rate_Volume_%': '{:.2f}%',
        'Dispute_Rate_Value_%': '{:.2f}%'
    }))
    
    st.subheader("Finding 2: Dispute Timings (Identification & Resolution)")
    st.error("**Insight:** The dispute process is significantly *slower* for e-invoices. On median, identification takes **9 days longer**, and resolution takes **25 days longer**.")
    
    time_metrics_df = switcher_df[switcher_df['Is_Disputed'] == True]
    time_summary = time_metrics_df.groupby('Paper / Electronic')[['Identification_Time', 'Resolution_Time']].agg(['mean', 'median', 'count'])
    st.dataframe(time_summary)
    
    # Melt for plot
    plot_time_df = time_metrics_df.melt(
        id_vars=['Customer', 'Paper / Electronic'], 
        value_vars=['Identification_Time', 'Resolution_Time'], 
        var_name='Metric_Type', 
        value_name='Days'
    )
    plot_time_df['Metric_Type'] = plot_time_df['Metric_Type'].replace({
        'Identification_Time': 'Identification Time',
        'Resolution_Time': 'Resolution Time'
    })
    
    fig2 = px.box(plot_time_df, 
                  x='Metric_Type', 
                  y='Days', 
                  color='Paper / Electronic', 
                  title="Dispute Time Analysis (Box Plot)",
                  labels={'Days': 'Days', 'Metric_Type': 'Dispute Metric'},
                  color_discrete_map={'Paper invoice': 'skyblue', 'E-invoice': 'lightgreen'})
    st.plotly_chart(fig2, use_container_width=True)

    # --- 5.0 Q3: Correlations ---
    st.header("5.0 Q3: Are there other notable correlations?")
    st.subheader("Finding: Impact Varies Significantly by Geography & Client")
    st.info("**Insight:** The positive payment trend is driven by **Spain** and **Portugal**. The **UK** shows a strong *negative* trend, with e-invoicing being significantly slower than paper.")

    # Calculate Q3 aggregates
    country_agg = switcher_df.groupby(['Country', 'Paper / Electronic'])['PD_from_Due_Date'].median().reset_index()
    revtype_agg = switcher_df.groupby(['Revtype', 'Paper / Electronic'])['PD_from_Due_Date'].median().reset_index()
    customer_agg = switcher_df.groupby(['Customer', 'Paper / Electronic'])['PD_from_Due_Date'].median().reset_index()

    # Plotly Bar Charts
    fig_country = px.bar(country_agg, 
                         x='Country', 
                         y='PD_from_Due_Date', 
                         color='Paper / Electronic', 
                         barmode='group', 
                         title='Median Payment Delay by Country',
                         labels={'PD_from_Due_Date': 'Median Days Delayed'},
                         color_discrete_map={'Paper invoice': 'skyblue', 'E-invoice': 'lightgreen'})
    
    fig_customer = px.bar(customer_agg, 
                          x='Customer', 
                          y='PD_from_Due_Date', 
                          color='Paper / Electronic', 
                          barmode='group', 
                          title='Median Payment Delay by Customer (Client Set)',
                          labels={'PD_from_Due_Date': 'Median Days Delayed'},
                          color_discrete_map={'Paper invoice': 'skyblue', 'E-invoice': 'lightgreen'})
    
    fig_revtype = px.bar(revtype_agg, 
                         x='Revtype', 
                         y='PD_from_Due_Date', 
                         color='Paper / Electronic', 
                         barmode='group', 
                         title='Median Payment Delay by Revtype (Brand Proxy)',
                         labels={'PD_from_Due_Date': 'Median Days Delayed'},
                         color_discrete_map={'Paper invoice': 'skyblue', 'E-invoice': 'lightgreen'})

    st.plotly_chart(fig_country, use_container_width=True)
    st.plotly_chart(fig_customer, use_container_width=True)
    
    with st.expander("Show Breakdown by Revtype (Brand Proxy)"):
        st.plotly_chart(fig_revtype, use_container_width=True)


    # --- 6.0 Conclusions & Recommendations ---
    st.header("6.0 Conclusions & Recommendations")
    
    st.subheader("Conclusions")
    st.success("✅ **1. Payment Speed (DSO): E-invoicing is a success.** It reduces the median invoice-to-cash cycle by **7 days** and cuts the rate of late payments significantly.")
    st.error("❌ **2. Dispute Process: E-invoicing is failing.** The process is significantly slower (median **25 days longer** to resolve) and the *volume* of disputes has more than doubled.")
    st.warning("⚠️ **3. Regional Variation: The program's success is not universal.** Spain and Portugal show great results, but the UK's performance is negative and requires immediate investigation.")
    
    st.subheader("Recommendations")
    st.markdown("""
    * **1. Investigate UK Performance:** Conduct a root-cause analysis for the UK's negative trend (16-day *increase* in median delay). This could be a technical, process, or client-side issue.
    * **2. Overhaul the E-Invoice Dispute Process:** A 147% increase in median resolution time (from 17 to 42 days) is unacceptable. Map the e-invoice dispute journey to identify and eliminate bottlenecks.
    * **3. Continue Rollout (with Caution):** The positive results in Spain and Portugal support the program's expansion. However, future rollouts must include pre-launch process checks for both payment and dispute systems to avoid repeating the issues seen in the UK.
    """)
