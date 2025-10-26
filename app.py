import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="B2B E-Invoicing Analysis")

# Define file paths based on the uploaded files
# --- FIX 1: Updated file paths to match your uploaded file names ---
RAW_DATA_FILE = "Spain Raw data.csv"
REVENUE_CODES_FILE = "Revenue Codes.csv"
QUESTIONS_FILE = "Sheet1.csv"

@st.cache_data
def load_data():
    """
    Loads and cleans the raw data and revenue codes.
    """
    try:
        # --- FIX 2: Added encoding='latin1' to handle the 'utf-8' codec error ---
        
        # Load Raw Data - skipping the metadata header rows
        raw_df = pd.read_csv(RAW_DATA_FILE, skiprows=6, encoding='latin1')
        
        # Load Revenue Codes - skipping the blank rows
        revenue_df = pd.read_csv(REVENUE_CODES_FILE, skiprows=[1, 2, 3], encoding='latin1')
        
    except FileNotFoundError as e:
        st.error(f"Error: File not found. Make sure the files are in the same directory.")
        st.stop()
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
        return None, None

    # --- Clean Raw Data ---
    # Rename columns for clarity and consistency
    raw_df.columns = raw_df.columns.str.strip() # Strip whitespace from headers
    raw_df.rename(columns={
        'Paper / Electronic': 'Invoice_Type',
        'Cust nbr': 'Customer_ID',
        'Inv nbr': 'Invoice_ID',
        'Inv date': 'Invoice_Date',
        'Due date': 'Due_Date',
        'Settled date': 'Settled_Date',
        'Inv tot amt': 'Invoice_Amount',
        'Pmt delay': 'PD_from_Due_Date',
        'Revtype': 'Rev_Type',
        'Disputenumber': 'Dispute_Number',
        'Invdisputedate': 'Dispute_Start_Date',
        'Resdate': 'Dispute_Resolution_Date',
        'Status2': 'Dispute_Status'
    }, inplace=True)

    # Filter for Spain data only (as per file name, but good practice)
    # Check if 'Country' column exists before filtering
    if 'Country' in raw_df.columns:
        raw_df = raw_df[raw_df['Country'] == 'Spain'].copy()

    # Clean string columns
    str_cols = ['Invoice_Type', 'Customer_ID', 'Rev_Type']
    for col in str_cols:
        if col in raw_df.columns:
            # FIX: Convert to string type *before* stripping to handle numeric IDs
            raw_df[col] = raw_df[col].astype(str).str.strip()

    # Convert date columns
    date_cols = ['Invoice_Date', 'Due_Date', 'Settled_Date', 'Dispute_Start_Date', 'Dispute_Resolution_Date']
    for col in date_cols:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors='coerce')

    # Convert numeric columns
    num_cols = ['Invoice_Amount', 'PD_from_Due_Date']
    for col in num_cols:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

    # Drop rows essential for analysis if they are missing
    raw_df.dropna(subset=['Invoice_Date', 'Settled_Date', 'Invoice_Type', 'Customer_ID'], inplace=True)

    # --- Clean Revenue Data ---
    revenue_df.columns = revenue_df.columns.str.strip()
    revenue_df.rename(columns={'R/T': 'Rev_Type', 'Descr': 'Revenue_Description'}, inplace=True)
    if 'Rev_Type' in revenue_df.columns:
        # FIX: Convert to string type *before* stripping for robustness
        revenue_df['Rev_Type'] = revenue_df['Rev_Type'].astype(str).str.strip()
    
    return raw_df, revenue_df

@st.cache_data
def preprocess_data(_raw_df, _revenue_df):
    """
    Merges data, engineers features, and identifies the transitioning client cohort.
    """
    # Merge raw data with revenue descriptions
    df = pd.merge(_raw_df, _revenue_df[['Rev_Type', 'Revenue_Description']], on='Rev_Type', how='left')

    # --- Feature Engineering (Q1 & Q2) ---
    
    # Calculate Payment Delay from Invoice Date
    df['PD_from_Invoice_Date'] = (df['Settled_Date'] - df['Invoice_Date']).dt.days

    # Identify "Transitioning Clients"
    # These are clients who have BOTH paper and e-invoices
    clients_with_e = set(df[df['Invoice_Type'] == 'E-invoice']['Customer_ID'].unique())
    clients_with_paper = set(df[df['Invoice_Type'] == 'Paper invoice']['Customer_ID'].unique())
    transitioning_clients_ids = list(clients_with_e.intersection(clients_with_paper))

    transitioning_df = df[df['Customer_ID'].isin(transitioning_clients_ids)].copy()

    # Create 'Period' column for Pre/Post analysis
    transitioning_df['Period'] = np.where(transitioning_df['Invoice_Type'] == 'E-invoice', 
                                          'Post-Enablement (E-invoice)', 
                                          'Pre-Enablement (Paper)')

    # --- Feature Engineering (Q2 - Disputes) ---
    # Note: 'Startdate' column from raw data might be 'Dispute_Start_Date'. Using 'Invdisputedate' as per Q2 text.
    df['Is_Disputed'] = ~df['Dispute_Number'].isnull()
    df['Dispute_Identification_Time'] = (df['Dispute_Start_Date'] - df['Invoice_Date']).dt.days
    df['Dispute_Resolution_Time'] = (df['Dispute_Resolution_Date'] - df['Dispute_Start_Date']).dt.days
    
    transitioning_df['Is_Disputed'] = ~transitioning_df['Dispute_Number'].isnull()
    transitioning_df['Dispute_Identification_Time'] = (transitioning_df['Dispute_Start_Date'] - transitioning_df['Invoice_Date']).dt.days
    transitioning_df['Dispute_Resolution_Time'] = (transitioning_df['Dispute_Resolution_Date'] - transitioning_df['Dispute_Start_Date']).dt.days

    return df, transitioning_df

# --- Main App ---
st.title("B2B E-Invoicing Impact Analysis (Spain)")

# Load and process data
raw_df, revenue_df = load_data()
if raw_df is not None and revenue_df is not None:
    all_data, transitioning_data = preprocess_data(raw_df, revenue_df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Introduction & Data Overview", 
        "Q1: Payment Delay Analysis", 
        "Q2: Dispute Analysis", 
        "Q3: Other Correlations (Brand)"
    ])

    # --- Tab 1: Introduction & Data Overview ---
    with tab1:
        st.header("Introduction")
        st.markdown("""
        This application analyzes the impact of electronic invoicing on revenue collection for ABC in Spain.
        The goal is to answer three key questions based on the provided data:
        1.  **Payment Speed:** Are clients paying faster after e-invoicing is enabled?
        2.  **Disputes:** Is there a positive impact on dispute volume, value, and resolution time?
        3.  **Correlations:** Are there other notable correlations (e.g., with specific brands/revenue types)?

        **Methodology:** The "Pre vs. Post" analysis (Q1 & Q2) is performed on a specific cohort of clients:
        those who **transitioned** from paper to e-invoicing. This allows for a more accurate comparison by
        observing the change in behavior for the *same client set*.
        """)
        
        st.header("Data Overview")
        st.info(f"""
        -   **{len(raw_df)}** total transactions loaded and cleaned from Spain.
        -   **{len(transitioning_data['Customer_ID'].unique())}** clients were identified as "Transitioning Clients" (having both paper and e-invoices).
        -   Q1 and Q2 analysis is based *only* on this transitioning cohort.
        """)
        
        st.subheader("Sample of Cleaned & Merged Data")
        st.dataframe(all_data.sample(10))
        
        st.subheader("Sample of Transitioning Client Data")
        st.dataframe(transitioning_data.sample(10))

    # --- Tab 2: Q1: Payment Delay Analysis ---
    with tab2:
        st.header("Question 1: Are clients paying faster after e-invoicing?")
        st.markdown("Comparing average payment delays for transitioning clients *before* (Paper) and *after* (E-invoice) enablement.")

        # Calculate Q1 metrics
        q1_analysis = transitioning_data.groupby('Period')[['PD_from_Invoice_Date', 'PD_from_Due_Date']].mean()

        st.subheader("Average Payment Delay Comparison")
        col1, col2 = st.columns(2)
        try:
            pre_invoice_pd = q1_analysis.loc['Pre-Enablement (Paper)', 'PD_from_Invoice_Date']
            post_invoice_pd = q1_analysis.loc['Post-Enablement (E-invoice)', 'PD_from_Invoice_Date']
            delta_invoice_pd = post_invoice_pd - pre_invoice_pd
            
            pre_due_pd = q1_analysis.loc['Pre-Enablement (Paper)', 'PD_from_Due_Date']
            post_due_pd = q1_analysis.loc['Post-Enablement (E-invoice)', 'PD_from_Due_Date']
            delta_due_pd = post_due_pd - pre_due_pd

            col1.metric(
                label="Avg. Delay from Invoice Date (Post-Enablement)",
                value=f"{post_invoice_pd:.1f} days",
                delta=f"{delta_invoice_pd:.1f} days (from Pre-Enablement)"
            )
            col1.metric(
                label="Avg. Delay from Invoice Date (Pre-Enablement)",
                value=f"{pre_invoice_pd:.1f} days",
                delta=None
            )
            
            col2.metric(
                label="Avg. Delay from Due Date (Post-Enablement)",
                value=f"{post_due_pd:.1f} days",
                delta=f"{delta_due_pd:.1f} days (from Pre-Enablement)"
            )
            col2.metric(
                label="Avg. Delay from Due Date (Pre-Enablement)",
                value=f"{pre_due_pd:.1f} days",
                delta=None
            )
        except KeyError:
            st.error("Could not calculate metrics. Data might be missing for Pre/Post periods.")
        
        st.dataframe(q1_analysis)

        st.subheader("Distribution of Payment Delays")
        st.markdown("Boxplots show the full distribution of payment delays, not just the average. Shorter boxes and lower medians (center line) are better.")
        
        # Filter out extreme outliers for better visualization (e.g., > 365 days)
        viz_data_q1 = transitioning_data[
            (transitioning_data['PD_from_Invoice_Date'].abs() < 365) &
            (transitioning_data['PD_from_Due_Date'].abs() < 365)
        ]

        chart1 = alt.Chart(viz_data_q1).mark_boxplot().encode(
            x=alt.X('Period:N', title="Enablement Period"),
            y=alt.Y('PD_from_Invoice_Date:Q', title="Days"),
            color=alt.Color('Period:N', legend=None)
        ).properties(
            title='Payment Delay from INVOICE Date'
        )
        
        chart2 = alt.Chart(viz_data_q1).mark_boxplot().encode(
            x=alt.X('Period:N', title="Enablement Period"),
            y=alt.Y('PD_from_Due_Date:Q', title="Days"),
            color=alt.Color('Period:N', legend=None)
        ).properties(
            title='Payment Delay from DUE Date'
        )
        
        st.altair_chart(chart1 | chart2, use_container_width=True)

        st.subheader("Implications")
        st.markdown("""
        **Interpretation:**
        -   A **negative delta** in the metrics above indicates an *improvement* (faster payment).
        -   **PD from Invoice Date:** This measures the total cash conversion cycle from the client's perspective.
        -   **PD from Due Date:** This measures client compliance with payment terms.
        
        Based on the data, we can conclude whether e-invoicing has led to faster payments. The boxplots help visualize if this change is a broad trend or just a shift in the average.
        """)

    # --- Tab 3: Q2: Dispute Analysis ---
    with tab3:
        st.header("Question 2: Is there a positive impact on disputes?")
        st.markdown("Comparing dispute metrics for transitioning clients *before* (Paper) and *after* (E-invoice) enablement.")

        # --- Metric 1: Dispute Volume & Value ---
        st.subheader("Dispute Volume & Value (Pre vs. Post)")
        
        disputed_data = transitioning_data[transitioning_data['Is_Disputed'] == True]
        
        # Aggregate stats
        grouped = transitioning_data.groupby('Period')
        total_invoices = grouped['Invoice_ID'].count()
        total_disputes = grouped['Is_Disputed'].sum()
        total_disputed_value = disputed_data.groupby('Period')['Invoice_Amount'].sum()

        dispute_summary = pd.DataFrame({
            'Total_Invoices': total_invoices,
            'Total_Disputes': total_disputes,
            'Total_Disputed_Value': total_disputed_value
        }).fillna(0)

        dispute_summary['Dispute_Rate (%)'] = (dispute_summary['Total_Disputes'] / dispute_summary['Total_Invoices']) * 100
        dispute_summary['Avg_Value_per_Dispute'] = (dispute_summary['Total_Disputed_Value'] / dispute_summary['Total_Disputes']).fillna(0)
        
        st.dataframe(dispute_summary.style.format({
            'Total_Invoices': '{:,.0f}',
            'Total_Disputes': '{:,.0f}',
            'Total_Disputed_Value': '${:,.2f}',
            'Dispute_Rate (%)': '{:.2f}%',
            'Avg_Value_per_Dispute': '${:,.2f}'
        }))

        # --- Metric 2 & 3: Dispute Timelines ---
        st.subheader("Dispute Timeline Analysis (Pre vs. Post)")
        st.markdown("Analyzing the average time to *identify* a dispute and the time to *resolve* it.")
        
        dispute_times = disputed_data.groupby('Period')[['Dispute_Identification_Time', 'Dispute_Resolution_Time']].mean()
        
        st.dataframe(dispute_times.style.format('{:.1f} days'))
        
        st.subheader("Implications")
        st.markdown("""
        **Interpretation:**
        -   **Dispute Rate:** A lower dispute rate post-enablement is desirable.
        -   **Identification Time:** A *shorter* identification time is better, as it means issues are being flagged faster.
        -   **Resolution Time:** A *shorter* resolution time is better, as it means disputes are being settled more quickly, improving cash flow.
        
        This analysis shows whether e-invoicing has helped streamline the dispute process.
        """)

    # --- Tab 4: Q3: Other Correlations (Brand) ---
    with tab4:
        st.header("Question 3: Are there other notable correlations?")
        st.markdown("""
        The prompt asks for correlations with Geography, Client Set (Industry), and Brand.
        -   **Geography:** This analysis is limited to **Spain**, so no cross-country comparison is possible.
        -   **Industry:** Industry data was not provided in the files.
        -   **Brand (Revenue Type):** We can analyze performance by 'Brand' (using the `Revenue_Description`).
        """)

        st.subheader("Analysis by Brand (Revenue Type)")
        st.markdown("This shows performance *after* enablement to identify brands that may still have collection challenges *despite* e-invoicing.")

        # Analyze all e-invoices
        e_invoices_df = all_data[all_data['Invoice_Type'] == 'E-invoice'].copy()

        brand_analysis = e_invoices_df.groupby('Revenue_Description').agg(
            Total_Invoices=pd.NamedAgg(column='Invoice_ID', aggfunc='count'),
            Avg_PD_from_Invoice=pd.NamedAgg(column='PD_from_Invoice_Date', aggfunc='mean'),
            Avg_PD_from_Due=pd.NamedAgg(column='PD_from_Due_Date', aggfunc='mean'),
            Total_Disputes=pd.NamedAgg(column='Is_Disputed', aggfunc='sum')
        ).reset_index()
        
        brand_analysis['Dispute_Rate (%)'] = (brand_analysis['Total_Disputes'] / brand_analysis['Total_Invoices']) * 100
        
        # Filter for brands with a meaningful number of invoices
        min_invoices = st.slider("Minimum number of invoices to show a brand:", 1, 100, 10)
        filtered_brands = brand_analysis[brand_analysis['Total_Invoices'] >= min_invoices]

        st.dataframe(
            filtered_brands.sort_values(by='Avg_PD_from_Invoice', ascending=False),
            use_container_width=True
        )

        st.subheader("Top 10 Worst Performing Brands (by Avg. Delay from Invoice Date)")
        top_10_worst = filtered_brands.nlargest(10, 'Avg_PD_from_Invoice')
        
        chart_brand = alt.Chart(top_10_worst).mark_bar().encode(
            x=alt.X('Revenue_Description:N', sort='-y', title="Brand / Revenue Description"),
            y=alt.Y('Avg_PD_from_Invoice:Q', title="Avg. Payment Delay from Invoice (Days)"),
            tooltip=['Revenue_Description', 'Total_Invoices', 'Avg_PD_from_Invoice', 'Dispute_Rate (%)']
        ).properties(
            title="Top 10 Brands by Payment Delay (Post-Enablement)"
        ).interactive()
        
        st.altair_chart(chart_brand, use_container_width=True)

        st.subheader("Implications")
        st.markdown("""
        This analysis helps pinpoint if collection issues are isolated to specific products or services (brands).
        
        Even with e-invoicing, brands with high average payment delays may have underlying issues related to:
        -   Complex billing for that service.
        -   Contract terms specific to those brands.
        -   Systemic issues with the clients who purchase those services.
        
        ABC could use this to target process improvements for the worst-performing brands.
        """)

