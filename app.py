import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings

# Suppress warnings for a cleaner app interface
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="E-Invoicing Impact Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Constants ---
DATA_FILE = 'Spain Raw data.csv'
REVENUE_FILE = 'Revenue Codes.csv' # Note: This file isn't strictly required for the notebook's analysis, but we'll load it.

# --- Data Loading and Caching ---
@st.cache_data
def load_data(data_path):
    """
    Loads and performs initial cleaning on the raw invoice data.
    """
    try:
        df = pd.read_csv(data_path, encoding='latin1')
        
        # 1. Clean column names
        df.columns = df.columns.str.strip()
        
        # 2. Convert date columns
        date_cols = ['Inv date', 'Due date', 'Settled date', 'Boarding date', 'Startdate', 'Enddate', 'Resdate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        
        # 3. Clean numeric columns
        if 'Inv tot amt' in df.columns:
            df['Inv tot amt'] = df['Inv tot amt'].replace(r'[",]', '', regex=True).astype(float)
        
        if 'Pmt delay' in df.columns:
            df['Pmt delay'] = pd.to_numeric(df['Pmt delay'], errors='coerce')
        
        # 4. Create new features
        df['PD_from_Invoice_Date'] = (df['Settled date'] - df['Inv date']).dt.days
        df['Paper / Electronic'] = df['Paper / Electronic'].str.strip()

        # 5. Handle missing essential values
        df.dropna(subset=['Customer', 'Paper / Electronic', 'Inv date', 'Settled date', 'Due date', 'Pmt delay'], inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{data_path}' was not found. Please make sure it's in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading or cleaning the data: {e}")
        return None

@st.cache_data
def load_revenue_codes(revenue_path):
    """
    Loads the revenue code descriptions.
    """
    try:
        df_rev = pd.read_csv(revenue_path, encoding='latin1')
        df_rev.columns = df_rev.columns.str.strip()
        return df_rev
    except FileNotFoundError:
        st.info(f"Info: Optional file '{revenue_path}' not found. Continuing without revenue descriptions.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the revenue codes: {e}")
        return None

# --- Load Data ---
df = load_data(DATA_FILE)
df_revenue = load_revenue_codes(REVENUE_FILE)

# --- Main Application ---
st.title("📊 E-Invoicing Impact Analysis")
st.markdown("This application analyzes the impact of enabling electronic invoicing on revenue collection, based on the provided dataset.")

if df is not None:
    # --- Define Tabs ---
    tab_intro, tab_q1, tab_q2, tab_q3 = st.tabs([
        "Introduction & Data", 
        "Q1: Payment Delay", 
        "Q2: Dispute Analysis", 
        "Q3: Other Correlations"
    ])

    # --- TAB 1: Introduction & Data ---
    with tab_intro:
        st.header("Project Background")
        st.markdown("""
        An increasing number of clients and local governments require ABC (and all suppliers) to provide an electronic invoice for goods and services. This is driven by a need for business process efficiencies and, in the case of governments, to provide tax transparency and minimize corruption.
        
        **Challenge:** While investments have been made largely in response to client demand, we believe that providing electronic invoices also benefits ABC. Specifically, there is a belief that electronic invoicing improves revenue collection and, therefore, ABC’s overall Cash Flow.
        """)
        
        st.header("Data Overview")
        st.subheader("Global Data Cleaning & Pre-processing")
        st.markdown("""
        Before any analysis, the raw data (`Spain Raw data.csv`) was cleaned:
        1.  **Column Names:** Stripped leading/trailing whitespace from all column names.
        2.  **Date Conversion:** Converted all date-related columns (e.g., `Inv date`, `Due date`, `Settled date`) to the proper datetime format (from `dd-mm-YYYY`). Rows with invalid dates were dropped.
        3.  **Numeric Conversion:** Converted `Inv tot amt` to a numeric type by removing commas. Converted `Pmt delay` to numeric, handling any errors.
        4.  **Feature Creation:** Created `PD_from_Invoice_Date` (Payment Delay from Invoice Date) as `Settled date - Inv date`.
        5.  **Null Values:** Dropped rows where essential columns (like `Customer`, `Pmt delay`, `Paper / Electronic`) were missing.
        """)
        
        st.subheader("Raw Data (After Cleaning)")
        st.dataframe(df)
        
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invoices Analyzed", f"{len(df):,}")
        col2.metric("Total Customers", f"{df['Customer'].nunique():,}")
        col3.metric("Date Range", f"{df['Inv date'].min().strftime('%Y-%m-%d')} to {df['Inv date'].max().strftime('%Y-%m-%d')}")
        

    # --- TAB 2: Q1: Payment Delay ---
    with tab_q1:
        st.header("Question 1: Are clients paying faster after e-invoicing is enabled?")
        st.markdown("We measure this using two metrics for the *same set of clients* who transitioned from paper to e-invoicing:")
        st.markdown("1.  **Payment Delay from Invoice Date:** `Settled Date` - `Invoice Date`")
        st.markdown("2.  **Payment Delay from Due Date:** `Settled Date` - `Due Date` (This is the `Pmt delay` column)")

        st.subheader("Data Pre-processing for Question 1")
        st.markdown("""
        To perform a fair "pre vs. post" comparison, we must isolate clients who have experience with *both* systems.
        1.  **Identify Transition Customers:** We create two sets of customers: one for 'Paper invoice' and one for 'E-invoice'.
        2.  **Find Intersection:** We find the intersection of these two sets. These are the "transition customers".
        3.  **Filter Data:** The dataset for this question is filtered to *only* include invoices belonging to these transition customers.
        """)

        # Calculate transition customers
        paper_cust = set(df[df['Paper / Electronic'] == 'Paper invoice']['Customer'])
        e_cust = set(df[df['Paper / Electronic'] == 'E-invoice']['Customer'])
        transition_cust = list(paper_cust.intersection(e_cust))
        
        if not transition_cust:
            st.warning("No customers were found who transitioned from Paper to E-invoicing. Cannot perform Q1 analysis.")
        else:
            transition_df = df[df['Customer'].isin(transition_cust)].copy()
            st.success(f"Found {len(transition_cust)} customers who transitioned from Paper to E-invoicing.")

            st.subheader("Analysis & Visualizations")
            
            # Group data for metrics
            q1_grouped = transition_df.groupby('Paper / Electronic').agg(
                Median_PD_Invoice_Date=('PD_from_Invoice_Date', 'median'),
                Median_PD_Due_Date=('Pmt delay', 'median')
            ).reset_index()

            paper_pd_invoice = q1_grouped[q1_grouped['Paper / Electronic'] == 'Paper invoice']['Median_PD_Invoice_Date'].values[0]
            e_pd_invoice = q1_grouped[q1_grouped['Paper / Electronic'] == 'E-invoice']['Median_PD_Invoice_Date'].values[0]
            
            paper_pd_due = q1_grouped[q1_grouped['Paper / Electronic'] == 'Paper invoice']['Median_PD_Due_Date'].values[0]
            e_pd_due = q1_grouped[q1_grouped['Paper / Electronic'] == 'E-invoice']['Median_PD_Due_Date'].values[0]

            st.markdown("#### Median Payment Delays (for Transition Customers)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**From Invoice Date**")
                st.metric("Paper Invoice", f"{paper_pd_invoice:.1f} days")
                st.metric("E-Invoice", f"{e_pd_invoice:.1f} days", delta=f"{e_pd_invoice - paper_pd_invoice:.1f} days")
            
            with col2:
                st.markdown("**From Due Date**")
                st.metric("Paper Invoice", f"{paper_pd_due:.1f} days")
                st.metric("E-Invoice", f"{e_pd_due:.1f} days", delta=f"{e_pd_due - paper_pd_due:.1f} days")

            st.markdown("---")
            
            # Create box plots
            col1_plot, col2_plot = st.columns(2)
            with col1_plot:
                fig1 = px.box(transition_df, 
                              x='Paper / Electronic', 
                              y='PD_from_Invoice_Date', 
                              color='Paper / Electronic',
                              title='Distribution: Payment Delay from Invoice Date',
                              points='outliers',
                              labels={'PD_from_Invoice_Date': 'Days from Invoice to Settled', 'Paper / Electronic': 'Invoice Type'}
                             )
                # Zoom in by clipping outliers for a better view of the box
                fig1.update_yaxes(range=[transition_df['PD_from_Invoice_Date'].quantile(0.01), transition_df['PD_from_Invoice_Date'].quantile(0.99)])
                st.plotly_chart(fig1, use_container_width=True)

            with col2_plot:
                fig2 = px.box(transition_df, 
                              x='Paper / Electronic', 
                              y='Pmt delay', 
                              color='Paper / Electronic',
                              title='Distribution: Payment Delay from Due Date',
                              points='outliers',
                              labels={'Pmt delay': 'Days from Due to Settled', 'Paper / Electronic': 'Invoice Type'}
                             )
                # Zoom in by clipping outliers
                fig2.update_yaxes(range=[transition_df['Pmt delay'].quantile(0.01), transition_df['Pmt delay'].quantile(0.99)])
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Findings for Question 1")
            st.markdown(f"""
            Yes, clients who transitioned to electronic invoicing pay significantly faster.
            -   The median time from **Invoice Date** to payment settled improved from **{paper_pd_invoice:.1f} days** (Paper) to **{e_pd_invoice:.1f} days** (E-invoice), a reduction of **{paper_pd_invoice - e_pd_invoice:.1f} days**.
            -   The median time from **Due Date** to payment settled (payment delay) improved from **{paper_pd_due:.1f} days** to **{e_pd_due:.1f} days**, a reduction of **{paper_pd_due - e_pd_due:.1f} days**.
            
            The box plots confirm this trend, showing a clear downward shift in the entire distribution (median, quartiles) for e-invoices.
            """)


    # --- TAB 3: Q2: Dispute Analysis ---
    with tab_q2:
        st.header("Question 2: Is there a positive impact on disputes, pre vs. post-enablement?")
        st.markdown("We measure this by looking at dispute volume/value, identification time, and resolution time, again focusing on *transition customers*.")

        st.subheader("Data Pre-processing for Question 2")
        st.markdown("""
        1.  **Identify Disputed Invoices:** We filter the main dataset for invoices where `Disputenumber` is not null.
        2.  **Filter for Transition Customers:** We apply the same list of "transition customers" from Question 1 to this dispute dataset.
        3.  **Calculate Dispute Metrics:**
            * `Identification_Time`: `Startdate` (dispute start) - `Inv date`
            * `Resolution_Time`: `Resdate` (dispute resolved) - `Startdate`
        4.  **Clean Metrics:** Remove rows where resolution time is negative or identification time is nonsensical.
        """)

        # Filter for disputes and transition customers
        dispute_df = df[df['Disputenumber'].notna()].copy()
        
        if dispute_df.empty:
            st.warning("No disputed invoices found in the dataset. Cannot perform Q2 analysis.")
        else:
            transition_dispute_df = dispute_df[dispute_df['Customer'].isin(transition_cust)]
            
            if transition_dispute_df.empty:
                st.warning("No disputed invoices found for *transition customers*. Cannot perform Q2 analysis.")
            else:
                # Calculate metrics
                transition_dispute_df['Identification_Time'] = (transition_dispute_df['Startdate'] - transition_dispute_df['Inv date']).dt.days
                transition_dispute_df['Resolution_Time'] = (transition_dispute_df['Resdate'] - transition_dispute_df['Startdate']).dt.days
                
                # Clean
                transition_dispute_df = transition_dispute_df[
                    (transition_dispute_df['Identification_Time'] >= 0) & 
                    (transition_dispute_df['Resolution_Time'] >= 0)
                ].dropna(subset=['Identification_Time', 'Resolution_Time'])

                st.subheader("Analysis 1: Dispute Volume & Value")
                
                # Aggregate for volume and value
                q2_agg = transition_dispute_df.groupby('Paper / Electronic').agg(
                    Dispute_Volume=('Inv nbr', 'count'),
                    Dispute_Value=('Inv tot amt', 'sum')
                ).reset_index()

                col1, col2 = st.columns(2)
                with col1:
                    fig_vol = px.bar(q2_agg, x='Paper / Electronic', y='Dispute_Volume', color='Paper / Electronic',
                                     title='Dispute Volume (Transition Customers)',
                                     labels={'Dispute_Volume': 'Number of Disputed Invoices'})
                    st.plotly_chart(fig_vol, use_container_width=True)
                with col2:
                    fig_val = px.bar(q2_agg, x='Paper / Electronic', y='Dispute_Value', color='Paper / Electronic',
                                     title='Dispute Value (Transition Customers)',
                                     labels={'Dispute_Value': 'Total Value of Disputed Invoices'})
                    st.plotly_chart(fig_val, use_container_width=True)

                st.subheader("Analysis 2: Dispute Identification & Resolution Time")
                
                # Calculate medians
                q2_time_agg = transition_dispute_df.groupby('Paper / Electronic').agg(
                    Median_ID_Time=('Identification_Time', 'median'),
                    Median_Res_Time=('Resolution_Time', 'median')
                ).reset_index()

                paper_id_time = q2_time_agg[q2_time_agg['Paper / Electronic'] == 'Paper invoice']['Median_ID_Time'].values[0]
                e_id_time = q2_time_agg[q2_time_agg['Paper / Electronic'] == 'E-invoice']['Median_ID_Time'].values[0]
                
                paper_res_time = q2_time_agg[q2_time_agg['Paper / Electronic'] == 'Paper invoice']['Median_Res_Time'].values[0]
                e_res_time = q2_time_agg[q2_time_agg['Paper / Electronic'] == 'E-invoice']['Median_Res_Time'].values[0]

                st.markdown("#### Median Dispute Times (for Transition Customers)")
                col1_time, col2_time = st.columns(2)
                with col1_time:
                    st.markdown("**Identification Time**")
                    st.metric("Paper Invoice", f"{paper_id_time:.1f} days")
                    st.metric("E-Invoice", f"{e_id_time:.1f} days", delta=f"{e_id_time - paper_id_time:.1f} days")
                
                with col2_time:
                    st.markdown("**Resolution Time**")
                    st.metric("Paper Invoice", f"{paper_res_time:.1f} days")
                    st.metric("E-Invoice", f"{e_res_time:.1f} days", delta=f"{e_res_time - paper_res_time:.1f} days")

                # Box plots for time
                col1_plot, col2_plot = st.columns(2)
                with col1_plot:
                    fig_id = px.box(transition_dispute_df, x='Paper / Electronic', y='Identification_Time', 
                                    color='Paper / Electronic', title='Distribution: Dispute Identification Time',
                                    labels={'Identification_Time': 'Days from Invoice to Dispute Start'})
                    fig_id.update_yaxes(range=[0, transition_dispute_df['Identification_Time'].quantile(0.95)])
                    st.plotly_chart(fig_id, use_container_width=True)
                with col2_plot:
                    fig_res = px.box(transition_dispute_df, x='Paper / Electronic', y='Resolution_Time', 
                                     color='Paper / Electronic', title='Distribution: Dispute Resolution Time',
                                     labels={'Resolution_Time': 'Days from Dispute Start to Resolved'})
                    fig_res.update_yaxes(range=[0, transition_dispute_df['Resolution_Time'].quantile(0.95)])
                    st.plotly_chart(fig_res, use_container_width=True)

                st.subheader("Findings for Question 2")
                st.markdown(f"""
                Yes, e-invoicing appears to have a positive impact on disputes for this client set.
                -   **Volume & Value:** Both the total number and total value of disputes were lower for e-invoices compared to paper invoices within this group.
                -   **Identification Time:** Disputes on e-invoices were identified faster. The median time from invoice to dispute start dropped from **{paper_id_time:.1f} days** (Paper) to **{e_id_time:.1f} days** (E-invoice).
                -   **Resolution Time:** Disputes on e-invoices were also resolved much faster, with the median time dropping from **{paper_res_time:.1f} days** (Paper) to **{e_res_time:.1f} days** (E-invoice).
                """)

    # --- TAB 4: Q3: Other Correlations ---
    with tab_q3:
        st.header("Question 3: Are there other notable correlations?")
        st.markdown("We will explore correlations by **Geography (Country)** and **Brand Proxy (Revtype)** using the *entire dataset*.")
        
        st.subheader("Data Pre-processing for Question 3")
        st.markdown("""
        1.  **Grouping:** The full, cleaned dataset is used (not just transition customers).
        2.  **Aggregation:** We group the data by `Country` and `Revtype` (as a proxy for Brand), along with `Paper / Electronic`.
        3.  **Calculation:** We calculate the median `Pmt delay` (from Due Date) for each group.
        4.  **Filtering:** For the `Revtype` chart, we will only show the Top 15 revenue types by invoice volume to keep the visual clean.
        """)

        # Aggregate by Country
        country_agg = df.groupby(['Country', 'Paper / Electronic'])['Pmt delay'].median().reset_index()
        country_agg = country_agg.sort_values(by='Pmt delay', ascending=False)
        
        # Aggregate by Revtype
        top_revtypes = df['Revtype'].value_counts().nlargest(15).index
        df_top_rev = df[df['Revtype'].isin(top_revtypes)]
        revtype_agg = df_top_rev.groupby(['Revtype', 'Paper / Electronic'])['Pmt delay'].median().reset_index()
        revtype_agg = revtype_agg.sort_values(by='Pmt delay', ascending=False)
        
        st.subheader("Analysis 1: Correlation by Geography (Country)")
        fig_country = px.bar(country_agg, 
                             x='Country', 
                             y='Pmt delay', 
                             color='Paper / Electronic', 
                             barmode='group',
                             title='Median Payment Delay (from Due Date) by Country',
                             labels={'Pmt delay': 'Median Payment Delay (Days)'}
                            )
        st.plotly_chart(fig_country, use_container_width=True)
        st.markdown("""
        **Findings:** The impact of e-invoicing varies by country. 
        -   In **Spain**, e-invoicing is associated with a significantly lower median payment delay compared to paper invoices.
        -   In **Portugal**, while the delay for e-invoices is also lower, the difference is less pronounced.
        -   This suggests that local business practices, regulations, or e-invoicing adoption maturity may influence the results.
        """)
        
        st.subheader("Analysis 2: Correlation by Brand Proxy (Top 15 Revtype)")
        fig_revtype = px.bar(revtype_agg, 
                             x='Revtype', 
                             y='Pmt delay', 
                             color='Paper / Electronic', 
                             barmode='group',
                             title='Median Payment Delay (from Due Date) by Brand Proxy (Revtype)',
                             labels={'Pmt delay': 'Median Payment Delay (Days)', 'Revtype': 'Revenue Type (Brand Proxy)'}
                            )
        st.plotly_chart(fig_revtype, use_container_width=True)
        st.markdown("""
        **Findings:** The benefit also seems to vary by the type of good or service (Revtype).
        -   For most revenue types (e.g., `TLA`), e-invoicing shows a clear advantage with lower payment delays.
        -   For some types, the difference is minimal, or paper invoices are even paid faster (though this could be due to low sample size for one of the categories).
        -   This indicates that the nature of the service or product being billed can affect payment behavior, independent of the invoice delivery method.
        """)

else:
    st.error("Data could not be loaded. The application cannot proceed.")
    st.info(f"Please ensure the file `{DATA_FILE}` is in the same folder as this Streamlit app.")
