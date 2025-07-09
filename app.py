import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import os
import re
import io

# Import your existing BankReconciliationSystem class
from main import BankReconciliationSystem

def main():
    st.set_page_config(
        page_title="Bank Reconciliation System", 
        page_icon="🏦", 
        layout="wide"
    )
    
    st.title("🏦 Bank Reconciliation System")
    st.markdown("Upload your daily bank statement and finsys files for automatic reconciliation")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    date_tolerance = st.sidebar.slider("Date Tolerance (±days)", 1, 30, 7)
    amount_tolerance = st.sidebar.number_input("Amount Tolerance (₹)", 0.01, 100.0, 0.01, step=0.01)
    
    # File upload section
    st.header("📁 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bank Statement")
        bank_file = st.file_uploader(
            "Upload Bank Statement (Excel/CSV)", 
            type=['xlsx', 'xls', 'csv'],
            key="bank_file"
        )
        if bank_file:
            st.success(f"✅ Bank file uploaded: {bank_file.name}")
    
    with col2:
        st.subheader("Finsys Data")
        finsys_file = st.file_uploader(
            "Upload Finsys Data (Excel/CSV)", 
            type=['xlsx', 'xls', 'csv'],
            key="finsys_file"
        )
        if finsys_file:
            st.success(f"✅ Finsys file uploaded: {finsys_file.name}")
    
    # Reconciliation button
    if st.button("🔄 Run Reconciliation", type="primary", use_container_width=True):
        if bank_file and finsys_file:
            run_reconciliation(bank_file, finsys_file, date_tolerance, amount_tolerance)
        else:
            st.error("❌ Please upload both files before running reconciliation")
    
    # Debug section
    st.sidebar.header("🔍 Debug Mode")
    debug_enabled = st.sidebar.checkbox("Enable Debug Mode")
    
    if debug_enabled:
        debug_amount = st.sidebar.number_input("Debug Amount", value=0.0)
        debug_date = st.sidebar.date_input("Debug Date", datetime.now())
        
        if st.sidebar.button("🐛 Debug Transaction"):
            if bank_file and finsys_file:
                debug_transaction(bank_file, finsys_file, debug_amount, debug_date.strftime("%d/%m/%Y"), date_tolerance, amount_tolerance)
            else:
                st.sidebar.error("Please upload files first")

def run_reconciliation(bank_file, finsys_file, date_tolerance, amount_tolerance):
    """Run the reconciliation process and display results"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize system
        status_text.text("🔧 Initializing reconciliation system...")
        progress_bar.progress(10)
        
        reco_system = BankReconciliationSystem(
            date_tolerance=date_tolerance,
            amount_tolerance=amount_tolerance
        )
        
        # Save uploaded files temporarily
        status_text.text("💾 Processing uploaded files...")
        progress_bar.progress(20)
        
        bank_data = load_uploaded_file(bank_file)
        finsys_data = load_uploaded_file(finsys_file)
        
        if bank_data is None or finsys_data is None:
            st.error("❌ Error loading files. Please check file format.")
            return
        
        # Standardize data
        status_text.text("🔄 Standardizing data...")
        progress_bar.progress(40)
        
        bank_clean = reco_system.standardize_bank_data(bank_data)
        finsys_clean = reco_system.standardize_finsys_data(finsys_data)
        
        if bank_clean is None or finsys_clean is None:
            st.error("❌ Error standardizing data. Please check column names.")
            return
        
        # Find matches
        status_text.text("🔍 Finding matches...")
        progress_bar.progress(60)
        
        matches_df = reco_system.find_matches(bank_clean, finsys_clean)
        
        # Identify unmatched
        status_text.text("📊 Identifying unmatched transactions...")
        progress_bar.progress(80)
        
        if not matches_df.empty:
            matched_bank_indices = matches_df['Bank_Index'].tolist()
            matched_finsys_indices = matches_df['Finsys_Index'].tolist()
            
            unmatched_bank = bank_data[~bank_data.index.isin(matched_bank_indices)].copy()
            unmatched_finsys = finsys_data[~finsys_data.index.isin(matched_finsys_indices)].copy()
        else:
            unmatched_bank = bank_data.copy()
            unmatched_finsys = finsys_data.copy()
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Reconciliation completed!")
        
        # Display results
        display_results(matches_df, unmatched_bank, unmatched_finsys, bank_clean, finsys_clean)
        
    except Exception as e:
        st.error(f"❌ Error during reconciliation: {str(e)}")
        st.exception(e)

def load_uploaded_file(uploaded_file):
    """Load uploaded file into pandas DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None

def display_results(matches_df, unmatched_bank, unmatched_finsys, bank_clean, finsys_clean):
    """Display reconciliation results in tabs"""
    
    # Summary metrics
    st.header("📊 Reconciliation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_bank = len(bank_clean)
    total_finsys = len(finsys_clean)
    total_matches = len(matches_df) if not matches_df.empty else 0
    match_rate = (total_matches / total_bank * 100) if total_bank > 0 else 0
    
    with col1:
        st.metric("Bank Transactions", f"{total_bank:,}")
    
    with col2:
        st.metric("Finsys Transactions", f"{total_finsys:,}")
    
    with col3:
        st.metric("Matched Transactions", f"{total_matches:,}")
    
    with col4:
        st.metric("Match Rate", f"{match_rate:.1f}%")
    
    # Amount summary
    if not matches_df.empty:
        bank_matched_amount = matches_df['Bank_Amount'].sum()
        finsys_matched_amount = matches_df['Finsys_Amount'].sum()
        amount_diff = finsys_matched_amount - bank_matched_amount
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Matched Bank Amount", f"₹{bank_matched_amount:,.2f}")
        
        with col2:
            st.metric("Matched Finsys Amount", f"₹{finsys_matched_amount:,.2f}")
        
        with col3:
            st.metric("Amount Difference", f"₹{amount_diff:,.2f}")
    
    # Tabs for detailed results
    st.header("📋 Detailed Results")
    
    tab1, tab2, tab3 = st.tabs(["✅ Matched Transactions", "🏦 Unmatched Bank", "💼 Unmatched Finsys"])
    
    with tab1:
        st.subheader("Matched Transactions")
        if not matches_df.empty:
            st.write(f"**Total Matches:** {len(matches_df):,}")
            
            # Add search functionality
            search_matched = st.text_input("🔍 Search matched transactions:", key="search_matched")
            
            display_df = matches_df.copy()
            if search_matched:
                mask = (
                    display_df['Bank_Description'].str.contains(search_matched, case=False, na=False) |
                    display_df['Finsys_Description'].str.contains(search_matched, case=False, na=False)
                )
                display_df = display_df[mask]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download button for matched transactions
            csv_matched = convert_df_to_csv(display_df)
            st.download_button(
                label="📥 Download Matched Transactions CSV",
                data=csv_matched,
                file_name=f"matched_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No matched transactions found.")
    
    with tab2:
        st.subheader("Unmatched Bank Transactions")
        if not unmatched_bank.empty:
            st.write(f"**Total Unmatched:** {len(unmatched_bank):,}")
            
            # Add search functionality
            search_bank = st.text_input("🔍 Search unmatched bank transactions:", key="search_bank")
            
            display_df = unmatched_bank.copy()
            if search_bank:
                # Search across all string columns
                string_cols = display_df.select_dtypes(include=['object']).columns
                mask = pd.Series(False, index=display_df.index)
                for col in string_cols:
                    mask |= display_df[col].astype(str).str.contains(search_bank, case=False, na=False)
                display_df = display_df[mask]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_bank = convert_df_to_csv(display_df)
            st.download_button(
                label="📥 Download Unmatched Bank CSV",
                data=csv_bank,
                file_name=f"unmatched_bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("All bank transactions matched!")
    
    with tab3:
        st.subheader("Unmatched Finsys Transactions")
        if not unmatched_finsys.empty:
            st.write(f"**Total Unmatched:** {len(unmatched_finsys):,}")
            
            # Add search functionality
            search_finsys = st.text_input("🔍 Search unmatched finsys transactions:", key="search_finsys")
            
            display_df = unmatched_finsys.copy()
            if search_finsys:
                # Search across all string columns
                string_cols = display_df.select_dtypes(include=['object']).columns
                mask = pd.Series(False, index=display_df.index)
                for col in string_cols:
                    mask |= display_df[col].astype(str).str.contains(search_finsys, case=False, na=False)
                display_df = display_df[mask]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_finsys = convert_df_to_csv(display_df)
            st.download_button(
                label="📥 Download Unmatched Finsys CSV",
                data=csv_finsys,
                file_name=f"unmatched_finsys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("All finsys transactions matched!")

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def debug_transaction(bank_file, finsys_file, debug_amount, debug_date, date_tolerance, amount_tolerance):
    """Debug a specific transaction"""
    
    st.sidebar.header("🔍 Debug Results")
    
    try:
        reco_system = BankReconciliationSystem(
            date_tolerance=date_tolerance,
            amount_tolerance=amount_tolerance
        )
        
        # Load and process data
        bank_data = load_uploaded_file(bank_file)
        finsys_data = load_uploaded_file(finsys_file)
        
        bank_clean = reco_system.standardize_bank_data(bank_data)
        finsys_clean = reco_system.standardize_finsys_data(finsys_data)
        
        # Capture debug output
        debug_output = io.StringIO()
        
        # Find the transaction for debugging
        matching_finsys = finsys_clean[
            (abs(finsys_clean['Amount'] - debug_amount) < amount_tolerance)
        ]
        
        if not matching_finsys.empty:
            st.sidebar.success(f"Found {len(matching_finsys)} finsys transaction(s)")
            
            for idx, finsys_row in matching_finsys.iterrows():
                st.sidebar.write(f"**Finsys Transaction #{idx}:**")
                st.sidebar.write(f"Date: {finsys_row['Date']}")
                st.sidebar.write(f"Amount: ₹{finsys_row['Amount']:,.2f}")
                st.sidebar.write(f"Description: {finsys_row['Description']}")
                
                # Look for bank matches
                bank_amount_matches = bank_clean[
                    (abs(bank_clean['Amount'] - finsys_row['Amount']) < amount_tolerance)
                ]
                
                st.sidebar.write(f"Bank transactions with same amount: {len(bank_amount_matches)}")
                
                if not bank_amount_matches.empty:
                    for bank_idx, bank_row in bank_amount_matches.iterrows():
                        date_diff = abs((bank_row['Date'] - finsys_row['Date']).days)
                        would_match = date_diff <= date_tolerance
                        
                        st.sidebar.write(f"**Bank Transaction #{bank_idx}:**")
                        st.sidebar.write(f"Date: {bank_row['Date']}")
                        st.sidebar.write(f"Amount: ₹{bank_row['Amount']:,.2f}")
                        st.sidebar.write(f"Date difference: {date_diff} days")
                        
                        if would_match:
                            st.sidebar.success("✅ Would match!")
                        else:
                            st.sidebar.error("❌ Date difference too large")
        else:
            st.sidebar.error("❌ No finsys transaction found with that amount")
    
    except Exception as e:
        st.sidebar.error(f"Debug error: {str(e)}")

if __name__ == "__main__":
    main()