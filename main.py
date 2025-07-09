import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import os
import re

class BankReconciliationSystem:
    """
    Advanced Bank Reconciliation System with flexible matching criteria
    """
    
    def __init__(self, date_tolerance=7, amount_tolerance=0.01):
        """
        Initialize the reconciliation system
        
        Args:
            date_tolerance: Days tolerance for date matching (default: ±2 days)
            amount_tolerance: Amount tolerance for matching (default: 0.01)
        """
        self.date_tolerance = date_tolerance
        self.amount_tolerance = amount_tolerance
        self.matches = []
        self.unmatched_bank = pd.DataFrame()
        self.unmatched_finsys = pd.DataFrame()
        
    def clean_text(self, text):
        """Clean and normalize text for comparison"""
        if pd.isna(text):
            return ""
        # Convert to string, uppercase, remove commas and extra spaces
        cleaned = str(text).upper().strip()
        cleaned = cleaned.replace(',', '')  # Remove all commas
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def calculate_name_similarity(self, name1, name2):
        """Calculate similarity between two names/descriptions"""
        name1_clean = self.clean_text(name1)
        name2_clean = self.clean_text(name2)
        
        if not name1_clean or not name2_clean:
            return 0.0
            
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, name1_clean, name2_clean).ratio()
        
        # Check for common words
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        if words1 and words2:
            common_words = len(words1.intersection(words2))
            total_words = len(words1.union(words2))
            word_similarity = common_words / total_words
            # Take the higher of sequence similarity or word similarity
            similarity = max(similarity, word_similarity)
        
        return similarity
    
    def load_and_prepare_data(self, bank_file, finsys_file):
        """
        Load and prepare data from bank statement and finsys files
        """
        print("📂 Loading data files...")
        
        try:
            # Load bank statement
            if bank_file.endswith('.xlsx'):
                bank_df = pd.read_excel(bank_file)
            else:
                bank_df = pd.read_csv(bank_file)
            print(f"✅ Bank statement loaded: {len(bank_df)} records")
            
            # Load finsys data
            if finsys_file.endswith('.xlsx'):
                finsys_df = pd.read_excel(finsys_file)
            else:
                finsys_df = pd.read_csv(finsys_file, encoding='utf-8')
            print(f"✅ Finsys data loaded: {len(finsys_df)} records")
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return None, None
        
        return bank_df, finsys_df
    
    def standardize_bank_data(self, bank_df):
        """Standardize bank statement data using DATE column"""
        print("🔄 Standardizing bank data...")
        
        bank_clean = bank_df.copy()
        
        # Use Value Dt column specifically for bank statement (DMY format)
        if 'Value Dt' in bank_clean.columns:
            bank_clean['Date'] = pd.to_datetime(bank_clean['Value Dt'], dayfirst=True, errors='coerce')
            # If the above fails, try other common DMY formats
            if bank_clean['Date'].isna().all():
                bank_clean['Date'] = pd.to_datetime(bank_clean['Value Dt'], format='%d/%m/%Y', errors='coerce')
            print("✅ Using Value Dt column for bank transactions (DMY format)")
            print(f"   Sample dates: {bank_clean['Date'].dropna().head(3).dt.strftime('%Y-%m-%d').tolist()}")
        else:
            print("❌ Value Dt column not found in bank statement")
            return None
        
        # Detect and clean amount columns - EXPANDED SEARCH
        debit_columns = ['WITHDRAWL', 'Withdrawl', 'WITHDRAWAL', 'DEBIT', 'DR', 'debit_amount', 'Debit', 'Withdrawal', 'WITHDRAWAL AMT', 'DEBIT AMT']
        credit_columns = ['DEPOSIT', 'Deposit Amt.', 'CREDIT', 'CR', 'credit_amount', 'Credit', 'Deposit', 'DEPOSIT AMT', 'CREDIT AMT']
        
        print(f"🔍 Available columns in bank data: {list(bank_clean.columns)}")
        
        bank_debit_col = None
        bank_credit_col = None
        
        for col in debit_columns:
            if col in bank_clean.columns:
                bank_debit_col = col
                break
                
        for col in credit_columns:
            if col in bank_clean.columns:
                bank_credit_col = col
                break
        
        # Clean amounts - remove commas, spaces, and convert to numeric
        if bank_debit_col:
            # Remove commas, spaces, and any other non-numeric characters except decimal point
            bank_clean['Debit'] = pd.to_numeric(
                bank_clean[bank_debit_col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('"', '').str.strip(), 
                errors='coerce'
            ).fillna(0)
            print(f"✅ Found debit column: {bank_debit_col}")
        else:
            bank_clean['Debit'] = 0
            print("⚠️ No debit column found, setting to 0")
            
        if bank_credit_col:
            # Remove commas, spaces, and any other non-numeric characters except decimal point
            bank_clean['Credit'] = pd.to_numeric(
                bank_clean[bank_credit_col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('"', '').str.strip(), 
                errors='coerce'
            ).fillna(0)
            print(f"✅ Found credit column: {bank_credit_col}")
        else:
            bank_clean['Credit'] = 0
            print("⚠️ No credit column found, setting to 0")
        
        # Create unified amount - CORRECTED LOGIC
        # Bank: DEPOSIT = money in (+), WITHDRAWL = money out (-)
        bank_clean['Amount'] = bank_clean['Credit'] - bank_clean['Debit']
        print(f"   Amount calculation: DEPOSIT - WITHDRAWL")
        print(f"   Sample amounts: {bank_clean['Amount'].head(3).tolist()}")
        
        # Detect description columns
        desc_columns = ['NARRATION', 'DESCRIPTION', 'PARTICULARS', 'description', 'narration']
        bank_desc_col = None
        for col in desc_columns:
            if col in bank_clean.columns:
                bank_desc_col = col
                break
        
        if bank_desc_col:
            bank_clean['Description'] = bank_clean[bank_desc_col].astype(str).str.replace(',', '')
            print(f"✅ Found description column: {bank_desc_col}")
        else:
            bank_clean['Description'] = ""
            print("⚠️ No description column found")
        
        # Detect reference columns
        ref_columns = ['REF', 'REF/CHQ NO', 'REFERENCE', 'CHQ_NO', 'reference']
        bank_ref_col = None
        for col in ref_columns:
            if col in bank_clean.columns:
                bank_ref_col = col
                break
        
        if bank_ref_col:
            bank_clean['Reference'] = bank_clean[bank_ref_col].astype(str).str.replace(',', '')
            print(f"✅ Found reference column: {bank_ref_col}")
        else:
            bank_clean['Reference'] = ""
            print("⚠️ No reference column found")
        
        # Add source identifier
        bank_clean['Source'] = 'BANK'
        bank_clean['Index_Original'] = bank_clean.index
        
        # Remove rows with invalid dates or zero amounts
        initial_count = len(bank_clean)
        bank_clean = bank_clean.dropna(subset=['Date'])
        bank_clean = bank_clean[bank_clean['Amount'] != 0]
        final_count = len(bank_clean)
        
        print(f"✅ Bank data standardized: {final_count} records (removed {initial_count - final_count} invalid/zero amount records)")
        return bank_clean
    
    def standardize_finsys_data(self, finsys_df):
        """Standardize finsys data using REFDATE column"""
        print("🔄 Standardizing finsys data...")
        
        finsys_clean = finsys_df.copy()
        
        # Use DATED column specifically for finsys data (DMY format)
        if 'DATED' in finsys_clean.columns:
            finsys_clean['Date'] = pd.to_datetime(finsys_clean['DATED'], dayfirst=True, errors='coerce')
            # If the above fails, try other common DMY formats
            if finsys_clean['Date'].isna().all():
                finsys_clean['Date'] = pd.to_datetime(finsys_clean['DATED'], format='%d/%m/%Y', errors='coerce')
            print("✅ Using DATED column for finsys transactions (DMY format)")
            print(f"   Sample dates: {finsys_clean['Date'].dropna().head(3).dt.strftime('%Y-%m-%d').tolist()}")
        else:
            print("❌ DATED column not found in finsys data")
            return None
        
        # Detect and clean amount columns - EXPANDED SEARCH
        debit_columns = ['DEBIT', 'DRAMT', 'DR_AMOUNT', 'debit_amount', 'Debit', 'DR', 'DEBIT AMT']
        credit_columns = ['CREDITS', 'CRAMT', 'CREDIT', 'CR_AMOUNT', 'credit_amount', 'Credit', 'CR', 'CREDIT AMT']
        
        print(f"🔍 Available columns in finsys data: {list(finsys_clean.columns)}")
        
        finsys_debit_col = None
        finsys_credit_col = None
        
        for col in debit_columns:
            if col in finsys_clean.columns:
                finsys_debit_col = col
                break
                
        for col in credit_columns:
            if col in finsys_clean.columns:
                finsys_credit_col = col
                break
        
        # Clean amounts - remove commas, spaces, and convert to numeric
        if finsys_debit_col:
            # Remove commas, spaces, and any other non-numeric characters except decimal point
            finsys_clean['Debit'] = pd.to_numeric(
                finsys_clean[finsys_debit_col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('"', '').str.strip(), 
                errors='coerce'
            ).fillna(0)
            print(f"✅ Found debit column: {finsys_debit_col}")
        else:
            finsys_clean['Debit'] = 0
            print("⚠️ No debit column found, setting to 0")
            
        if finsys_credit_col:
            # Remove commas, spaces, and any other non-numeric characters except decimal point
            finsys_clean['Credit'] = pd.to_numeric(
                finsys_clean[finsys_credit_col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('"', '').str.strip(), 
                errors='coerce'
            ).fillna(0)
            print(f"✅ Found credit column: {finsys_credit_col}")
        else:
            finsys_clean['Credit'] = 0
            print("⚠️ No credit column found, setting to 0")
        
        # Create unified amount - CORRECTED LOGIC
        # Finsys: DEBIT = money in (+), CREDITS = money out (-)  
        # To match bank: DEBIT - CREDITS (same as bank DEPOSIT - WITHDRAWL)
        finsys_clean['Amount'] = finsys_clean['Debit'] - finsys_clean['Credit']
        print(f"   Amount calculation: DEBIT - CREDITS")
        print(f"   Sample amounts: {finsys_clean['Amount'].head(3).tolist()}")
        
        # Detect description columns
        desc_columns = ['NARATION', 'NARRATION', 'DESCRIPTION', 'description', 'particulars']
        finsys_desc_col = None
        for col in desc_columns:
            if col in finsys_clean.columns:
                finsys_desc_col = col
                break
        
        if finsys_desc_col:
            finsys_clean['Description'] = finsys_clean[finsys_desc_col].astype(str).str.replace(',', '')
            print(f"✅ Found description column: {finsys_desc_col}")
        else:
            finsys_clean['Description'] = ""
            print("⚠️ No description column found")
        
        # Detect reference columns
        ref_columns = ['REFNUM', 'REF', 'REFERENCE', 'reference']
        finsys_ref_col = None
        for col in ref_columns:
            if col in finsys_clean.columns:
                finsys_ref_col = col
                break
        
        if finsys_ref_col:
            finsys_clean['Reference'] = finsys_clean[finsys_ref_col].astype(str).str.replace(',', '')
            print(f"✅ Found reference column: {finsys_ref_col}")
        else:
            finsys_clean['Reference'] = ""
            print("⚠️ No reference column found")
        
        # Add source identifier
        finsys_clean['Source'] = 'FINSYS'
        finsys_clean['Index_Original'] = finsys_clean.index
        
        # Remove rows with invalid dates or zero amounts
        initial_count = len(finsys_clean)
        finsys_clean = finsys_clean.dropna(subset=['Date'])
        finsys_clean = finsys_clean[finsys_clean['Amount'] != 0]
        final_count = len(finsys_clean)
        
        print(f"✅ Finsys data standardized: {final_count} records (removed {initial_count - final_count} invalid/zero amount records)")
        return finsys_clean
    
    def find_matches(self, bank_df, finsys_df, date_tolerance=None):
        """
        Find matches between bank and finsys transactions
        Simple matching: exact amount + date within tolerance
        """
        if date_tolerance is None:
            date_tolerance = self.date_tolerance
            
        print(f"🔍 Finding matches with ±{date_tolerance} days tolerance")
        
        matches = []
        used_finsys_indices = set()
        
        for bank_idx, bank_row in bank_df.iterrows():
            best_match = None
            
            for finsys_idx, finsys_row in finsys_df.iterrows():
                if finsys_idx in used_finsys_indices:
                    continue
                
                # Check if amounts match exactly (within tolerance)
                amount_diff = abs(bank_row['Amount'] - finsys_row['Amount'])
                if amount_diff > self.amount_tolerance:
                    continue
                
                # Check if dates are within tolerance
                date_diff = abs((bank_row['Date'] - finsys_row['Date']).days)
                if date_diff > date_tolerance:
                    continue
                
                # Found a match!
                best_match = finsys_idx
                break  # Take the first match found
            
            if best_match is not None:
                matches.append({
                    'Bank_Index': bank_row['Index_Original'],
                    'Finsys_Index': finsys_df.loc[best_match, 'Index_Original'],
                    'Bank_Date': bank_row['Date'],
                    'Finsys_Date': finsys_df.loc[best_match, 'Date'],
                    'Bank_Amount': bank_row['Amount'],
                    'Finsys_Amount': finsys_df.loc[best_match, 'Amount'],
                    'Bank_Description': bank_row['Description'],
                    'Finsys_Description': finsys_df.loc[best_match, 'Description'],
                    'Date_Difference': (finsys_df.loc[best_match, 'Date'] - bank_row['Date']).days,
                    'Amount_Difference': finsys_df.loc[best_match, 'Amount'] - bank_row['Amount']
                })
                used_finsys_indices.add(best_match)
        
        matches_df = pd.DataFrame(matches)
        print(f"✅ Found {len(matches)} matches")
        
        return matches_df
    
    def debug_specific_transaction(self, bank_df, finsys_df, target_amount, target_date_str):
        """
        Debug a specific transaction to see why it's not matching
        """
        print(f"\n🔍 DEBUGGING TRANSACTION: Amount {target_amount} on {target_date_str}")
        print("="*80)
        
        # Find the finsys transaction
        matching_finsys = finsys_df[
            (abs(finsys_df['Amount'] - target_amount) < 0.01)
        ]
        
        if matching_finsys.empty:
            print("❌ Finsys transaction not found with that amount")
            return
            
        print(f"📊 Found {len(matching_finsys)} finsys transaction(s) with amount {target_amount}")
        
        for idx, finsys_row in matching_finsys.iterrows():
            print(f"\n🔸 Finsys Transaction #{idx}:")
            print(f"   Date: {finsys_row['Date']} (parsed from DATED)")
            print(f"   Amount: {finsys_row['Amount']}")
            print(f"   Description: {finsys_row['Description']}")
            
            # Look for potential bank matches
            print(f"\n🔍 Looking for bank matches...")
            
            # Check for exact amount matches
            bank_amount_matches = bank_df[
                (abs(bank_df['Amount'] - finsys_row['Amount']) < 0.01)
            ]
            
            print(f"   Bank transactions with same amount: {len(bank_amount_matches)}")
            
            if not bank_amount_matches.empty:
                for bank_idx, bank_row in bank_amount_matches.iterrows():
                    date_diff = abs((bank_row['Date'] - finsys_row['Date']).days)
                    score, factors = self.calculate_match_score(bank_row, finsys_row)
                    
                    print(f"\n   🏦 Bank Transaction #{bank_idx}:")
                    print(f"      Date: {bank_row['Date']} (parsed from Value Dt)")
                    print(f"      Amount: {bank_row['Amount']}")
                    print(f"      Description: {bank_row['Description']}")
                    print(f"      Date difference: {date_diff} days")
                    print(f"      Match score: {score:.3f}")
                    print(f"      Match factors: {' | '.join(factors)}")
                    print(f"      Would match: {'✅ YES' if score >= 0.7 else '❌ NO'}")
            
            # Check for amount matches within date range
            finsys_date = finsys_row['Date']
            date_range_start = finsys_date - timedelta(days=self.date_tolerance)
            date_range_end = finsys_date + timedelta(days=self.date_tolerance)
            
            bank_date_range_matches = bank_df[
                (bank_df['Date'] >= date_range_start) &
                (bank_df['Date'] <= date_range_end)
            ]
            
            print(f"\n   📅 Bank transactions in date range ({date_range_start.date()} to {date_range_end.date()}): {len(bank_date_range_matches)}")
            
            if not bank_date_range_matches.empty:
                print("      Date range transactions:")
                for bank_idx, bank_row in bank_date_range_matches.head(5).iterrows():
                    amount_diff = abs(bank_row['Amount'] - finsys_row['Amount'])
                    print(f"         #{bank_idx}: {bank_row['Date'].date()}, ₹{bank_row['Amount']}, diff: ₹{amount_diff:.2f}")
        
        print("="*80)
    
    def reconcile(self, bank_file, finsys_file, output_dir='output', debug_transaction=None):
        """
        Perform complete bank reconciliation
        debug_transaction: tuple of (amount, date_string) to debug specific transaction
        """
        print("🏦 Starting Simple Bank Reconciliation System")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        bank_df, finsys_df = self.load_and_prepare_data(bank_file, finsys_file)
        if bank_df is None or finsys_df is None:
            return
        
        # Standardize data
        bank_clean = self.standardize_bank_data(bank_df)
        finsys_clean = self.standardize_finsys_data(finsys_df)
        
        if bank_clean is None or finsys_clean is None:
            print("❌ Failed to standardize data")
            return
        
        print(f"\n📊 Data Summary:")
        print(f"Bank transactions: {len(bank_clean)}")
        print(f"Finsys transactions: {len(finsys_clean)}")
        
        # Debug specific transaction if requested
        if debug_transaction:
            amount, date_str = debug_transaction
            self.debug_specific_transaction(bank_clean, finsys_clean, amount, date_str)
        
        # Find matches (simple: amount + date only)
        matches_df = self.find_matches(bank_clean, finsys_clean)
        
        # Identify unmatched transactions
        if not matches_df.empty:
            matched_bank_indices = matches_df['Bank_Index'].tolist()
            matched_finsys_indices = matches_df['Finsys_Index'].tolist()
            
            self.unmatched_bank = bank_df[~bank_df.index.isin(matched_bank_indices)].copy()
            self.unmatched_finsys = finsys_df[~finsys_df.index.isin(matched_finsys_indices)].copy()
        else:
            self.unmatched_bank = bank_df.copy()
            self.unmatched_finsys = finsys_df.copy()
        
        # Generate reports
        self.generate_reports(matches_df, bank_clean, finsys_clean, output_dir)
        
        return matches_df, self.unmatched_bank, self.unmatched_finsys
    
    def generate_reports(self, matches_df, bank_df, finsys_df, output_dir):
        """Generate comprehensive reconciliation reports"""
        print("\n📋 Generating Reports...")
        
        # Summary statistics
        total_bank = len(bank_df)
        total_finsys = len(finsys_df)
        total_matches = len(matches_df) if not matches_df.empty else 0
        
        bank_matched_amount = matches_df['Bank_Amount'].sum() if not matches_df.empty else 0
        finsys_matched_amount = matches_df['Finsys_Amount'].sum() if not matches_df.empty else 0
        
        unmatched_bank_count = len(self.unmatched_bank)
        unmatched_finsys_count = len(self.unmatched_finsys)
        
        unmatched_bank_amount = self.unmatched_bank['DEPOSIT'].fillna(0).sum() - self.unmatched_bank['WITHDRAWL'].fillna(0).sum() if 'DEPOSIT' in self.unmatched_bank.columns else 0
        unmatched_finsys_amount = self.unmatched_finsys['DEBIT'].fillna(0).sum() - self.unmatched_finsys['CREDITS'].fillna(0).sum() if 'DEBIT' in self.unmatched_finsys.columns else 0
        
        # Print summary
        print("\n" + "="*60)
        print("📊 RECONCILIATION SUMMARY")
        print("="*60)
        print(f"Total Bank Transactions: {total_bank:,}")
        print(f"Total Finsys Transactions: {total_finsys:,}")
        print(f"Matched Transactions: {total_matches:,}")
        print(f"Match Rate: {(total_matches/total_bank*100):.1f}%" if total_bank > 0 else "0%")
        print(f"\nMatched Amount (Bank): ₹{bank_matched_amount:,.2f}")
        print(f"Matched Amount (Finsys): ₹{finsys_matched_amount:,.2f}")
        print(f"Amount Difference: ₹{(finsys_matched_amount - bank_matched_amount):,.2f}")
        print(f"\nUnmatched Bank Transactions: {unmatched_bank_count:,} (₹{unmatched_bank_amount:,.2f})")
        print(f"Unmatched Finsys Transactions: {unmatched_finsys_count:,} (₹{unmatched_finsys_amount:,.2f})")
        print("="*60)
        
        # Save detailed reports
        if not matches_df.empty:
            matches_df.to_csv(f'{output_dir}/matched_transactions.csv', index=False)
            print(f"✅ Matched transactions saved: {output_dir}/matched_transactions.csv")
        
        if not self.unmatched_bank.empty:
            self.unmatched_bank.to_csv(f'{output_dir}/unmatched_bank.csv', index=False)
            print(f"✅ Unmatched bank transactions saved: {output_dir}/unmatched_bank.csv")
        
        if not self.unmatched_finsys.empty:
            self.unmatched_finsys.to_csv(f'{output_dir}/unmatched_finsys.csv', index=False)
            print(f"✅ Unmatched finsys transactions saved: {output_dir}/unmatched_finsys.csv")
        
        # Create summary report
        summary_data = {
            'Metric': [
                'Total Bank Transactions', 'Total Finsys Transactions', 'Matched Transactions',
                'Match Rate (%)', 'Unmatched Bank', 'Unmatched Finsys',
                'Matched Bank Amount', 'Matched Finsys Amount', 'Amount Difference'
            ],
            'Value': [
                total_bank, total_finsys, total_matches,
                f"{(total_matches/total_bank*100):.1f}" if total_bank > 0 else "0",
                unmatched_bank_count, unmatched_finsys_count,
                f"₹{bank_matched_amount:,.2f}", f"₹{finsys_matched_amount:,.2f}",
                f"₹{(finsys_matched_amount - bank_matched_amount):,.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/reconciliation_summary.csv', index=False)
        print(f"✅ Summary report saved: {output_dir}/reconciliation_summary.csv")
        
        print(f"\n🎉 Reconciliation completed! Check '{output_dir}' folder for all reports.")


# Example usage
def run_reconciliation():
    """Example function to run the reconciliation"""
    
    # Initialize the reconciliation system
    reco_system = BankReconciliationSystem(
        date_tolerance=7,  # ±2 days as requested
        amount_tolerance=0.01  # 1 paisa tolerance
    )
    
    # Run reconciliation
    # Replace with your actual file paths
    bank_file = "data/bank_statement.xlsx"  # Contains DATE column (MDY format)
    finsys_file = "data/finsys_data.csv"    # Contains REFDATE column (DMY format)
    
    matches, unmatched_bank, unmatched_finsys = reco_system.reconcile(
        bank_file=bank_file,
        finsys_file=finsys_file,
        output_dir='reconciliation_output'
    )
    
    return matches, unmatched_bank, unmatched_finsys

# Example usage with debugging
def run_reconciliation_with_debug():
    """Example function to run reconciliation with debugging"""
    
    # Initialize the reconciliation system
    reco_system = BankReconciliationSystem(
        date_tolerance=7,  # ±2 days as requested
        amount_tolerance=0.01  # 1 paisa tolerance
    )
    
    # Run reconciliation with debugging for specific transaction
    # Replace with your actual file paths
    bank_file = "data/bank_statement.xlsx"  # Contains DATE column (MDY format)
    finsys_file = "data/finsys_data.csv"    # Contains REFDATE column (DMY format)
    
    matches, unmatched_bank, unmatched_finsys = reco_system.reconcile(
        bank_file=bank_file,
        finsys_file=finsys_file,
        output_dir='reconciliation_output',
        debug_transaction=(260121.37, "23/05/2025")  # Debug the specific transaction
    )
    
    return matches, unmatched_bank, unmatched_finsys

if __name__ == "__main__":
    # DEBUG MODE: Uncomment the line below to debug a specific transaction
    # Change the amount and date to match your problematic transaction
    
    # For INOX AIR PRODUCTS transaction:
    # run_reconciliation_with_debug_inox()
    
    # For SAI HARDWARE STORE transaction:
    # run_reconciliation_with_debug_sai()
    
    # For normal reconciliation without debugging:
    run_reconciliation()

# Debug functions for specific transactions
def run_reconciliation_with_debug_inox():
    """Debug the INOX AIR PRODUCTS transaction"""
    
    reco_system = BankReconciliationSystem(
        date_tolerance=7,
        amount_tolerance=0.01
    )
    
    bank_file = "data/bank_statement.xlsx"
    finsys_file = "data/ledger.csv"
    
    print("🔍 DEBUGGING INOX AIR PRODUCTS TRANSACTION")
    print("="*60)
    
    matches, unmatched_bank, unmatched_finsys = reco_system.reconcile(
        bank_file=bank_file,
        finsys_file=finsys_file,
        output_dir='reconciliation_output',
        debug_transaction=(4986888.0, "01/04/2025")
    )
    
    return matches, unmatched_bank, unmatched_finsys

def run_reconciliation_with_debug_sai():
    """Debug the SAI HARDWARE STORE transaction"""
    
    reco_system = BankReconciliationSystem(
        date_tolerance=7,
        amount_tolerance=0.01
    )
    
    bank_file = "data/bank_statement.xlsx"
    finsys_file = "data/ledger.csv"
    
    print("🔍 DEBUGGING SAI HARDWARE STORE TRANSACTION")
    print("="*60)
    
    matches, unmatched_bank, unmatched_finsys = reco_system.reconcile(
        bank_file=bank_file,
        finsys_file=finsys_file,
        output_dir='reconciliation_output',
        debug_transaction=(-2378.0, "17/05/2025")  # Net amount for SAI transaction
    )
    
    return matches, unmatched_bank, unmatched_finsys