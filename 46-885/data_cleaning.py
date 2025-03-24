import pandas as pd
import numpy as np
import os
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from transformers import pipeline

# Set data directory
output_dir = "/Users/mlwu/Documents/CMU/Data Exploration and Visualization/Final Project/Data"
os.makedirs(output_dir, exist_ok=True)

# Function to load CSV with encoding handling and fix delimiter issues
def load_csv(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    print(f"Loaded {filepath} with columns:\n{df.columns.tolist()}\n")
    return df

# Function to clean column names
def clean_column_names(df):
    # Convert column names to lowercase, replace spaces with underscores, and strip whitespace
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    
    # Remove parentheses and their contents
    df.columns = df.columns.str.replace(r'\([^)]*\)', '', regex=True)
    
    # Remove special characters
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    
    # Replace multiple underscores with single underscore
    df.columns = df.columns.str.replace(r'_+', '_', regex=True)
    
    # Strip trailing underscores
    df.columns = df.columns.str.replace(r'_$', '', regex=True)
    
    return df

# Function to convert percentage strings to decimal values
def convert_percentages_to_decimal(df):
    for col in df.columns:
        if col in df and pd.api.types.is_object_dtype(df[col]):  # More robust type check
            # Check if column contains at least some percentage values
            if df[col].astype(str).str.contains('%', na=False).any():
                # Convert only rows that contain percentage signs
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)

                # Try converting to float
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                
    return df

# Function to expand abbreviated numbers (K, M, B)
def expand_abbreviated_numbers(df):
    for col in df.columns:
        if col in df and pd.api.types.is_object_dtype(df[col]):  # Ensuring valid column access
            # Check if column contains abbreviated numbers
            if df[col].astype(str).str.contains(r'[KkMmBb]', na=False).any():
                
                def expand_value(val):
                    if pd.isna(val) or not isinstance(val, str):
                        return val
                    
                    val = val.strip()
                    
                    # Handle K (thousands)
                    if val.endswith(('K', 'k')):
                        try:
                            return float(val[:-1]) * 1000
                        except ValueError:
                            return val
                    
                    # Handle M (millions)
                    elif val.endswith(('M', 'm')):
                        try:
                            return float(val[:-1]) * 1000000
                        except ValueError:
                            return val
                    
                    # Handle B (billions)
                    elif val.endswith(('B', 'b')):
                        try:
                            return float(val[:-1]) * 1000000000
                        except ValueError:
                            return val
                    
                    return val
                
                # Apply the conversion function safely
                df[col] = df[col].apply(expand_value)

                # Attempt to convert column to numeric if possible
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
    return df

# Function to remove duplicate columns
def remove_duplicate_columns(df):
    # Identify duplicate columns
    duplicate_cols = []
    cols = df.columns.tolist()
    
    # Check each pair of columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            # If column names are similar (case insensitive)
            if cols[i].lower() == cols[j].lower():
                duplicate_cols.append(cols[j])
            # Check for similar names with _x or _y suffixes
            elif cols[i].endswith('_x') and cols[j].endswith('_y'):
                base_i = cols[i][:-2]
                base_j = cols[j][:-2]
                if base_i == base_j:
                    duplicate_cols.append(cols[j])
    
    # Drop duplicate columns
    if duplicate_cols:
        print(f"Removing duplicate columns: {duplicate_cols}")
        df = df.drop(columns=duplicate_cols, errors='ignore')
    
    return df

# Load datasets
df_financial = load_csv(f"{output_dir}/Cleaned_Financial_Data.csv")
df_ecommerce = load_csv(f"{output_dir}/e-commerce.csv")
df_g2 = load_csv(f"{output_dir}/g2.csv")
df_snp500 = load_csv(f"{output_dir}/snp500.csv")
df_snp500_f = load_csv(f"{output_dir}/snp500_financials.csv")
df_stocks = load_csv(f"{output_dir}/stocks.csv")
df_trustpilot = load_csv(f"{output_dir}/trustpilot.csv")
df_customer_demographics = load_csv(f"{output_dir}/customer_demographics.csv")
df_customer_cases = load_csv(f"{output_dir}/customer_cases.csv")
df_fp_dataset_raw = load_csv(f"{output_dir}/fp_dataset_raw.csv")

# Clean and standardize all dataframes
all_dfs = [df_financial, df_ecommerce, df_g2, df_snp500, df_snp500_f, 
           df_stocks, df_trustpilot, df_customer_demographics, df_customer_cases]

for i, df in enumerate(all_dfs):
    # Store original column mapping for reference
    original_columns = df.columns.tolist()
    
    # Clean column names
    df = clean_column_names(df)
    
    # Convert percentages to decimal
    df = convert_percentages_to_decimal(df)
    
    # Expand abbreviated numbers
    df = expand_abbreviated_numbers(df)
    
    # Update the dataframe in the list
    all_dfs[i] = df
    
    # Print column mapping for reference
    print(f"Original columns: {original_columns[:5]}{'...' if len(original_columns) > 5 else ''}")
    print(f"Cleaned columns: {df.columns.tolist()[:5]}{'...' if len(df.columns) > 5 else ''}\n")

# Reassign cleaned dataframes
[df_financial, df_ecommerce, df_g2, df_snp500, df_snp500_f, 
 df_stocks, df_trustpilot, df_customer_demographics, df_customer_cases] = all_dfs

# Determine the common merge column for S&P 500 datasets
# Find company/symbol column across datasets
company_col_in_snp500 = None
symbol_col_in_snp500f = None

# Look for company column in snp500
for col in df_snp500.columns:
    if 'company' in col or 'name' in col:
        company_col_in_snp500 = col
        break

# Look for symbol column in snp500f
for col in df_snp500_f.columns:
    if 'symbol' in col or 'ticker' in col:
        symbol_col_in_snp500f = col
        break

if not company_col_in_snp500:
    print("Cannot find company column in S&P 500 dataset")
    company_col_in_snp500 = 'company'  # Fallback

if not symbol_col_in_snp500f:
    print("Cannot find symbol column in S&P 500 financials dataset")
    symbol_col_in_snp500f = 'symbol'  # Fallback

# Merge S&P 500 datasets
print(f"Merging S&P 500 datasets on {company_col_in_snp500} and {symbol_col_in_snp500f}")
df_snp500_merged = pd.merge(
    df_snp500, 
    df_snp500_f, 
    left_on=company_col_in_snp500, 
    right_on=symbol_col_in_snp500f, 
    how="left"
)

# Remove duplicate columns from merged dataset
df_snp500_merged = remove_duplicate_columns(df_snp500_merged)

# Ensure company column exists
company_col = 'company'
if company_col not in df_snp500_merged.columns:
    # Find a suitable company column
    potential_cols = [col for col in df_snp500_merged.columns if 'company' in col or 'name' in col]
    if potential_cols:
        print(f"Using {potential_cols[0]} as company column")
        df_snp500_merged.rename(columns={potential_cols[0]: company_col}, inplace=True)
    else:
        print("Cannot find company column in merged S&P 500 dataset")

# Prepare financial data
financial_cols = [
    'company', 
    'market_capitalization', 
    'income', 
    'revenue', 
    'performance'
]

df_financial_cleaned = df_financial.copy()
# Ensure all required columns exist
for col in financial_cols:
    if not any(c.startswith(col) for c in df_financial_cleaned.columns):
        print(f"Cannot find {col} column in financial dataset")

# Convert Transaction Date to datetime
date_col = next((col for col in df_customer_demographics.columns if 'date' in col), None)
if date_col:
    df_customer_demographics[date_col] = pd.to_datetime(df_customer_demographics[date_col], errors='coerce')

# Compute Churn Rate if transaction_type exists
transaction_type_col = next((col for col in df_customer_demographics.columns if 'type' in col), None)
if transaction_type_col:
    df_customer_demographics['churned'] = df_customer_demographics[transaction_type_col].apply(
        lambda x: 1 if isinstance(x, str) and 'reduction' in x.lower() else 0
    )
    
    # Calculate churn rate by date
    if date_col:
        churn_rate = df_customer_demographics.groupby(date_col)['churned'].mean().rename('churn_rate').reset_index()
        df_final = df_customer_demographics.merge(churn_rate, on=date_col, how='left')
    else:
        df_final = df_customer_demographics.copy()
else:
    df_final = df_customer_demographics.copy()

# Ensure consistent customer ID column for merging
customer_id_col_demographics = next((col for col in df_final.columns if 'customer' in col and 'id' in col), None)
customer_id_col_cases = next((col for col in df_customer_cases.columns if 'customer' in col and 'id' in col), None)

if customer_id_col_demographics and customer_id_col_cases:
    # Standardize customer case column names
    df_customer_cases_cleaned = df_customer_cases.copy()
    
    # Find relevant columns
    case_id_col = next((col for col in df_customer_cases.columns if 'case' in col and 'id' in col), 'case_id')
    case_date_col = next((col for col in df_customer_cases.columns if 'date' in col), 'case_date')
    channel_col = next((col for col in df_customer_cases.columns if 'channel' in col), 'support_channel')
    reason_col = next((col for col in df_customer_cases.columns if 'reason' in col), 'case_reason')
    
    # Rename columns to standardized names
    column_mapping = {
        case_id_col: 'case_id',
        case_date_col: 'case_date',
        customer_id_col_cases: 'customer_id',
        channel_col: 'support_channel',
        reason_col: 'case_reason'
    }
    
    df_customer_cases_cleaned.rename(columns=column_mapping, inplace=True)
    
    # Rename customer ID column in demographics to match
    df_final.rename(columns={customer_id_col_demographics: 'customer_id'}, inplace=True)
    
    # Merge customer demographics and cases
    df_customer_merged = pd.merge(df_final, df_customer_cases_cleaned, on='customer_id', how='left')
else:
    print("Warning: Customer ID column is missing from one of the datasets. Skipping customer case merge.")
    df_customer_merged = df_final.copy()

# Perform NLP sentiment analysis if transformer pipeline is available
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    def analyze_sentiment(df, text_column, score_column):
        if text_column not in df.columns:
            return None
        df = df.dropna(subset=[text_column]).copy()
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        all_scores = []
        
        for i in range(0, len(df), batch_size):
            batch = df[text_column].iloc[i:i+batch_size].tolist()
            sentiments = sentiment_pipeline(batch)
            batch_scores = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments]
            all_scores.extend(batch_scores)
        
        df['sentiment_score'] = all_scores
        return df[[score_column, 'sentiment_score']]

    # Find review text column in trustpilot
    review_col_trustpilot = next((col for col in df_trustpilot.columns if 'review' in col), None)
    score_col_trustpilot = next((col for col in df_trustpilot.columns if 'score' in col), 'trust_score')
    
    if review_col_trustpilot:
        df_trustpilot_sentiment = analyze_sentiment(df_trustpilot, review_col_trustpilot, score_col_trustpilot)
    else:
        df_trustpilot_sentiment = None
    
    # Find pros list column in g2
    pros_col_g2 = next((col for col in df_g2.columns if 'pros' in col), None)
    rating_col_g2 = next((col for col in df_g2.columns if 'rating' in col), 'rating')
    
    if pros_col_g2:
        df_g2_sentiment = analyze_sentiment(df_g2, pros_col_g2, rating_col_g2)
    else:
        df_g2_sentiment = None
        
except Exception as e:
    print(f"Skipping sentiment analysis due to error: {str(e)}")
    df_trustpilot_sentiment = None
    df_g2_sentiment = None

# Prepare for final merge
if company_col in df_snp500_merged.columns and company_col in df_financial_cleaned.columns:
    # Merge financial data with S&P 500 data
    merged_fp_snp500 = pd.merge(df_financial_cleaned, df_snp500_merged, on=company_col, how='left')
    
    # Remove duplicate columns
    merged_fp_snp500 = remove_duplicate_columns(merged_fp_snp500)
    
    # Define the column renaming mapping
    column_renaming_mapping = {
        'company': 'Company',
        'market_capitalization_x': 'Market Capitalization',
        'income_x': 'Income (TTM)',
        'revenue_x': 'Revenue (TTM)',
        'performance': 'Performance (Year)',
        'major_index_membership': 'Major index membership',
        'book_value_per_share': 'Book value per share (mrq)',
        'cash_per_share': 'Cash per share (mrq)',
        'dividend': 'Dividend (annual)',
        'dividend_yield_x': 'Dividend Yield (annual)',
        'full_time_employees': 'Full time employees',
        'stock_has_options_trading_on_a_market_exchange': 'Stock has options trading on a market exchange',
        'stock_available_to_sell_short': 'Stock available to sell short',
        'analysts_mean_recommendation': "Analysts' mean recommendation (1=Buy 5=Sell)",
        'pricetoearnings': 'Price-to-Earnings (ttm)',
        'forward_pricetoearnings': 'Forward Price-to-Earnings (next fiscal year)',
        'pricetoearningstogrowth': 'Price-to-Earnings-to-Growth',
        'pricetosales': 'Price-to-Sales (ttm)',
        'pricetobook': 'Price-to-Book (mrq)',
        'price_to_cash_per_share': 'Price to cash per share (mrq)',
        'price_to_free_cash_flow': 'Price to Free Cash Flow (ttm)',
        'quick_ratio': 'Quick Ratio (mrq)',
        'current_ratio': 'Current Ratio (mrq)',
        'total_debt_to_equity': 'Total Debt to Equity (mrq)',
        'long_term_debt_to_equity': 'Long Term Debt to Equity (mrq)',
        'distance_from_20day_simple_moving_average': 'Distance from 20-Day Simple Moving Average',
        'diluted_eps': 'Diluted EPS (ttm)',
        'eps_estimate_for_next_year': 'EPS estimate for next year',
        'eps_estimate_for_next_quarter': 'EPS estimate for next quarter',
        'eps_growth_this_year': 'EPS growth this year',
        'eps_growth_next_year': 'EPS growth next year',
        'long_term_annual_growth_estimate': 'Long term annual growth estimate (5 years)',
        'annual_eps_growth_past_5_years': 'Annual EPS growth past 5 years',
        'annual_sales_growth_past_5_years': 'Annual sales growth past 5 years',
        'quarterly_revenue_growth': 'Quarterly revenue growth (YoY)',
        'quarterly_earnings_growth': 'Quarterly earnings growth (YoY)',
        'earnings_datebrbrbmo_before_market_openbramc_after_market_close': 'Earnings date',
        'distance_from_50day_simple_moving_average': 'Distance from 50-Day Simple Moving Average',
        'insider_ownership': 'Insider ownership',
        'insider_transactions': 'Insider transactions (6-Month change in Insider Ownership)',
        'institutional_ownership': 'Institutional ownership',
        'institutional_transactions': 'Institutional transactions (3-Month change in Institutional Ownership)',
        'return_on_assets': 'Return on Assets (ttm)',
        'return_on_equity': 'Return on Equity (ttm)',
        'return_on_investment': 'Return on Investment (ttm)',
        'gross_margin': 'Gross Margin (ttm)',
        'operating_margin': 'Operating Margin (ttm)',
        'net_profit_margin': 'Net Profit Margin (ttm)',
        'dividend_payout_ratio': 'Dividend Payout Ratio (ttm)',
        'distance_from_200day_simple_moving_average': 'Distance from 200-Day Simple Moving Average',
        'shares_outstanding': 'Shares outstanding',
        'shares_float': 'Shares float',
        'short_interest_share_ratio': 'Short interest share / ratio',
        'short_interest': 'Short interest',
        'analysts_mean_target_price': "Analysts' mean target price",
        '52week_trading_range': '52-Week trading range',
        'distance_from_52week_high': 'Distance from 52-Week High',
        'distance_from_52week_low': 'Distance from 52-Week Low',
        'relative_strength_index': 'Relative Strength Index',
        'relative_volume': 'Relative volume',
        'average_volume': 'Average volume (3 month)',
        'volume': 'Volume',
        'beta': 'Beta',
        'average_true_range': 'Average True Range (14)',
        'volatility': 'Volatility (Week, Month)',
        'previous_close': 'Previous close',
        'current_stock_price': 'Current stock price',
        'symbol': 'Name',
        'name': 'Name',
        'sector': 'Sector',
        'price': 'Price',
        'priceearnings': 'Price/Earnings',
        'earningsshare': 'Earnings/Share',
        '52_week_low': '52 Week Low',
        '52_week_high': '52 Week High',
        'market_cap': 'Market Cap',
        'ebitda': 'EBITDA',
        'pricesales': 'Price/Sales',
        'pricebook': 'Price/Book',
        'sec_filings': 'SEC Filings'
    }
    
    # Make a list of columns to remove before renaming (to avoid conflicts)
    columns_to_remove = ['unnamed_0']
    merged_fp_snp500.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Apply the renaming mapping to our dataset
    merged_fp_snp500 = merged_fp_snp500.rename(columns=column_renaming_mapping)
    
    # Add customer data columns with NaN values
    customer_columns = {
        'CustomerID': np.nan,
        'Transaction Type': np.nan,
        'Transaction Date': np.nan,
        'Subscription Type': np.nan,
        'Subscription Price': np.nan,
        'Gender': np.nan,
        'Age Group': np.nan,
        'Customer Country': np.nan,
        'Referral Type': np.nan, 
        'Churned': np.nan,
        'Case ID': np.nan,
        'Case Date': np.nan,
        'Support Channel': np.nan,
        'Case Reason': np.nan,
        'Trustpilot Score': np.nan,
        'Sentiment Score': np.nan,
        'G2 Rating': np.nan
    }
    
    # Add customer columns only if they don't already exist
    for col_name, default_value in customer_columns.items():
        if col_name not in merged_fp_snp500.columns:
            merged_fp_snp500[col_name] = default_value
    
    # Save the final cleaned dataset
    final_filepath = os.path.join(output_dir, "final_fp_dataset_cleaned.csv")
    merged_fp_snp500.to_csv(final_filepath, index=False)
    print(f"\nFinal cleaned dataset saved to: {final_filepath}")
else:
    print(f"Cannot merge datasets: 'company' column missing from one or both datasets")
    print(f"Financial cleaned columns: {df_financial_cleaned.columns.tolist()}")
    print(f"S&P 500 merged columns: {df_snp500_merged.columns.tolist()[:10]} ...")

# Set up word cloud output directory
wordcloud_dir = os.path.join(output_dir, "wordclouds")
os.makedirs(wordcloud_dir, exist_ok=True)

# Define a list of words to remove (in addition to built-in stopwords)
# custom_stopwords = STOPWORDS.union({"text", "count", "star", "customer", "features", "support", "use"})

# Simpler preprocessing function that keeps more words
def preprocess_text(series_list):
    """Preprocess multiple text columns with minimal filtering"""
    text_data = []
    
    for series in series_list:
        if series is not None and not series.empty:
            # Only convert to lowercase and strip whitespace, keep more words
            text_data.append(' '.join(series.dropna().astype(str).str.lower().str.strip()))
    
    # Join all text data
    text = ' '.join(text_data)
    
    # Debugging: print the first 100 characters of text
    print(f"Preprocessed text sample (first 100 chars): {text[:100]}")
    print(f"Total text length: {len(text)} characters")
    
    return text

def generate_and_save_wordcloud(df, columns, title, filename):
    """Generate and save a word cloud with more lenient preprocessing"""
    # First check if columns exist in the dataframe
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        print(f"No valid columns found for {title} word cloud. Available columns: {df.columns.tolist()[:5]}...")
        return
    
    print(f"Generating word cloud for {title} using columns: {valid_columns}")
    
    # Print sample data from each column to debug
    for col in valid_columns:
        sample = df[col].dropna().astype(str).iloc[:3].tolist() if not df[col].empty else []
        print(f"Sample from {col}: {sample}")
    
    text = preprocess_text([df[col] for col in valid_columns])
    
    if not text or len(text.strip()) < 5:  # Check if text is essentially empty
        print(f"Skipping word cloud for {title} - insufficient text data after preprocessing")
        return
    
    try:
        # Create word cloud with minimal filtering
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            min_word_length=2,  # Include shorter words
            collocations=False,  # Avoid duplicating similar words
            max_words=200       # Allow more words
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(os.path.join(wordcloud_dir, filename), bbox_inches='tight')
        plt.close()
        print(f"✅ Word cloud saved: {filename}")
        
    except ValueError as e:
        print(f"Error generating word cloud for {title}: {str(e)}")
        print(f"Text length: {len(text)}, First 200 chars: {text[:200]}")

# Load the datasets
def load_and_process_datasets():
    # Load datasets
    try:
        df_trustpilot = pd.read_csv(f"{output_dir}/trustpilot.csv")
        print(f"Loaded trustpilot.csv with {len(df_trustpilot)} rows")
        
        df_g2 = pd.read_csv(f"{output_dir}/g2.csv")
        print(f"Loaded g2.csv with {len(df_g2)} rows")
        
        # Find relevant text columns by checking column names
        # Print all column names to help with debugging
        print("Trustpilot columns:", df_trustpilot.columns.tolist())
        print("G2 columns:", df_g2.columns.tolist())
        
        # Try to find review-related columns in Trustpilot data
        review_cols_trustpilot = [col for col in df_trustpilot.columns 
                                 if any(term in col.lower() for term in ['review', 'comment', 'text', 'feedback'])]
        
        # If no review columns found with specific terms, use any text-like column
        if not review_cols_trustpilot:
            # Look for columns that might contain text (object type with longer values)
            for col in df_trustpilot.select_dtypes(include=['object']).columns:
                # Check sample values to see if they look like text
                sample = df_trustpilot[col].dropna().astype(str).iloc[:5]
                if any(len(str(val)) > 20 for val in sample):  # Assume text columns have some longer values
                    review_cols_trustpilot.append(col)
                    print(f"Using column '{col}' as potential review text")
        
        # Find pros/cons columns in G2 data
        pros_cols_g2 = [col for col in df_g2.columns if 'pros' in col.lower() or 'positive' in col.lower()]
        cons_cols_g2 = [col for col in df_g2.columns if 'cons' in col.lower() or 'negative' in col.lower()]
        
        # If no specific pros/cons columns found, look for any potential text columns
        if not pros_cols_g2 and not cons_cols_g2:
            text_cols_g2 = []
            for col in df_g2.select_dtypes(include=['object']).columns:
                sample = df_g2[col].dropna().astype(str).iloc[:5]
                if any(len(str(val)) > 20 for val in sample):
                    text_cols_g2.append(col)
                    print(f"Using column '{col}' as potential G2 review text")
            
            # Generate combined word cloud for G2 if specific pros/cons not found
            if text_cols_g2:
                generate_and_save_wordcloud(df_g2, text_cols_g2, 'G2 Reviews', 'g2_reviews.png')
        
        # Generate individual word clouds if specific columns found
        if review_cols_trustpilot:
            generate_and_save_wordcloud(df_trustpilot, review_cols_trustpilot, 'Trustpilot Reviews', 'trustpilot_reviews.png')
        
        if pros_cols_g2:
            generate_and_save_wordcloud(df_g2, pros_cols_g2, 'G2 Pros', 'g2_pros.png')
        
        if cons_cols_g2:
            generate_and_save_wordcloud(df_g2, cons_cols_g2, 'G2 Cons', 'g2_cons.png')
            
    except Exception as e:
        print(f"Error during word cloud generation: {str(e)}")

print("\nData processing completed successfully!")
