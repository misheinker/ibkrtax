import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
import os
import glob
import numpy as np

# Configuration
BASE_CURRENCY = 'ILS'
TARGET_CURRENCY = 'USD'
# FRED Series for Israel CPI Growth Rate (Monthly, Not Seasonally Adjusted)
# We will reconstruct an index from this.
CPI_SERIES_ID = 'CPALTT01ILM657N' 

def parse_ibkr_reports(folder_path):
    """
    Reads all CSV files in the folder and extracts the Trades section.
    """
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(files)} files in {folder_path}")

    dfs = []
    
    for file in files:
        print(f"Processing {file}...")
        try:
            # Use csv module to parse lines correctly (handling quotes)
            import csv
            from io import StringIO
            
            raw_lines = []
            with open(file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.startswith("Trades,Header") or line.startswith("Trades,Data,Order"):
                        raw_lines.append(line)
            
            if raw_lines:
                csv_io = StringIO("".join(raw_lines))
                reader = csv.reader(csv_io)
                rows = list(reader)
                
                # Find header
                header_row = None
                data_rows_csv = []
                for row in rows:
                    if len(row) > 1 and row[0] == 'Trades' and row[1] == 'Header':
                        header_row = row
                    elif len(row) > 2 and row[0] == 'Trades' and row[1] == 'Data' and row[2] == 'Order':
                        data_rows_csv.append(row)
                
                if header_row and data_rows_csv:
                    # Normalize lengths
                    header_len = len(header_row)
                    clean_rows = [r for r in data_rows_csv if len(r) == header_len]
                    if clean_rows:
                        # Deduplicate header
                        seen = {}
                        new_header = []
                        for col in header_row:
                            if col in seen:
                                seen[col] += 1
                                new_header.append(f"{col}.{seen[col]}")
                            else:
                                seen[col] = 0
                                new_header.append(col)
                        
                        df = pd.DataFrame(clean_rows, columns=new_header)
                        dfs.append(df)
                    else:
                        print(f"Warning: No valid data rows found in {file} matching header length {header_len}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter for Stocks
    if 'Asset Category' in df.columns:
        df = df[df['Asset Category'] == 'Stocks']
    
    # Parse Dates
    # Format: "2022-12-09, 10:35:13"
    # Sometimes it has quotes.
    if 'Date/Time' in df.columns:
        df['Date/Time'] = df['Date/Time'].astype(str).str.replace('"', '')
        df['Date'] = pd.to_datetime(df['Date/Time']).dt.date
    
    # Parse Numbers
    numeric_cols = ['Quantity', 'T. Price', 'Proceeds', 'Comm/Fee', 'Basis']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Sort by Date
    if 'Date/Time' in df.columns:
        df = df.sort_values('Date/Time')
    
    return df

def get_fx_rates(dates):
    """
    Fetches USD/ILS rates for the given dates.
    Returns a Series indexed by date.
    """
    if not dates:
        return pd.Series()
        
    min_date = min(dates) - datetime.timedelta(days=5) # Buffer
    max_date = max(dates) + datetime.timedelta(days=5)
    
    print(f"Fetching FX rates from {min_date} to {max_date}...")
    
    try:
        # Yahoo Finance
        ticker = "USDILS=X"
        df = yf.download(ticker, start=min_date, end=max_date, progress=False)
        
        # Create a complete date range and fill forward (for weekends/holidays)
        full_range = pd.date_range(start=min_date, end=max_date)
        df_reindexed = df['Close'].reindex(full_range).ffill()
        
        # Map requested dates
        rates = {}
        for d in dates:
            ts = pd.Timestamp(d)
            if ts in df_reindexed.index:
                rates[d] = float(df_reindexed.loc[ts].iloc[0]) if isinstance(df_reindexed.loc[ts], pd.Series) or isinstance(df_reindexed.loc[ts], pd.DataFrame) else float(df_reindexed.loc[ts])
            else:
                # Fallback to nearest
                try:
                    idx = df_reindexed.index.get_indexer([ts], method='nearest')[0]
                    rates[d] = float(df_reindexed.iloc[idx].iloc[0]) if isinstance(df_reindexed.iloc[idx], pd.Series) or isinstance(df_reindexed.iloc[idx], pd.DataFrame) else float(df_reindexed.iloc[idx])
                except:
                    rates[d] = 3.5 # Fallback default
                    
        return pd.Series(rates)
    except Exception as e:
        print(f"Error fetching FX: {e}")
        return pd.Series({d: 3.5 for d in dates}) # Fallback

def get_cpi_index(start_date, end_date):
    """
    Fetches Israel CPI growth rates and reconstructs an index.
    """
    print(f"Fetching CPI data from {start_date} to {end_date}...")
    try:
        # Fetch monthly growth rates
        # Buffer dates
        s = start_date - datetime.timedelta(days=60)
        e = end_date + datetime.timedelta(days=60)
        
        df = web.DataReader(CPI_SERIES_ID, "fred", s, e)
        
        # df contains percent change from previous month (e.g., 0.2 for 0.2%)
        # Convert to multiplier: 1 + (value / 100)
        df['Multiplier'] = 1 + (df[CPI_SERIES_ID] / 100)
        
        # Calculate cumulative product to get Index
        # We can normalize the first available date to 100
        df['Index'] = 100 * df['Multiplier'].cumprod()
        
        # Resample to daily to allow lookup (ffill)
        # CPI is usually released for the previous month.
        # We will just ffill the monthly value.
        df_daily = df['Index'].resample('D').ffill()
        
        return df_daily
    except Exception as e:
        print(f"Error fetching CPI: {e}")
        return pd.Series()

def process_fifo(trades_df):
    """
    Matches Buy and Sell trades using FIFO, handling both Long and Short positions.
    """
    matches = []
    # Dictionaries to hold queues for each symbol
    long_queues = {}  # Buys waiting to be sold
    short_queues = {} # Short Sells waiting to be covered
    
    for _, row in trades_df.iterrows():
        symbol = row['Symbol']
        date = row['Date']
        qty = row['Quantity']
        price = row['T. Price']
        comm = row['Comm/Fee'] if not pd.isna(row['Comm/Fee']) else 0
        
        if symbol not in long_queues:
            long_queues[symbol] = []
        if symbol not in short_queues:
            short_queues[symbol] = []
            
        if qty > 0:
            # Buy Transaction
            qty_to_process = qty
            comm_per_unit = comm / qty
            
            # 1. Check if we need to cover any Short positions
            while qty_to_process > 0 and short_queues[symbol]:
                short_item = short_queues[symbol][0]
                matched_qty = min(qty_to_process, short_item['qty'])
                
                matches.append({
                    'Symbol': symbol,
                    'Buy Date': date,          # Closing Date (Cover)
                    'Sell Date': short_item['date'], # Opening Date (Short Sell)
                    'Quantity': matched_qty,
                    'Buy Price USD': price,
                    'Sell Price USD': short_item['price'],
                    'Buy Comm USD': comm_per_unit * matched_qty,
                    'Sell Comm USD': short_item['comm_per_unit'] * matched_qty,
                    'Type': 'Short'
                })
                
                qty_to_process -= matched_qty
                short_item['qty'] -= matched_qty
                
                if short_item['qty'] < 1e-9:
                    short_queues[symbol].pop(0)
            
            # 2. If quantity remains, add to Long queue
            if qty_to_process > 1e-9:
                long_queues[symbol].append({
                    'date': date,
                    'qty': qty_to_process,
                    'price': price,
                    'comm_per_unit': comm_per_unit
                })

        elif qty < 0:
            # Sell Transaction
            qty_to_process = abs(qty)
            comm_per_unit = comm / qty_to_process # comm is negative
            
            # 1. Check if we can close any Long positions
            while qty_to_process > 0 and long_queues[symbol]:
                long_item = long_queues[symbol][0]
                matched_qty = min(qty_to_process, long_item['qty'])
                
                matches.append({
                    'Symbol': symbol,
                    'Buy Date': long_item['date'],   # Opening Date (Buy)
                    'Sell Date': date,               # Closing Date (Sell)
                    'Quantity': matched_qty,
                    'Buy Price USD': long_item['price'],
                    'Sell Price USD': price,
                    'Buy Comm USD': long_item['comm_per_unit'] * matched_qty,
                    'Sell Comm USD': comm_per_unit * matched_qty,
                    'Type': 'Long'
                })
                
                qty_to_process -= matched_qty
                long_item['qty'] -= matched_qty
                
                if long_item['qty'] < 1e-9:
                    long_queues[symbol].pop(0)
            
            # 2. If quantity remains, add to Short queue
            if qty_to_process > 1e-9:
                short_queues[symbol].append({
                    'date': date,
                    'qty': qty_to_process,
                    'price': price,
                    'comm_per_unit': comm_per_unit
                })
    
    # Print summary of open positions
    print("\n--- Open Positions Summary (End of all processed files) ---")
    open_longs = 0
    open_shorts = 0
    for sym, q in long_queues.items():
        qty = sum(i['qty'] for i in q)
        if qty > 1e-9:
            # print(f"Long {sym}: {qty:.4f}") # Optional: detailed print
            open_longs += 1
    for sym, q in short_queues.items():
        qty = sum(i['qty'] for i in q)
        if qty > 1e-9:
            print(f"WARNING: Open Short {sym}: {qty:.4f} (Check if missing Buy history)")
            open_shorts += 1
            
    print(f"Total Open Long Positions: {open_longs}")
    print(f"Total Open Short Positions: {open_shorts}")
    print("-----------------------------------------------------------\n")
                    
    return pd.DataFrame(matches)

def main(folder_path, tax_year):
    # 1. Parse Trades
    print(f"Parsing trades from {folder_path}...")
    trades = parse_ibkr_reports(folder_path)
    if trades.empty:
        print("No trades found.")
        return

    # 2. FIFO Matching
    print("Processing FIFO...")
    matches = process_fifo(trades)
    if matches.empty:
        print("No matches found.")
        return
        
    # 3. FX Conversion
    print("Fetching FX rates...")
    all_dates = pd.concat([matches['Buy Date'], matches['Sell Date']]).unique()
    all_dates = sorted([d for d in all_dates if isinstance(d, datetime.date)])
    
    fx_rates = get_fx_rates(all_dates)
    
    matches['Buy Rate'] = matches['Buy Date'].map(lambda d: fx_rates.get(d, 0))
    matches['Sell Rate'] = matches['Sell Date'].map(lambda d: fx_rates.get(d, 0))
    
    matches['Buy Price ILS'] = matches['Buy Price USD'] * matches['Buy Rate']
    matches['Sell Price ILS'] = matches['Sell Price USD'] * matches['Sell Rate']
    matches['Buy Comm ILS'] = matches['Buy Comm USD'] * matches['Buy Rate']
    matches['Sell Comm ILS'] = matches['Sell Comm USD'] * matches['Sell Rate']
    
    # Total Basis and Proceeds in ILS
    # Basis = (Price * Qty) + abs(Comm)  (Comm is negative)
    # Proceeds = (Price * Qty) + Comm    (Comm is negative)
    # Wait, usually:
    # Cost Basis = (Buy Price * Qty) + Commission Paid (Positive value)
    # Net Proceeds = (Sell Price * Qty) - Commission Paid (Positive value)
    # In the data, Comm is negative.
    # So Cost Basis = (Buy Price * Qty) - Buy Comm
    # Net Proceeds = (Sell Price * Qty) + Sell Comm
    
    matches['Cost Basis ILS'] = (matches['Buy Price ILS'] * matches['Quantity']) - matches['Buy Comm ILS']
    matches['Net Proceeds ILS'] = (matches['Sell Price ILS'] * matches['Quantity']) + matches['Sell Comm ILS']
    
    # 4. Inflation Adjustment
    print("Fetching Inflation data...")
    min_date = matches['Buy Date'].min()
    max_date = matches['Sell Date'].max()
    
    cpi_index = get_cpi_index(min_date, max_date)
    
    def get_cpi(d):
        ts = pd.Timestamp(d)
        # Find nearest previous index
        # If exact match missing, use asof
        if ts in cpi_index.index:
            return cpi_index.loc[ts]
        else:
            # Use asof to get the latest available CPI before or on this date
            idx = cpi_index.index.asof(ts)
            if pd.isna(idx):
                return cpi_index.iloc[0] # Fallback to first
            return cpi_index.loc[idx]

    matches['Buy CPI'] = matches['Buy Date'].map(get_cpi)
    matches['Sell CPI'] = matches['Sell Date'].map(get_cpi)
    
    matches['Inflation Factor'] = matches['Sell CPI'] / matches['Buy CPI']
    matches['Adjusted Cost Basis ILS'] = matches['Cost Basis ILS'] * matches['Inflation Factor']
    
    # 5. Tax Calculation
    matches['Real Gain ILS'] = matches['Net Proceeds ILS'] - matches['Adjusted Cost Basis ILS']
    matches['Nominal Gain ILS'] = matches['Net Proceeds ILS'] - matches['Cost Basis ILS']
    
    # Filter for Tax Year
    # Tax event is on the Closing Date
    # For Long: Sell Date
    # For Short: Buy Date (Cover Date)
    
    def get_tax_year(row):
        if row['Type'] == 'Short':
            return pd.to_datetime(row['Buy Date']).year
        else:
            return pd.to_datetime(row['Sell Date']).year
            
    matches['Tax Year'] = matches.apply(get_tax_year, axis=1)
    report = matches[matches['Tax Year'] == int(tax_year)].copy()
    
    # Output
    output_file = f"Tax_Report_{tax_year}.csv"
    
    with open(output_file, 'w', newline='') as f:
        f.write("SECTION: FIFO MATCHES AND CALCULATIONS\n")
        report.to_csv(f, index=False)
        
        f.write("\nSECTION: SUMMARY\n")
        total_proceeds = report['Net Proceeds ILS'].sum()
        total_cost_adj = report['Adjusted Cost Basis ILS'].sum()
        total_gain_real = report['Real Gain ILS'].sum()
        total_gain_nominal = report['Nominal Gain ILS'].sum()
        
        f.write(f"Total Proceeds ILS,{total_proceeds}\n")
        f.write(f"Total Adjusted Cost ILS,{total_cost_adj}\n")
        f.write(f"Total Real Gain ILS (Taxable Amount),{total_gain_real}\n")
        f.write(f"Total Nominal Gain ILS,{total_gain_nominal}\n")
        
        tax_25 = total_gain_real * 0.25
        tax_28 = total_gain_real * 0.28
        
        f.write(f"Estimated Tax Liability (25%),{tax_25}\n")
        f.write(f"Estimated Tax Liability (28%),{tax_28}\n")
        
        f.write("\nSECTION: EXPLANATION\n")
        f.write("This report calculates the Capital Gains Tax liability for the specified tax year based on FIFO (First-In, First-Out) accounting principles.\n")
        f.write("1. Transactions are matched: Sells are matched with the earliest available Buys.\n")
        f.write("2. Amounts are converted to ILS: All USD amounts are converted to ILS using the daily USD/ILS exchange rate on the transaction date.\n")
        f.write("3. Inflation Adjustment: The Cost Basis is adjusted for inflation in Israel from the Buy Date to the Sell Date.\n")
        f.write("   - Adjusted Cost Basis = Cost Basis (ILS) * (CPI at Sell Date / CPI at Buy Date)\n")
        f.write("4. Real Gain: Calculated as Net Proceeds (ILS) - Adjusted Cost Basis (ILS). This is the taxable amount in Israel.\n")
        f.write("5. Nominal Gain: Calculated as Net Proceeds (ILS) - Cost Basis (ILS) (without inflation adjustment).\n")

        f.write("\nSECTION: DATA SOURCES\n")
        f.write("Foreign Exchange Rates (USD/ILS): Yahoo Finance (Ticker: USDILS=X)\n")
        f.write(f"Inflation Data (Israel CPI): Federal Reserve Economic Data (FRED) (Series ID: {CPI_SERIES_ID})\n")
        f.write("   - Description: Consumer Price Index: All Items for Israel, Growth Rate Previous Period, Monthly, Not Seasonally Adjusted.\n")
        f.write("   - Note: The inflation index is reconstructed from the monthly growth rates provided by the OECD via FRED.\n")
        
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python tax_calculator.py <folder_path> <tax_year>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    tax_year = sys.argv[2]
    main(folder_path, tax_year)
