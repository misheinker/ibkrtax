# IBKR Israel Tax Calculator

This Python script parses Interactive Brokers (IBKR) Activity Statements (CSV format) and calculates the Capital Gains Tax liability for Israeli tax residents. It handles FIFO (First-In, First-Out) matching, currency conversion to ILS, and inflation adjustment according to Israeli tax rules.

## Features

*   **FIFO Matching**: Automatically matches Sell transactions with the earliest Buy transactions.
*   **Short Selling Support**: Correctly handles short sales (Sell to Open, Buy to Close).
*   **Currency Conversion**: Converts all USD transaction amounts to ILS using historical daily exchange rates (sourced from Yahoo Finance).
*   **Inflation Adjustment**: Adjusts the Cost Basis for Israeli inflation (CPI) from the purchase date to the sale date (sourced from FRED).
*   **Tax Calculation**: Calculates "Real Gain" (Taxable Amount) and provides estimated tax liability at 25% and 28%.
*   **Dividend and Interest Income**: Parses and converts all dividend and interest payments to ILS for tax reporting.
*   **Spill-over Handling**: Tracks open positions across multiple years (provided you supply the history).

## Prerequisites

*   Python 3.8 or higher.
*   Interactive Brokers Activity Statements in CSV format.

## Setup

1.  **Clone or Download** this repository.

2.  **Create a Virtual Environment** (recommended):
    ```bash
    # Windows
    py -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1.  Create a folder named `ibkr_reports` (or modify the `folder` variable in the script).
2.  **Download your Activity Statements** from Interactive Brokers:
    *   Log in to IBKR Portal.
    *   Go to **Performance & Reports** > **Statements**.
    *   Select **Activity Statement**.
    *   Generate a report for **each year** you have traded (e.g., 2022, 2023, 2024).
    *   **Format**: Select **CSV**.
    *   **Period**: **Annual** (Jan 1 - Dec 31).
3.  Place all downloaded CSV files into the folder.
    *   *Important*: You must include historical files even if you are only calculating tax for the current year, so the script can find the original purchase date/price for assets sold this year.

## Usage

Run the script from the terminal, specifying the folder path containing the reports and the tax year.

```bash
# Windows (with virtual env activated)
python tax_calculator.py ibkr_reports 2023

# Or using the direct path
.venv\Scripts\python tax_calculator.py ibkr_reports 2023
```

The script will generate a file named `Tax_Report_<Year>_<Name>_<Account>.csv` in the same directory.

## Output Explanation

The generated CSV report contains several sections:

1. **CAPITAL GAINS - FIFO MATCHES AND CALCULATIONS**: Detailed list of all closed positions (trades).
2. **DIVIDENDS AND INTEREST**: All dividend and interest payments received during the tax year.
3. **SUMMARY**: Aggregated totals and tax calculations.
4. **EXPLANATION**: Detailed methodology explanation.
5. **DATA SOURCES**: References for exchange rates and inflation data.

### Summary Fields

**Capital Gains:**

*   **Capital Gains - Total Proceeds ILS**:
    The total net amount received from all sales in the tax year, converted to Shekels (ILS) at the exchange rate on the sale date.

*   **Capital Gains - Total Adjusted Cost ILS**:
    The original cost of the assets sold, converted to ILS at the purchase date, and **adjusted for inflation** (Israeli CPI) up to the sale date. This ensures you don't pay tax on inflation.

*   **Capital Gains - Total Real Gain ILS (Taxable)**:
    `Total Proceeds` - `Total Adjusted Cost`.
    This is your actual profit in purchasing power terms. This is the amount generally subject to Capital Gains Tax in Israel.

*   **Capital Gains - Total Nominal Gain ILS**:
    `Total Proceeds` - `Original Cost (in ILS)`.
    Your profit in Shekels ignoring inflation. Provided for reference only.

**Dividends and Interest:**

*   **Dividends - Total ILS**:
    The total amount of dividend payments received during the tax year, converted to ILS at the exchange rate on each payment date.

*   **Interest - Total ILS**:
    The total amount of interest (credit or debit) during the tax year, converted to ILS at the exchange rate on each payment date.

*   **Total Dividend and Interest Income ILS (Taxable)**:
    The sum of all dividends and interest. This amount is fully taxable as income in Israel.
    **Note**: US withholding tax on dividends (typically 25-30%) may be claimed as a credit against your Israeli tax liability. Consult with your tax advisor on how to report this.

**Combined Tax:**

*   **Combined Taxable Income**:
    The sum of Capital Gains Real Gain and Dividend/Interest Income.

*   **Estimated Tax Liability (25%)**:
    25% of the Combined Taxable Income. This is the standard rate for individuals.

*   **Estimated Tax Liability (28%)**:
    28% of the Combined Taxable Income. This rate applies to "substantial shareholders" or high-income earners subject to Surtax.

## Data Sources

*   **FX Rates**: Yahoo Finance (Ticker: `USDILS=X`).
*   **Inflation Data**: Federal Reserve Economic Data (FRED) (Series ID: `CPALTT01ILM657N` - Israel CPI Growth Rate).

## Disclaimer

**I am not a tax accountant.** This script is for informational purposes only and should not be considered professional tax advice. Always verify the results with a certified tax professional (CPA) before filing your taxes.
