# Financial Data Processing and Analysis Pipeline

This project processes and analyzes financial and economic datasets to predict IPO performance and identify underpriced offerings.

## Data Sources

The pipeline combines data from the following sources:
- Bloomberg data (raw financial data)
- Federal Reserve Economic Data (FRED):
  - DFF (Federal Funds Rate)
  - CPI (Consumer Price Index)
  - UMCSENT (University of Michigan Consumer Sentiment)
  - UNRATE (Unemployment Rate)

## Data Processing Steps

1. **Data Loading and Initial Processing**
   - Loads raw Bloomberg data from `input_csv/bloomberg_data_raw.csv`
   - Converts trade dates to datetime format
   - Creates month-based timestamps for alignment

2. **Economic Indicators Integration**
   - Loads and processes additional economic indicators:
     - Federal Funds Rate (DFF)
     - Consumer Price Index (CPI)
     - Consumer Sentiment (UMCSENT)
     - Unemployment Rate (UNRATE)
   - Standardizes date formats across all datasets

3. **Data Merging**
   - Combines all datasets using month-based timestamps
   - Performs left joins to preserve all Bloomberg data points
   - Renames columns for clarity:
     - CPALTT01USM657N → CPI
     - DFF → Fed Rate
     - UMCSENT → Consumer Confidence
     - UNRATE → Unemployment Rate

4. **Output Generation**
   - Saves the final combined dataset to `output_csv/bloomberg_data.csv`
   - Removes intermediate timestamp columns
   - Preserves all original data points

## Usage

1. Place all input CSV files in the `input_csv` directory
2. Run the `build_dataset.ipynb` notebook to process the data
3. Run the `initial_data_exploration.ipynb` notebook to make an initial exploration of the data
4. Use either `regression.ipynb` or `classification.ipynb` to perform the desired analysis
5. Find the processed dataset in `output_csv/bloomberg_data.csv`