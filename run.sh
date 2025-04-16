# Financial Data Processing Pipeline


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

## Project Structure

```
.
├── build_dataset/
│   └── build_dataset.ipynb    # Main data processing notebook
├── input_csv/
│   ├── bloomberg_data_raw.csv # Raw Bloomberg data
│   ├── DFF.csv               # Federal Funds Rate data
│   ├── CPI.csv               # Consumer Price Index data
│   ├── UMCSENT.csv           # Consumer Sentiment data
│   └── UNRATE.csv            # Unemployment Rate data
└── output_csv/
    └── bloomberg_data.csv    # Final processed dataset
```

## Dependencies

The project uses the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib

## Usage

1. Place all input CSV files in the `input_csv` directory
2. Run the `build_dataset.ipynb` notebook
3. Find the processed dataset in `output_csv/bloomberg_data.csv`