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

## Analysis Components

### Regression Analysis
The regression analysis focuses on predicting the "Offer To 1st Close" percentage change using various machine learning models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- LightGBM
- Support Vector Regression (SVR)

Key features used include:
- Company financial metrics (Sales Growth, Profit Margin, ROA)
- Market indicators (Offer Size, Shares Outstanding, Market Cap)
- Economic indicators (Fed Rate, CPI, Consumer Confidence, Unemployment)
- Industry sector information
- Temporal features (Trade Month, Day, Year)

### Classification Analysis
The classification analysis predicts whether an IPO will be underpriced (Offer To 1st Close < 0) using:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Classifier
- LightGBM Classifier

The analysis includes:
- Feature engineering and preprocessing
- Handling class imbalance using SMOTE
- Model evaluation using accuracy, F1 score, precision, and recall
- Feature importance analysis

## Dependencies

The project uses the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- lightgbm
- imbalanced-learn (for SMOTE)

## Usage

1. Place all input CSV files in the `input_csv` directory
2. Run the `build_dataset.ipynb` notebook to process the data
3. Use either `regression.ipynb` or `classification.ipynb` to perform the desired analysis
4. Find the processed dataset in `output_csv/bloomberg_data.csv`