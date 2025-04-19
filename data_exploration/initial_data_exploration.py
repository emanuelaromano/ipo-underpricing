# -*- coding: utf-8 -*-
"""
Initial Data Exploration Script for IPO Underpricing Analysis

This script performs comprehensive exploratory data analysis on IPO data,
including data cleaning, visualization, and statistical analysis.
The analysis focuses on understanding IPO characteristics and their relationship
with underpricing (Offer To 1st Close).

Original file was created in Google Colab and has been adapted for local use.
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

"""## Data Exploration"""

# Load the IPO dataset
df = pd.read_csv("../build_dataset/output_csv/bloomberg_data.csv")

def remove_max_target(df):
    """
    Remove extreme values from the target variable (Offer To 1st Close)
    to reduce the impact of outliers on the analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe containing IPO data
        
    Returns:
        pd.DataFrame: Filtered dataframe with extreme values removed
    """
    lower = df['Offer To 1st Close'].min()
    upper = df['Offer To 1st Close'].max()
    df = df[(df['Offer To 1st Close'] > lower) & (df['Offer To 1st Close'] < upper)].reset_index(drop=True)
    return df

# Apply the outlier removal function
df = remove_max_target(df)

"""### Dataset Information"""

# Display basic information about the dataset including data types and non-null counts
df_info = df.info()
df_info

"""### Describe"""

# Generate descriptive statistics for all numeric columns
df_description = df.describe()
df_description

"""### Initial Cleaning Based on Describe"""

# Remove records with invalid offer sizes
df = df[df['Offer Size (M)'] > 0]

# Remove records with extreme negative underpricing that exceeds the offer price
df = df[~((df['Offer To 1st Close'] < 0) & (df['Offer To 1st Close'].abs() >= df['Offer Price']))]

# Save the cleaned dataset
df.to_csv('../build_dataset/output_csv/bloomberg_data.csv', index=False)

"""### Count Duplicates"""

# Count and display the number of duplicate records in the dataset
duplicate_count = df.duplicated().sum()
print(f"Number of duplicates: {duplicate_count}")

"""### Count Unique"""

# Count unique values in each column to understand data cardinality
countunique = pd.DataFrame(df.nunique(), columns=['Count'])
countunique

"""### Missing Values"""

def missing_values_plot(df):
    """
    Analyze and visualize missing values in the dataset.
    Creates a bar plot showing the percentage of missing values for each column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame containing missing value statistics
    """
    # Calculate missing values and their percentages
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentages
    }).sort_values('Percentage', ascending=False)

    # Filter to only show columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0]

    # Create visualization
    plt.figure(figsize=(12, 8))
    ax = missing_df['Percentage'].plot(kind='bar')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=90)

    # Add percentage labels to bars
    for i, v in enumerate(missing_df['Percentage']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return missing_df

# Generate missing values analysis
missing_df = missing_values_plot(df)
missing_df

"""### Numerical Outliers"""

def numerical_outlier_analysis(df):
    """
    Analyze numerical outliers in the dataset using the IQR method.
    Identifies and reports outliers for each numeric column.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_data = []

    # Calculate outliers for each numeric column
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        if len(outliers) > 0:
            outlier_data.append({
                'Column': col,
                'Number of Outliers': len(outliers),
                'Percentage of Outliers': (len(outliers)/len(df))*100,
                'Lower Bound': Q1 - 1.5 * IQR,
                'Upper Bound': Q3 + 1.5 * IQR
            })

    # Display outlier statistics
    outlier_df = pd.DataFrame(outlier_data)
    if not outlier_df.empty:
        display(outlier_df)

# Perform outlier analysis
numerical_outlier_analysis(df)

"""### Distribution of IPO Offer Prices"""

def offer_price_distribution(df):
    """
    Visualize the distribution of IPO offer prices using a histogram.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Offer Price'].dropna(), bins=30, kde=True)
    plt.xlabel('Offer Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of IPO Offer Prices')
    plt.show()

# Generate offer price distribution plot
offer_price_distribution(df)

"""### Distribution of Market Capitalization at Offer"""

def market_cap_at_offer_distribution(df):
    """
    Visualize the distribution of market capitalization at IPO using a histogram.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Market Cap at Offer (M)'].dropna(), bins=30, kde=True)
    plt.xlabel('Market Cap at Offer (M)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Market Capitalization at Offer')
    plt.show()

# Generate market cap distribution plot
market_cap_at_offer_distribution(df)

"""### Number of IPOs over time"""

def number_of_ipos_over_time(df):
    """
    Analyze and visualize the trend of IPO activity over time.
    Creates a line plot showing the number of IPOs per year.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Prepare data for time series analysis
    df_tmp = df.copy()
    df_tmp['Trade Date (US)'] = pd.to_datetime(df_tmp['Trade Date (US)'], errors='coerce')
    df_tmp.set_index('Trade Date (US)', inplace=True)
    df_tmp['IPO Count'] = 1
    ipo_trend = df_tmp.resample('YE')['IPO Count'].sum()

    # Create visualization
    plt.figure(figsize=(10, 5))
    ipo_trend.plot()
    plt.xlabel('Year')
    plt.ylabel('Number of IPOs')
    plt.title('IPO Trends Over Time')
    plt.grid(False)
    plt.show()

# Generate IPO trend analysis
number_of_ipos_over_time(df)

"""### Industry Sector Distribution"""

def industry_sector_distribution(df):
    """
    Visualize the distribution of IPOs across different industry sectors.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(10, 5))
    df['Industry Sector'].value_counts().plot(kind='bar')
    plt.xlabel('Industry Sector')
    plt.ylabel('Count')
    plt.title('Distribution of IPOs by Industry Sector')
    plt.xticks(rotation=45)
    plt.show()

# Generate industry sector distribution plot
industry_sector_distribution(df)

"""### Correlation Matrix"""

def correlation_matrix(df):
    """
    Analyze and visualize correlations between numeric features.
    Creates a heatmap showing the correlation matrix.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Select numeric columns excluding the target variable
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[numeric_cols != 'Offer To 1st Close']

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.show()

# Generate correlation analysis
correlation_matrix(df)

"""### Offer To 1st Close Distribution"""

def offer_to_1st_close_distribution(df):
    """
    Visualize the distribution of IPO underpricing (Offer To 1st Close)
    using a box plot.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Offer To 1st Close'].dropna())
    plt.xlabel('Offer To 1st Close')
    plt.title('Distribution of Offer To 1st Close')
    plt.show()

# Generate underpricing distribution plot
offer_to_1st_close_distribution(df)

def offer_to_1st_close_distribution_no_outliers(df):
    """
    Visualize the distribution of IPO underpricing after removing outliers
    using the IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Calculate IQR bounds
    IQR = df['Offer To 1st Close'].quantile(0.75) - df['Offer To 1st Close'].quantile(0.25)
    lower_bound = df['Offer To 1st Close'].quantile(0.25) - 1.5 * IQR
    upper_bound = df['Offer To 1st Close'].quantile(0.75) + 1.5 * IQR
    
    # Filter out outliers
    df_no_outliers = df[(df['Offer To 1st Close'] > lower_bound) & (df['Offer To 1st Close'] < upper_bound)]
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_no_outliers['Offer To 1st Close'].dropna())
    plt.xlabel('Offer To 1st Close')
    plt.title('Distribution of Offer To 1st Close (No Outliers)')
    plt.show()

# Generate underpricing distribution plot without outliers
offer_to_1st_close_distribution_no_outliers(df)

"""### Offer To 1st Close Distribution by Industry"""

def analyze_target_variable(df):
    """
    Analyze the distribution of underpricing across different industry sectors.
    Creates a box plot showing the distribution for the top 10 sectors.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(12, 6))
    # Select top 10 sectors by count
    top_sectors = df['Industry Sector'].value_counts().head(10).index
    df_top_sectors = df[df['Industry Sector'].isin(top_sectors)].copy()

    df_top_sectors = df_top_sectors.reset_index(drop=True)
    ax = sns.boxplot(x='Industry Sector', y='Offer To 1st Close', data=df_top_sectors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Offer To 1st Close by Top 10 Industry Sectors')
    plt.tight_layout()
    plt.show()

# Generate industry-wise underpricing analysis
analyze_target_variable(df)

def analyze_target_variable_no_outliers(df):
    """
    Analyze the distribution of underpricing across different industry sectors
    after removing outliers using the IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(12, 6))
    # Calculate IQR bounds
    IQR = df['Offer To 1st Close'].quantile(0.75) - df['Offer To 1st Close'].quantile(0.25)
    lower_bound = df['Offer To 1st Close'].quantile(0.25) - 1.5 * IQR
    upper_bound = df['Offer To 1st Close'].quantile(0.75) + 1.5 * IQR
    
    # Filter out outliers
    df_no_outliers = df[(df['Offer To 1st Close'] > lower_bound) & (df['Offer To 1st Close'] < upper_bound)]
    
    # Select top 10 sectors by count
    top_sectors = df_no_outliers['Industry Sector'].value_counts().head(10).index
    df_top_sectors = df_no_outliers[df_no_outliers['Industry Sector'].isin(top_sectors)].copy()

    df_top_sectors = df_top_sectors.reset_index(drop=True)
    ax = sns.boxplot(x='Industry Sector', y='Offer To 1st Close', data=df_top_sectors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Offer To 1st Close by Top 10 Industry Sectors (No Outliers)')
    plt.tight_layout()
    plt.show()

# Generate industry-wise underpricing analysis without outliers
analyze_target_variable_no_outliers(df)