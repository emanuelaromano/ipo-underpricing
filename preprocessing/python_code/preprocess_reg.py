import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, cross_validate
from datetime import datetime

import csv
from datetime import datetime

def convert_offer_to_underpriced(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read header and find target column indexes
        headers = next(reader)
        try:
            col_index_trade_date = headers.index('Trade Date (US)')
        except ValueError as e:
            raise ValueError(f"Required column not found: {e}")

        # Create new header: replace 'Trade Date (US)' with new columns
        new_headers = [
            h for i, h in enumerate(headers) if i not in [col_index_trade_date]
        ] + ['Trade Month', 'Trade Day', 'Trade Year']
        writer.writerow(new_headers)

        # Process rows
        for row in reader:
            new_row = [v for i, v in enumerate(row) if i not in [col_index_trade_date]]

            # Process 'Trade Date (US)'
            trade_month, trade_day, trade_year = '0', '0', '0'
            if len(row) > col_index_trade_date:
                trade_date = row[col_index_trade_date].strip()

                if '/' in trade_date:  # M/D/YYYY format
                    parts = trade_date.split("/")
                    if len(parts) == 3 and all(part.isdigit() for part in parts):
                        month, day, year = map(int, parts)
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1000 <= year <= 9999:
                            trade_month, trade_day, trade_year = str(month), str(day), str(year)

                elif '-' in trade_date:  # YYYY-MM-DD format
                    parts = trade_date.split("-")
                    if len(parts) == 3 and all(part.isdigit() for part in parts):
                        year, month, day = map(int, parts)
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1000 <= year <= 9999:
                            trade_month, trade_day, trade_year = str(month), str(day), str(year)
                        else:
                            print(f"Warning: Invalid date format found: {trade_date}")
                    else:
                        print(f"Warning: Invalid date format found: {trade_date}")

            # Append new columns to row
            new_row.extend([trade_month, trade_day, trade_year])
            writer.writerow(new_row)



#Drop columns and write it to new csv file
def drop_column(columns,filename,outfile):
    df = pd.read_csv(filename)
    df = df.drop(columns=columns)
    df.to_csv(outfile,index=False)

#Show column changes
def show_conversion(prevfile,new_file):
    df = pd.read_csv(prevfile)
    print('---------------------------------------------------All Columns in initial Data File-------------------------------------------------------------------')
    print(df.columns.tolist())
    print(df.head(10))
    print('---------------------------------------------------All Columns in Label Converted Data File-------------------------------------------------------------------')
    df_new = pd.read_csv(new_file)
    print(df_new.columns.tolist())
    print(df_new.head(10))
    diff = set(df.columns.tolist()) - set(df_new.columns.tolist())
    diff2 = set(df_new.columns.tolist()) - set(df.columns.tolist())
    print(f'Old columns removed: {diff if diff else "None"}')
    print(f'New columns added: {diff2 if diff2 else "None"}')

def show_correlation(filename,outfile):
    df = pd.read_csv(filename)
    label = df['Offer To 1st Close']
    df = df.drop(columns=['Offer To 1st Close'])
    corr_matrix = df.corr().abs()
    
    # Find correlated features to drop
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.9)]
    
    # Drop correlated features
    X_reduced = df.drop(columns=to_drop)
    
    # Create final DataFrame
    df_final = pd.concat([X_reduced, label], axis=1)
    
    # Ensure Offer To 1st Close is last column
    column_order = X_reduced.columns.tolist() + ['Offer To 1st Close']
    df_final = df_final[column_order]
    for col in df_final.columns:
        df_final[col].fillna(df_final[col].mean(), inplace=True)
    # Save to CSV
    df_final.to_csv(outfile, index=False)
    print(f"Removed {len(to_drop)} features: {to_drop}")


    

#Encode the data
def encoding(filename,outfile):
    df = pd.read_csv(filename)
    print(df['Offer To 1st Close'].head(10))
    #Apply One Hot encoding on these columns
    oe = OrdinalEncoder()
    oe_column = ['Trade Month','Trade Day', 'Trade Year'] 
    OHE_Column = ['Industry Sector','Industry Group','Industry Subgroup']
    Scaling_Column = ['Sales - 1 Yr Growth','Profit Margin','Return on Assets','Offer Size (M)','Shares Outstanding (M)','Offer Price','Market Cap at Offer (M)','Cash Flow per Share','Offer Size (M).1','Shares Outstanding (M).1','Instit Owner (% Shares Out)','Instit Owner (Shares Held)','Fed Rate','CPI','Consumer Confidence','Unemployment Rate']
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    ss=StandardScaler()
    Preprocess_step = ColumnTransformer(transformers=[('ohe',ohe,OHE_Column),('ss',ss,Scaling_Column),('oe',oe,oe_column)],remainder='passthrough')
    transformed_data = Preprocess_step.fit_transform(df)
    ohe_features = Preprocess_step.named_transformers_['ohe'].get_feature_names_out(OHE_Column)
    remaining_features = df.columns.drop(OHE_Column + Scaling_Column).tolist()
    feature_names = list(ohe_features) + Scaling_Column + remaining_features
    df_transformed = pd.DataFrame(transformed_data, columns=feature_names)
    df_transformed.to_csv(outfile, index=False)

def checkimbalance(filename):
    outliers = False
    df = pd.read_csv(filename)
    label = df['Offer To 1st Close']
    print('Minimum value: ', label.min())
    print('Q1: ', label.quantile(0.25))
    print('Median value: ', label.median())
    print('Q3: ', label.quantile(0.75))
    print('Maximum value: ', label.max())
    print('Mean value: ', label.mean())
    print('Standard deviation: ', label.std())
    if label.max() > label.quantile(0.75) + 1.5 * (label.quantile(0.75) - label.quantile(0.25)):
        print('There are outliers in the data')
        outliers = True
    elif label.min() < label.quantile(0.25) - 1.5 * (label.quantile(0.75) - label.quantile(0.25)):
        print('There are outliers in the data')
        outliers = True
    else:
        print('There are no outliers in the data')
    if outliers:
        remove_outlier(filename,'Final_Output_Reg_no_outliers.csv')

def remove_outlier(filename,outfile):
    df = pd.read_csv(filename)
    label = df['Offer To 1st Close']
    IQR = label.quantile(0.75) - label.quantile(0.25)
    lower_bound = label.quantile(0.25) - 1.5 * IQR
    upper_bound = label.quantile(0.75) + 1.5 * IQR
    df = df[df['Offer To 1st Close'] < upper_bound]
    df = df[df['Offer To 1st Close'] > lower_bound]
    df.to_csv(outfile,index=False)


if __name__ == '__main__':
    convert_offer_to_underpriced('bloomberg_data.csv', 'Label_Converted_data.csv')
    show_conversion('bloomberg_data.csv', 'Label_Converted_data.csv')
    drop_column(['Issuer Tickere','Issuer Name','Filing Term Price Range','cusip','Priced Range'],'Label_Converted_data.csv','Issuer_removed_data.csv')
    show_conversion('Label_Converted_data.csv','Issuer_removed_data.csv')
    encoding('Issuer_removed_data.csv','Encoded_data.csv')
    show_conversion('Issuer_removed_data.csv','Encoded_data.csv')
    show_correlation('Encoded_data.csv','Final_Output_Reg.csv')
    checkimbalance('Final_Output_Reg.csv')
    import os
    
    files_to_keep = ['Final_Output_Reg.csv', 'Final_Output_Class.csv', 'Final_Output_Reg_no_outliers.csv', 'bloomberg_data.csv', 'Final_Output_Class_resampled.csv']
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    for file in csv_files:
        if file not in files_to_keep:
            try:
                os.remove(file)
                print(f"Removed {file}")
            except OSError as e:
                print(f"Error removing {file}: {e}")
    pass