import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    # Load the data from the excel file
    df = pd.read_excel(file_path)
    return df

def remove_nan_columns(df):
    # Remove columns with excessive NaN or zero values
    for c in df.columns:
        nan_count = df[c].isnull().sum()
        zero_count = sum(df[c] == 0)
        if nan_count > 100 or zero_count > 1700:
            df = df.drop(c, axis=1)
    return df


def calculate_position_ratio(df):
    # Calculate position ratio from geometryText and replace null values with mean
    mean_0 = []
    mean_1 = []
    position_ratio = []

    for i in range(len(df['geometryText'])):
        if df['geometryText'][i] is not np.nan:
            arr = df['geometryText'][i].split(',')
            left = arr[1].split(': ')[1]
            top = arr[2].split(': ')[1]
            ratio = int(left) / int(top)
            position_ratio.append(ratio)
            if df['is_shop'][i] == 1:
                mean_1.append(ratio)
            else:
                mean_0.append(ratio)
        else:
            position_ratio.append(-1)

    mean_1 = sum(mean_1) / len(mean_1)
    mean_0 = sum(mean_0) / len(mean_0)

    for i in range(len(position_ratio)):
        if position_ratio[i] == -1:
            if df['is_shop'][i] == 1:
                position_ratio[i] = mean_1
            else:
                position_ratio[i] = mean_0

    df['position_ratio'] = position_ratio
    return df

def remove_useless_columns(df):
    # Remove useless columns
    df = df.drop(['applied_score', 'classListSubtree', 'geometryText'], axis=1)
    return df

def encode_categorical_columns(df):
    # Encode categorical columns and drop irrelevant ones
    le = LabelEncoder()
    for c in df.columns:
        if df[c].dtype.name == 'object':
            if c != 'cssPosition' or c != 'html_tag':
                df = df.drop(c, axis=1)
            else:
                df[c] = df[c].apply(le.fit_transform)
    return df

def scale_features(X):
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def check_balance(y):
    print('Slightly imbalanced ', (sum(y == 0) / sum(y == 1)))