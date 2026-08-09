import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # TotalCharges has some blank strings, not true NaN - fix that
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID - not a useful feature
    df = df.drop('customerID', axis=1)
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def encode_categoricals(df):
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

if __name__ == '__main__':
    df = load_and_clean_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print('After cleaning:')
    print(df.info())
    
    df_encoded = encode_categoricals(df)
    print('\nAfter encoding:')
    print(df_encoded.shape)
    print(df_encoded.head())
