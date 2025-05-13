import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_credit_data(client_id):
    df = pd.read_csv('data/credit_data.csv')
    categorical = df.select_dtypes(include='object').columns.tolist()

    enc = OneHotEncoder(sparse_output=False)
    enc_data = enc.fit_transform(df[categorical])
    encoded_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(categorical))

    num_df = df.select_dtypes(exclude='object')
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)

    df_final = pd.concat([scaled_df, encoded_df], axis=1)

    y = df_final['credit_risk_good']
    X = df_final.drop(['credit_risk_good', 'credit_risk_bad'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=client_id, test_size=0.2)
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
