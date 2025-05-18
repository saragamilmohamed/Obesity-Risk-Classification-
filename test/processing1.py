import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

def preprocess_input(data):
    # Load encoders and scaler
    # label_gender = joblib.load('label_Gender.pkl')
    # label_family = joblib.load('label_family_history_with_overweight.pkl')
    # label_smoke = joblib.load('label_SMOKE.pkl')
    # label_favc = joblib.load('label_FAVC.pkl')
    # label_scc = joblib.load('label_SCC.pkl')
    # ordinal_encoder = joblib.load('ordinal_encoder.pkl')
    ohe = joblib.load('ohe_mtrans.pkl')
    # scaler = joblib.load('scaler.pkl')
    # required_columns = joblib.load('required_columns.pkl')

    # 🔹 Clean and normalize string inputs
    # for col in ['Gender', 'family_history_with_overweight', 'SMOKE', 'FAVC', 'SCC', 'CAEC', 'CALC', 'MTRANS']:
    #     input_df[col] = input_df[col].str.lower()

    # # 🔹 Label Encoding
    # input_df['Gender'] = label_gender.transform(input_df['Gender'])
    # input_df['family_history_with_overweight'] = label_family.transform(input_df['family_history_with_overweight'])
    # input_df['SMOKE'] = label_smoke.transform(input_df['SMOKE'])
    # input_df['FAVC'] = label_favc.transform(input_df['FAVC'])
    # input_df['SCC'] = label_scc.transform(input_df['SCC'])

    # # 🔹 Ordinal Encoding for CAEC and CALC
    # input_df[['CAEC', 'CALC']] = ordinal_encoder.transform(input_df[['CAEC', 'CALC']])

    # # 🔹 One-hot Encoding for MTRANS
    # mtrans_encoded = ohe.transform(input_df[['MTRANS']])
    # mtrans_encoded_df = pd.DataFrame(mtrans_encoded, columns=ohe.get_feature_names_out(['MTRANS']))
    # input_df = pd.concat([input_df.reset_index(drop=True), mtrans_encoded_df.reset_index(drop=True)], axis=1)
    # input_df.drop(columns=['MTRANS'], inplace=True)

    # # 🔹 Adjust numerical fields
    # input_df['Height'] = np.round(input_df['Height'] * 100)   # meters to centimeters
    # input_df['NCP'] = np.round(input_df['NCP'])               # round meals
    # input_df['Age'] = np.round(input_df['Age'])               # round age
    # input_df['CH2O'] = np.ceil(input_df['CH2O'])              # ceil water consumption
    # input_df['FAF'] = np.round(input_df['FAF'])               # round physical activity
    # input_df['TUE'] = np.round(input_df['TUE'] * 60)          # hours to minutes

    # # 🔹 Final preprocessing: reorder and scale
    # input_df = input_df[required_columns]
    # input_df_scaled = scaler.transform(input_df)
    # ===========================================
    #if 'id' not in data.columns:
   #     data['id'] = 0 

    data['Age'] = np.round(data['Age'])
    data['Height'] = np.round(data['Height'] * 100)
    data['NCP'] = np.round(data['NCP'])
    data['CH2O'] = np.ceil(data['CH2O'])
    data['FAF'] = np.round(data['FAF'])
    data['TUE'] = np.round(data['TUE'] * 60)
    data_cat = data.select_dtypes(include=['object', 'category']).columns.tolist()
    label_encoding_categorical_feature = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    ordinal_encoding_categorical_feature = ["CAEC", "CALC"]
    one_hot_encoding_categorical_feature = ["MTRANS"]
    for col in label_encoding_categorical_feature:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])


    ordinal_categories = {
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],  
    "CALC": ["no", "Sometimes", "Frequently"]  
    }

    ordinal_encoder = OrdinalEncoder(
        categories=[
            ordinal_categories[col] for col in ordinal_encoding_categorical_feature
        ]
    )

    # Fit and transform the encoder on the entire subset of columns at once
    data[ordinal_encoding_categorical_feature] = ordinal_encoder.fit_transform(data[ordinal_encoding_categorical_feature])

    #ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid dummy variable trap
# One-hot encode MTRANS and concatenate
    mtrans_encoded = ohe.transform(data[['MTRANS']])
    mtrans_encoded_df = pd.DataFrame(mtrans_encoded, columns=ohe.get_feature_names_out(['MTRANS']))

    # Append and drop original column
    data = pd.concat([data.reset_index(drop=True), mtrans_encoded_df.reset_index(drop=True)], axis=1)
    data.drop(columns=['MTRANS'], inplace=True)

    return data


# (Optional) If you don’t want the one-hot encoded columns at all, don't concatenate them


    # Drop the original 'MTRANS' column
    #data.drop(columns=['MTRANS'], inplace=True)


    #return data