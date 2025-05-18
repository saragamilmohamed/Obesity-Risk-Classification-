import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

def preprocess_input(data):
    data = pd.DataFrame(data)
    # Round and transform numerical fields
    data['Age'] = np.round(data['Age'])
    data['Height'] = np.round(data['Height'] * 100)
    data['NCP'] = np.round(data['NCP'])
    data['CH2O'] = np.ceil(data['CH2O'])
    data['FAF'] = np.round(data['FAF'])
    data['TUE'] = np.round(data['TUE'] * 60)

    # Full class mappings for label encoded columns
    label_classes = {
        "Gender": ['Male', 'Female'],
        "family_history_with_overweight": ['yes', 'no'],
        "FAVC": ['yes', 'no'],
        "SMOKE": ['no', 'yes'],
        "SCC": ['no', 'yes'],
    }

    # Apply Label Encoding using predefined classes
    for col, classes in label_classes.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        data[col] = le.transform(data[col])

    # Ordinal encoding
    ordinal_categories = {
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "CALC": ["no", "Sometimes", "Frequently"]
    }

    ordinal_features = list(ordinal_categories.keys())
    ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_features])
    data[ordinal_features] = ordinal_encoder.fit_transform(data[ordinal_features])

    # One-hot encoding
    # === One-Hot Encoding for MTRANS with fixed categories ===
    mtrans_categories = ['Automobile' , 'Bike',  'Motorbike', 'Public_Transportation', 'Walking']
    ohe = OneHotEncoder(categories=[mtrans_categories], sparse_output=False)

    if 'MTRANS' not in data.columns:
        raise ValueError("MTRANS column is missing from input data")

    val = data.loc[0, 'MTRANS']
    if val not in mtrans_categories:
        raise ValueError(f"Unknown category '{val}' in MTRANS")

    mtrans_encoded = ohe.fit_transform(data[['MTRANS']])
    mtrans_encoded_df = pd.DataFrame(mtrans_encoded, columns=ohe.get_feature_names_out(['MTRANS']))

    # Join encoded MTRANS and drop original column
    data = pd.concat([data.reset_index(drop=True), mtrans_encoded_df.reset_index(drop=True)], axis=1)
    data.drop(columns=['MTRANS'], inplace=True)

    return data
