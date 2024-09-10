import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("C:\\Users\\91762\\Desktop\\BT\\brain_stroke.csv")

# Encode categorical variables
enc_work_type = LabelEncoder()
enc_gender = LabelEncoder()
enc_smoking_status = LabelEncoder()
enc_residence_type = LabelEncoder()
enc_ever_married = LabelEncoder()

data['work_type'] = enc_work_type.fit_transform(data['work_type'])
data['gender'] = enc_gender.fit_transform(data['gender'])
data['smoking_status'] = enc_smoking_status.fit_transform(data['smoking_status'])
data['Residence_type'] = enc_residence_type.fit_transform(data['Residence_type'])
data['ever_married'] = enc_ever_married.fit_transform(data['ever_married'])

# Prepare features and target
X = data.drop('stroke', axis=1)
y = data['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101, stratify=y
)

# Standardize the data
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

# Resample the data
st = SMOTETomek()
X_train_re, y_train_re = st.fit_resample(X_train_std, y_train)

# Train the model
rf = RandomForestClassifier()
rf.fit(X_train_re, y_train_re)

# Streamlit app
st.title("Stroke Risk Prediction")

# Input fields
work_type = st.selectbox("Work Type", ["Govt_job", "Private", "Self_employed", "Children"])
gender = st.selectbox("Gender", ["Male", "Female"])
smoking_status = st.selectbox("Smoking Status", ["Unknown", "Formerly_smoked", "Never_smoked", "Smokes"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
hypertension = st.number_input("Hypertension (0/1)", min_value=0, max_value=1, step=1)
heart_disease = st.number_input("Heart Disease (0/1)", min_value=0, max_value=1, step=1)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, step=0.1)

# Encoding input values
input_data = pd.DataFrame({
    'work_type': [enc_work_type.transform([work_type])[0]],
    'gender': [enc_gender.transform([gender])[0]],
    'smoking_status': [enc_smoking_status.transform([smoking_status])[0]],
    'Residence_type': [enc_residence_type.transform([residence_type])[0]],
    'ever_married': [enc_ever_married.transform([ever_married])[0]],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi]
})

# Standardize the input data
input_data_std = std.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = rf.predict(input_data_std)
    if prediction[0] == 1:
        st.write("You are at risk of stroke.")
    else:
        st.write("You are not at risk of stroke.")
