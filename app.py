import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------
# App Title
# -----------------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App (Random Forest)")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes_dataset.csv")
    return data

df = load_data()

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------
df_encoded = df.copy()

le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df_encoded["gender"] = le_gender.fit_transform(df_encoded["gender"])
df_encoded["smoking_history"] = le_smoking.fit_transform(df_encoded["smoking_history"])

# -----------------------------------
# Features & Target
# -----------------------------------
X = df_encoded.drop("diabetes", axis=1)
y = df_encoded["diabetes"]

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Model Training
# -----------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------
# Sidebar Input (ALL INT VALUES)
# -----------------------------------
st.sidebar.header("Enter Patient Details")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
smoking_history = st.sidebar.selectbox("Smoking History", df["smoking_history"].unique())
bmi = st.sidebar.number_input("BMI (Integer Only)", min_value=10, max_value=60, step=1)
hba1c = st.sidebar.number_input("HbA1c Level (Integer Only)", min_value=1, max_value=15, step=1)
glucose = st.sidebar.number_input("Blood Glucose Level", min_value=50, max_value=300, step=1)

# -----------------------------------
# Prediction
# -----------------------------------
if st.sidebar.button("Predict"):

    input_data = pd.DataFrame({
        "gender": [le_gender.transform([gender])[0]],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [le_smoking.transform([smoking_history])[0]],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The patient is Diabetic")
    else:
        st.success("✅ The patient is NOT Diabetic")

# -----------------------------------
# Model Accuracy
# -----------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("### Model Accuracy:")
st.write(f"{accuracy:.2f}")