import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Prediction using Random Forest")
st.write("### Enter Patient Details Below")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_dataset.csv")

df = load_data()

# -------------------------------------------------
# Encoding
# -------------------------------------------------
df_encoded = df.copy()

le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df_encoded["gender"] = le_gender.fit_transform(df_encoded["gender"])
df_encoded["smoking_history"] = le_smoking.fit_transform(df_encoded["smoking_history"])

X = df_encoded.drop("diabetes", axis=1)
y = df_encoded["diabetes"]

# -------------------------------------------------
# Train Model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------
# Layout Columns
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Information")

    gender = st.selectbox("Gender", df["gender"].unique())
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    smoking_history = st.selectbox("Smoking History", df["smoking_history"].unique())

with col2:
    st.subheader("🧪 Medical Details")

    bmi = st.number_input("BMI (Integer)", min_value=10, max_value=60, step=1)
    hba1c = st.number_input("HbA1c Level (Integer)", min_value=1, max_value=15, step=1)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, step=1)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Diabetes"):

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
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {probability:.2f}")

# -------------------------------------------------
# Model Performance Section
# -------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Accuracy: {accuracy:.2f}")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.matshow(cm)
plt.title("Confusion Matrix")
plt.colorbar(ax.matshow(cm))

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f"{val}", ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

