import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
from PIL import Image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Background and fonts */
    .stApp {
        background-color: #F5F7FA;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1, h2, h3 {
        color: #004C99;
        font-weight: bold;
        text-align: center;
    }

    /* Paragraph text */
    p {
        color: #333333;
        text-align: center;
        font-size: 16px;
    }

    /* Inputs - keep native dropdown style */
    .stSelectbox, .stNumberInput {
        border-radius: 6px !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0066CC;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #004C99;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown(
    """
    <div style='text-align: center; color: #004080;'>
        <h3>👇 Scroll down to use the Predictor 👇</h3>
        <p style='font-size:16px;'>Just fill the details and click <b>Predict</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
banner = Image.open("Hospital.png")
st.image(banner, use_container_width=True, output_format="auto", clamp=True)
st.title("🏥 Heart Disease Predictor")
st.write("Your trusted AI-powered health screening tool.")

# ==================== DOWNLOAD FUNCTION ====================
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="color:#1E90FF; font-weight:bold;">📥 Download Predictions CSV</a>'

# ==================== MODEL LOADING FUNCTION ====================
algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'SVM', 'GridRandom']
modelnames = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl', 'grid_random_forest_model.pkl']

def predict_heart_disease(data):
    predictions = []
    for modelname in modelnames:
        with open(modelname, 'rb') as file:
            model = pickle.load(file)
            prediction = model.predict(data)
            predictions.append(prediction)
    return predictions

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(['🩺 Predict', '📂 Bulk Predict', '📊 Model Information'])

# ---------- TAB 1: Single Prediction ----------
with tab1:
    st.subheader("Enter Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
        'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
    })

    if st.button("🚀 Predict"):
        st.subheader('Results')
        result = predict_heart_disease(input_data)
        for i in range(len(result)):
            st.markdown(f"**{algonames[i]}**: {'✅ No disease' if result[i][0] == 0 else '⚠️ Disease detected'}")

# ---------- TAB 2: Bulk Predictions ----------
with tab2:
    st.subheader("📂 Bulk Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticR.pkl", "rb"))
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        if set(expected_columns).issubset(input_data.columns):
            input_data['prediction LR'] = model.predict(input_data[expected_columns])
            st.dataframe(input_data)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please ensure the CSV has the correct columns.")

# ---------- TAB 3: Model Info ----------
with tab3:
    st.subheader("📊 Model Performance")
    accuracy_data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'SVM': 82.93
    }
    f1_data = {
        'Decision Trees': 0.79,
        'Logistic Regression': 0.85,
        'Random Forest': 0.83,
        'SVM': 0.81
    }
    df = pd.DataFrame({
        'Models': list(accuracy_data.keys()),
        'Accuracies': list(accuracy_data.values()),
        'F1 Scores': [f1_data[m] for m in accuracy_data]
    })

    fig_acc = px.bar(df, x='Models', y='Accuracies', title="Model Accuracies", text='Accuracies',
                     color='Models', color_discrete_sequence=['#1E90FF']*len(df))
    fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_acc.update_layout(yaxis_range=[0, 100])

    fig_f1 = px.line(df, x='Models', y='F1 Scores', title="Model F1 Scores",
                     markers=True, text='F1 Scores')
    fig_f1.update_traces(texttemplate='%{text:.2f}', textposition='top center', line_color='#1E90FF')
    fig_f1.update_layout(yaxis_range=[0, 1])

    st.plotly_chart(fig_acc, use_container_width=True)
    st.plotly_chart(fig_f1, use_container_width=True)
