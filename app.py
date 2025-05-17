import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
from sklearn.metrics import f1_score

# Function to generate downloadable CSV link
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
    return href

# App Title
st.title("Heart Disease Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# ---------------- TAB 1 ----------------
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Input data for prediction
    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain], 'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol], 'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
        'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina], 'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
    modelnames = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl', 'grid_random_forest_model.pkl']

    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("Submit"):
        st.subheader('Results...')
        st.markdown('----------------------')
        result = predict_heart_disease(input_data)
        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('----------------')

# ---------------- TAB 2 ----------------
with tab2:
    st.title("Upload CSV File")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticR.pkl", "rb"))

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['prediction LR'] = model.predict(input_data[expected_columns])
            st.write(input_data)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("Please ensure the CSV has the correct columns.")

# ---------------- TAB 3 ----------------
with tab3:
    accuracy_data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 82.93
    }

    f1_data = {
        'Decision Trees': 0.79,
        'Logistic Regression': 0.85,
        'Random Forest': 0.83,
        'Support Vector Machine': 0.81
    }

    Models = list(accuracy_data.keys())
    Accuracies = list(accuracy_data.values())
    F1_Scores = [f1_data[model] for model in Models]

    df = pd.DataFrame({'Models': Models, 'Accuracies': Accuracies, 'F1 Scores': F1_Scores})

    # Accuracy Bar Chart
    fig_acc = px.bar(df, x='Models', y='Accuracies', title="Model Accuracies", text='Accuracies')
    fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_acc.update_layout(yaxis_range=[0, 100])

    # F1 Score Line Chart
    fig_f1 = px.line(df, x='Models', y='F1 Scores', title="Model F1 Scores", markers=True, text='F1 Scores')
    fig_f1.update_traces(texttemplate='%{text:.2f}', textposition='top center')
    fig_f1.update_layout(yaxis_range=[0, 1])

    st.plotly_chart(fig_acc)
    st.plotly_chart(fig_f1)
