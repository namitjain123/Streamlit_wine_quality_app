import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define the Wine Quality Model class
class WineQualityModel:
    def __init__(self, model_type="Random Forest"):
        self.scaler = MinMaxScaler()
        self.model_type = model_type
        self.model = self._choose_model()

    def _choose_model(self):
        if self.model_type == "Random Forest":
            return RandomForestClassifier(n_estimators=100)
        elif self.model_type == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        elif self.model_type == "SVM":
            return SVC(probability=True)

    def load_data(self, path):
        df = pd.read_excel(path)
        df["quality_bin"] = np.where(df['quality'] > 6.5, 'good quality', 'bad quality')
        self.X = df.drop(["quality", "quality_bin"], axis=1)
        self.Y = df["quality_bin"]
        self.X_scaled = self.scaler.fit_transform(self.X)

    def train(self, test_size=0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_scaled, self.Y, test_size=test_size, random_state=42
        )
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, input_dict):
        df = pd.DataFrame(input_dict, index=[0])
        scaled = self.scaler.transform(df)
        prediction = self.model.predict(scaled)
        confidence = self.model.predict_proba(scaled).max()
        return prediction, confidence

    def predict_batch(self, df):
        feature_columns = self.X.columns  # use only columns used in training
        df_filtered = df[feature_columns]
        scaled = self.scaler.transform(df_filtered)
        return self.model.predict(scaled)


    def evaluate(self):
        pred = self.model.predict(self.X_test)  # Predict values for the test set
        cm = confusion_matrix(self.Y_test, pred)  # Compute confusion matrix

        # Return a dictionary of performance metrics
        return {
            "accuracy": accuracy_score(self.Y_test, pred),  # Accuracy on test set
            "precision": precision_score(self.Y_test, pred, average="weighted", zero_division=1),  # Weighted precision
            "recall": recall_score(self.Y_test, pred, average="weighted", zero_division=1),  # Weighted recall
            "cross_val": cross_val_score(self.model, self.X_scaled, self.Y, cv=5),  # Cross-validation scores (5 folds)
            "confusion_matrix": cm,  # Actual vs predicted class distribution
            "train_acc": accuracy_score(self.Y_train, self.model.predict(self.X_train)),  # Accuracy on training set
            "test_acc": accuracy_score(self.Y_test, pred)  # Reconfirmed accuracy on test set
        }

# UI Part
st.markdown("""
<h1 style='text-align: center; color: purple; font-family: cursive;'>Wine Quality Prediction Model</h1>
""", unsafe_allow_html=True)

image = Image.open('wine.png')
st.image(image, use_column_width=True)

st.markdown("""
## Introduction

Welcome to the Wine Quality Prediction application! This app uses a machine learning model to predict the quality of wine based on various physicochemical properties.

Cheers!! 🍻
""")

# Model selection
model_choice = st.selectbox("Choose ML Model", ["Random Forest", "Logistic Regression", "SVM"])
model_obj = WineQualityModel(model_type=model_choice)
model_obj.load_data("winequality-red.xlsx")
model_obj.train()

# File uploader for batch prediction
st.markdown("## 📂 Upload File for Batch Prediction")
batch_file = st.file_uploader("Upload your wine data file (CSV, XLSX, etc.)")
if batch_file is not None:
    file_extension = os.path.splitext(batch_file.name)[1].lower()
    if file_extension == ".csv":
        batch_df = pd.read_csv(batch_file)
    elif file_extension in [".xls", ".xlsx"]:
        batch_df = pd.read_excel(batch_file)
    elif file_extension == ".txt":
        batch_df = pd.read_csv(batch_file, delimiter='\t')
    else:
        st.warning("Unsupported file format. Please upload a CSV or Excel file.")
        batch_df = None

    if batch_df is not None:
        predictions = model_obj.predict_batch(batch_df)
        batch_df['Prediction'] = predictions
        st.write(batch_df)
        st.download_button("Download Predictions", data=batch_df.to_csv(index=False), file_name="wine_predictions.csv")

# Form for input
st.markdown("## 🧪 Predict Single Wine Sample")
with st.form(key='my_form'):
    fixed_acidity = st.number_input('Enter Fixed Acidity')
    volatile_acidity = st.number_input('Enter Volatile Acidity')
    citric_acid = st.number_input('Enter Citric Acid')
    residual_sugar = st.number_input('Enter Residual Sugar')
    chlorides = st.number_input('Enter Chlorides')
    free_sulfur_dioxide = st.number_input('Enter Free Sulfur Dioxide')
    total_sulfur_dioxide = st.number_input('Enter Total Sulfur Dioxide')
    density = st.number_input('Enter Density')
    ph_value = st.number_input('Enter pH')
    sulphates = st.number_input('Enter Sulphates')
    alcohol = st.number_input('Enter Alcohol')

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])
    submit_button = col3.form_submit_button(label='Predict')
    clear_button = col5.form_submit_button(label='Clear')

inputs = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph_value, sulphates, alcohol]

if all(inputs):
    if submit_button:
        input_dict = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': ph_value,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        prediction, confidence = model_obj.predict(input_dict)
        color = 'green' if prediction[0] == 'good quality' else 'red'
        st.markdown(f"<h2 style='text-align: center; color: {color};'>Prediction: {prediction[0]} ({confidence*100:.2f}% confidence)</h2>", unsafe_allow_html=True)
    elif clear_button:
        st.experimental_rerun()
else:
    st.warning("Please fill out all the inputs! ⚠️")

# Display model evaluation results
metrics = model_obj.evaluate()
st.markdown("## 📊 Model Performance")
st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
st.write(f"**Precision:** {metrics['precision']:.2f}")
st.write(f"**Recall:** {metrics['recall']:.2f}")
st.write(f"**Cross-Validation Scores:** {metrics['cross_val']}")

# Show confusion matrix
st.markdown("## 🔍 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Bar chart of training vs testing accuracy
st.markdown("## 📈 Training vs Testing Accuracy")
fig2, ax2 = plt.subplots()
ax2.bar(["Training", "Testing"], [metrics['train_acc'], metrics['test_acc']], color=['#90ee90','#add8e6'])
ax2.set_ylabel("Accuracy")
st.pyplot(fig2)
