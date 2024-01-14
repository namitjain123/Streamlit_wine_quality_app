#importing libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# reading and printing the dataset
data = pd.read_excel(r"C:\Users\ameen\OneDrive\Desktop\OneDrive\winequality-red.xlsx")

# Convert the wine quality into a binary variable, good or bad, based on a threshold value of 6.5
data["quality_bin"] = np.where(data['quality'] > 6.5, 'good quality', 'bad quality')

# Splitting the into features and target
X = data.drop(["quality", "quality_bin"], axis=1)
Y = data["quality_bin"]

# Scale the features
min_max_scaler = MinMaxScaler()
X_min_max_scaled = min_max_scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_min_max_scaled, Y, test_size= 0.2, random_state = 42)

# Create and fit the Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# Heading
st.markdown("<h1 style='text-align: center; color: purple; font-family: cursive;'>Wine Quality Prediction model</h1>", unsafe_allow_html=True)

# Display an image
image = Image.open(r"C:\Users\ameen\OneDrive\Pictures\wine.png")
st.image(image, use_column_width=True)

st.markdown("""
## Introduction

Welcome to the Wine Quality Prediction application! This application uses a machine learning model to predict the quality of wine based on various physicochemical properties. The model has been trained on a dataset of red wines, each labeled as either 'good' or 'bad' based on a quality score threshold of 6.5.

The model takes as input the following features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. These features are used to predict whether a given wine is 'good' or 'bad'.

## How to Use

To use the application, simply input the values for the aforementioned features. Once you've input all the necessary information, click the 'Predict' button. The application will then use the machine learning model to predict the quality of the wine and display the result.

Please note that this is a prediction based on the model's training and may not always accurately reflect the true quality of the wine. Always trust your own palate when it comes to enjoying wine!

Cheers!! 🍻
""")

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

# Check if all inputs have been filled out
inputs = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph_value, sulphates, alcohol]

if all(inputs):
    # Make predictions and display them
    if submit_button:
        df = {'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity, 'citric acid': citric_acid, 'residual sugar': residual_sugar, 'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide, 'total sulfur dioxide': total_sulfur_dioxide, 'density': density, 'pH': ph_value, 'sulphates': sulphates, 'alcohol': alcohol}
        
        wine = pd.DataFrame(df, index=[0])
        wine_scaled = min_max_scaler.transform(wine)
        prediction = model.predict(wine_scaled)

        # Display the prediction result
        if prediction[0] == 'bad quality':
            st.markdown(f"<h2 style='text-align: center; color: red;'>Prediction: {prediction[0]}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; color: green;'>Prediction: {prediction[0]}</h2>", unsafe_allow_html=True)
    elif clear_button:
        st.experimental_rerun()
else:
    st.warning("Please fill out all the inputs! ⚠️")

# Make predictions on the testing set
Y_pred = model.predict(X_test)

# Calculate precision, accuracy and recall
precision = precision_score(Y_test, Y_pred, average = "weighted", zero_division=1)
print("Precision score:", precision)

accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy: %.2f' % accuracy)

recall = metrics.recall_score(Y_test, Y_pred, average="weighted")
print('Recall: %.2f' % recall)

# Perform cross-validation
scores = cross_val_score(model, X_min_max_scaled, Y, cv=5)
print('Cross-validation scores: ', scores)

# Evaluation of the model on training & testing data
pred_on_training_data = model.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, pred_on_training_data)

print(accuracy_on_training_data)

pred_on_testing_data = model.predict(X_test)
accuracy_on_testing_data = accuracy_score(Y_test, pred_on_testing_data)

print(accuracy_on_testing_data)