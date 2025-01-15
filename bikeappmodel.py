import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title("Bike Sharing Demand Prediction")

# Function to train the model and calculate metrics
def train_model(train_file):
    data = pd.read_csv(train_file)
    
    # Prepare the data (similar to the uploaded script logic)
    numcols = data[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]
    objcols_dummy = pd.get_dummies(objcols, columns=objcols.columns)
    final_data = pd.concat([numcols, objcols_dummy], axis=1)

    # Split into X and y
    y = final_data['cnt']
    X = final_data.drop(columns=['cnt', 'atemp', 'registered'])  # Drop multicollinear columns

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate R² and RMSE
    predictions = model.predict(X)
    r2_score = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return model, r2_score, rmse

# Function to evaluate the model on test data
def evaluate_model(test_file, model):
    data = pd.read_csv(test_file)

    # Prepare the test data (should match training data processing)
    numcols = data[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]
    objcols_dummy = pd.get_dummies(objcols, columns=objcols.columns)
    final_data = pd.concat([numcols, objcols_dummy], axis=1)

    # Ensure same columns as training data
    X_test = final_data.drop(columns=['cnt', 'atemp', 'registered'])
    y_test = final_data['cnt']

    # Predict and calculate RMSE
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return rmse

# File upload for training
data_train = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
if data_train is not None:
    with st.spinner('Training the model...'):
        model, r2_score, rmse_train = train_model(data_train)
        st.success('Model trained successfully!')
        st.write(f"### Training Metrics")
        st.write(f"- R² Score: {r2_score:.4f}")
        st.write(f"- RMSE: {rmse_train:.4f}")

# File upload for testing
data_test = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
if data_test is not None and 'model' in locals():
    with st.spinner('Evaluating the model on test data...'):
        rmse_test = evaluate_model(data_test, model)
        st.success('Evaluation completed!')
        st.write(f"### Test Data RMSE")
        st.write(f"- RMSE: {rmse_test:.4f}")
elif data_test is not None:
    st.warning("Please upload training data first to build the model.")



