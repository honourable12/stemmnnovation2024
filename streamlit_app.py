import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
st.set_page_config(
    page_title='AQI data app',
    page_icon='🌡️',
)
# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('assests/aqi_dataset_1500_days.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# App Title
st.title("AQI Data Analysis and Prediction for Precision Agriculture")


# Sidebar for navigation
st.sidebar.image("assests/icon.jpeg", caption="logo", use_column_width=True)
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Select a page:", ["Exploratory Data Analysis", "Prediction Model"])

if options == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Display dataset
    st.subheader("Dataset")
    st.dataframe(df.head())
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Distribution of AQI values
    st.subheader("Distribution of AQI Values")
    fig, ax = plt.subplots()
    sns.histplot(df['AQI'], bins=50, kde=True, color='skyblue', ax=ax)
    st.pyplot(fig)
    
    # Distribution of AQI categories
    st.subheader("Distribution of AQI Categories")
    fig, ax = plt.subplots()
    sns.countplot(x='Category', data=df, palette='viridis', order=df['Category'].value_counts().index, ax=ax)
    st.pyplot(fig)
    
    # AQI Distribution by Location
    st.subheader("AQI Distribution by Location")
    fig, ax = plt.subplots()
    sns.boxplot(x='Location', y='AQI', data=df, palette='Set3', ax=ax)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix of Pollutant Concentrations and AQI")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df[['PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'O3 (ppm)', 'CO (ppm)', 'SO2 (ppm)', 'NO2 (ppm)', 'AQI']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    # Time series analysis
    st.subheader("Time Series of AQI Values")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(x='Date', y='AQI', data=df, ci=None, ax=ax)
    st.pyplot(fig)

    # AQI over time by location
    st.subheader("Time Series of AQI Values by Location")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(x='Date', y='AQI', hue='Location', data=df, ci=None, ax=ax)
    st.pyplot(fig)

elif options == "Prediction Model":
    st.header("AQI Prediction Model using Logistic Regression")
    
    # Convert AQI values to categories
    def categorize_aqi(aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    df['AQI_Category'] = df['AQI'].apply(categorize_aqi)

    # Features and target variable
    X = df[['PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'O3 (ppm)', 'CO (ppm)', 'SO2 (ppm)', 'NO2 (ppm)']]
    y = df['AQI_Category']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model performance
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Model Performance")
    st.write(report_df)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    # User input for prediction
    st.subheader("Predict AQI Category for New Data")
    pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=500.0, value=50.0)
    o3 = st.number_input("O3 (ppm)", min_value=0.0, max_value=300.0, value=0.1)
    co = st.number_input("CO (ppm)", min_value=0.0, max_value=10.0, value=0.1)
    so2 = st.number_input("SO2 (ppm)", min_value=0.0, max_value=100.0, value=0.1)
    no2 = st.number_input("NO2 (ppm)", min_value=0.0, max_value=200.0, value=0.1)
    
    new_data = np.array([[pm25, pm10, o3, co, so2, no2]])
    new_aqi_category = model.predict(new_data)
    
    st.write(f"Predicted AQI Category: {new_aqi_category[0]}")
