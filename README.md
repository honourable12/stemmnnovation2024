# stemmnnovation2024
AQI Data Analysis and Prediction for Precision Agriculture

This Streamlit app provides an exploratory data analysis (EDA) of Air Quality Index (AQI) data and a logistic regression-based prediction model. The app is designed to help in understanding air quality and making predictions that could be useful for agricultural research and precision farming.
Table of Contents

    Features
    Installation
    Usage
    File Structure
    Dependencies
    License

Features

    Exploratory Data Analysis (EDA)
        Display the dataset
        Basic statistical summary
        Distribution of AQI values and categories
        AQI distribution by location
        Correlation matrix of pollutant concentrations and AQI
        Time series analysis of AQI values

    Prediction Model
        Logistic regression model to predict AQI categories
        Model performance evaluation with classification report and confusion matrix
        User input for predicting AQI categories for new data

Installation

    Clone the repository

    bash

git clone <repository_url>
cd <repository_directory>

Install required dependencies

bash

    pip install -r requirements.txt

    Ensure dataset and image files are in the correct locations
        Place aqi_dataset_1500_days.csv in the assets directory.
        Place icon.jpeg in the assets directory.

Usage

    Run the Streamlit app

    bash

    streamlit run app.py

    Navigate the app
        Use the sidebar to switch between the "Exploratory Data Analysis" and "Prediction Model" pages.
        Explore various visualizations and model predictions.

File Structure

lua

|-- assets
|   |-- aqi_dataset_1500_days.csv
|   |-- icon.jpeg
|-- app.py
|-- README.md
|-- requirements.txt

Dependencies

    streamlit
    pandas
    numpy
    scikit-learn
    bokeh

Install the dependencies using:

bash

pip install -r requirements.txt

requirements.txt:

streamlit
pandas
numpy
scikit-learn
bokeh
