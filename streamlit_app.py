import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6

st.set_page_config(
    page_title='AQI Data App',
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
    hist, edges = np.histogram(df['AQI'], bins=50)
    p1 = figure(title="Distribution of AQI Values", x_axis_label='AQI', y_axis_label='Frequency', plot_height=400, plot_width=700)
    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="skyblue", line_color="black")
    st.bokeh_chart(p1)
    
    # Distribution of AQI categories
    st.subheader("Distribution of AQI Categories")
    category_counts = df['Category'].value_counts()
    p2 = figure(x_range=category_counts.index.tolist(), plot_height=400, plot_width=700, title="Distribution of AQI Categories")
    p2.vbar(x=category_counts.index.tolist(), top=category_counts.values, width=0.9, color=factor_cmap('x', palette=Spectral6, factors=category_counts.index.tolist()))
    p2.xgrid.grid_line_color = None
    p2.y_range.start = 0
    p2.add_tools(HoverTool(tooltips=[("Category", "@x"), ("Count", "@top")]))
    st.bokeh_chart(p2)
    
    # AQI Distribution by Location
    st.subheader("AQI Distribution by Location")
    p3 = figure(x_range=df['Location'].unique().tolist(), plot_height=400, plot_width=700, title="AQI Distribution by Location")
    p3.vbar(x=df['Location'].unique().tolist(), top=df.groupby('Location')['AQI'].mean(), width=0.9, color=factor_cmap('x', palette=Spectral6, factors=df['Location'].unique().tolist()))
    p3.xgrid.grid_line_color = None
    p3.y_range.start = 0
    p3.add_tools(HoverTool(tooltips=[("Location", "@x"), ("Average AQI", "@top")]))
    st.bokeh_chart(p3)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix of Pollutant Concentrations and AQI")
    corr = df[['PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'O3 (ppm)', 'CO (ppm)', 'SO2 (ppm)', 'NO2 (ppm)', 'AQI']].corr()
    p4 = figure(title="Correlation Matrix", plot_height=400, plot_width=700)
    p4.image(image=[corr.values], x=0, y=0, dw=10, dh=10, palette="Viridis256")
    st.bokeh_chart(p4)
    
    # Time series analysis
    st.subheader("Time Series of AQI Values")
    p5 = figure(title="Time Series of AQI Values", x_axis_label='Date', y_axis_label='AQI', x_axis_type='datetime', plot_height=400, plot_width=700)
    p5.line(df['Date'], df['AQI'], line_width=2)
    st.bokeh_chart(p5)
    
    # AQI over time by location
    st.subheader("Time Series of AQI Values by Location")
    p6 = figure(title="Time Series of AQI Values by Location", x_axis_label='Date', y_axis_label='AQI', x_axis_type='datetime', plot_height=400, plot_width=700)
    for loc in df['Location'].unique():
        loc_data = df[df['Location'] == loc]
        p6.line(loc_data['Date'], loc_data['AQI'], legend_label=loc, line_width=2)
    p6.legend.title = 'Location'
    st.bokeh_chart(p6)

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
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    p7 = figure(title="Confusion Matrix", x_axis_label='Predicted', y_axis_label='Actual', plot_height=400, plot_width=700)
    p7.image(image=[cm_df.values], x=0, y=0, dw=10, dh=10, palette="Blues256")
    st.bokeh_chart(p7)
    
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
