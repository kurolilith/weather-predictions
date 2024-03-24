# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger


import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Weather predictions",
        page_icon=":)",
    )

# Display title
st.title('Random Forest')

# Display subheader for raw data
st.subheader('Raw Data')

# Read the CSV data into a DataFrame
data = 'seattle-weather.csv'
df = pd.read_csv(data)

# Display the dataset
st.write(df)

# Display numerical plots section
st.write('### Display Numerical Plots')

# Select box to choose which numerical feature to plot
feature_to_plot = st.selectbox('Select a numerical feature to plot', ['precipitation', 'temp_max', 'temp_min', 'wind'])

# Plot the selected numerical feature
if feature_to_plot:
    st.write(f'Distribution of {feature_to_plot}:')
    fig = plt.figure(figsize=(10, 6))
    plt.hist(df[feature_to_plot], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(feature_to_plot)
    plt.ylabel('Count')
    st.pyplot(fig)
    
# Display categorical plots section
st.write('### Display Categorical Plots')

# Select box to choose which categorical feature to plot
feature_to_plot = st.selectbox('Select a feature to plot', ['weather'])

# Plot the selected categorical feature
if feature_to_plot:
    st.write(f'Distribution of {feature_to_plot}:')
    bar_chart = st.bar_chart(df[feature_to_plot].value_counts())

# Display relationships section
st.write('### Display Relationships')

# Create dropdown menus for user selection of variables
x_variable = st.selectbox('Select x-axis variable:', df.columns)
y_variable = st.selectbox('Select y-axis variable:', df.columns)
color_variable = st.selectbox('Select color variable:', df.columns)
size_variable = st.selectbox('Select size variable:', df.columns)

# Create scatter plot using Plotly Express
fig = px.scatter(df, x=x_variable, y=y_variable, color=color_variable, size=size_variable, hover_data=[color_variable])

# Display the plot
st.plotly_chart(fig)

# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract month
df['month'] = df['date'].dt.month

# Encode categorical variable 'weather'
df['weather_encode'] = LabelEncoder().fit_transform(df['weather'])

# Transform the 'weather' variable using Box-Cox transformation
df['charges_transform'], lambda_value = stats.boxcox(df['weather_encode'])

# Define features (X) and target (y) and remove duplicate features that will not be used in the model
X = df[['month','precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather_encode']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Instantiate a Random forest classifier
rf_model = RandomForestClassifier(n_estimators = 733, max_depth = None, max_features = 'sqrt', min_samples_split = 11, min_samples_leaf = 10, bootstrap = False)

# Fit the Random forest model using the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Create section for user to predict their own weather
st.write('## Predict Your Own Weather')

# User input for features
month = st.slider('Month', min_value=df['month'].min(), max_value=df['month'].max())
precipitation = st.slider('precipitation', min_value=df['precipitation'].min(), max_value=df['precipitation'].max(), value=0)
temp_min = st.slider('Minimum temperature', min_value=df['min_temp'].min(), max_value=df['min_temp'].max(), value=0, format="%d")
temp_max = st.slider('Maximum temperature', min_value=df['max_temp'].min(), max_value=df['max_temp'].max(), value=0, format="%d")
wind = st.slider('Wind speed', min_value=df['wind'].min(), max_value=df['wind'].max())

# Predict charges for user input
predicted_charges_transformed = rf_model.predict([[month, precipitation, temp_min, temp_max, wind]])

# Reverse the Box-Cox transformation to get the predicted charges
predicted_charges = inv_boxcox(predicted_charges_transformed, lambda_value)

# Display the predicted charges
st.write('Predicted Charges:', round(predicted_charges[0], 0))

# Run the Streamlit app
if __name__ == '__main__':
    run()





