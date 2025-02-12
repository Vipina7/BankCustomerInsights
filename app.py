import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

# App title
st.title('Bank Marketing Campaign - Customer Segmentation')

st.write(
    "This application predicts customer responses to bank marketing campaigns and segments them based on various demographic and financial attributes. "
    "Adjust the input fields below to enter customer details and get predictions."
)

balance = st.number_input('Account Balance', min_value=-6000, max_value=82000, value=500.)
duration = st.number_input('Call Duration', min_value=2, max_value=3880, value=150)
campaign = st.slider('Campaign contacts', min_value=1, max_value=63, value=3)
previous = st.slider('Previous Campaign contacts', min_value=0, max_value=60, value=1)
job = st.selectbox('Job Type', ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'])
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
education = st.selectbox('Education Level', ['secondary', 'tertiary', 'primary', 'unknown'])
default = st.selectbox('Credit Default', ['no', 'yes'])
housing = st.selectbox('Housing Loan', ['yes', 'no'])
loan = st.selectbox('Personal Loan', ['no', 'yes'])
contact = st.selectbox('Contact Type', ['unknown', 'cellular', 'telephone'])
poutcome = st.selectbox('Previous campaign Outcome', ['unknown', 'other', 'failure', 'success'])
month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
deposit = st.selectbox('Subscribed', ['yes', 'no'])