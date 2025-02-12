import streamlit as st
import pandas as pd
import numpy as np
import sys

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

# App title
st.title('Bank Marketing Campaign - Customer Segmentation')

st.write(
    "This application predicts customer responses to bank marketing campaigns and segments them based on various demographic and financial attributes. "
    "Adjust the input fields below to enter customer details and get predictions."
)

balance = st.number_input('Account Balance', min_value=-6000, max_value=82000, value=500)
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


if st.button("Predict Cluster"):
    try:
        data = CustomData(
            balance,
            duration,
            campaign,
            previous,
            job,
            marital,
            education,
            default,
            housing,
            loan,
            contact,
            poutcome,
            month
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        if prediction == 0:
            st.subheader("📊 Customer belongs to Cluster 1/2:")
            st.write("- Customers in this cluster have **lower engagement** and a **lower likelihood of subscribing** (~45%).")
            st.write("- **Key Factors Influencing Deposits:**")
            st.write(f"  - 📞 **Call Duration:** Longer calls (>483s) improve engagement.")
            st.write(f"  - 💰 **Balance:** Customers with a balance >$1,612 are more likely to deposit.")
            st.write(f"  - 📅 **Month:** March (88%) and April (63%) show higher deposit rates, while May is the lowest (33%).")
            st.write(f"  - 🔄 **Contacts:** Customers contacted 2-3 times respond better.")
            st.write(f"  - 📊 **Previous Interactions:** Past contact history slightly increases deposit chances.")

            st.success("### 🎯 Key Suggestions:")
            st.write("📞 **Increase follow-up calls** and personalize offers.")
            st.write("🎯 **Target with promotional discounts** or incentives.")
            st.write("🕵️ **Analyze past interactions** to improve engagement.")

            # Show toast message and balloons
            st.toast("✅ Customer analysis complete! Key insights identified to improve subscription likelihood.")
            st.balloons()

        elif prediction == 1:
            st.subheader("📊 Customer belongs to Cluster 2/2:")
            st.write("- Customers in this cluster are **highly engaged** and have a **higher likelihood of subscribing** (~50%).")
            st.write("- **Key Factors Influencing Deposits:**")
            st.write(f"  - 📞 **Call Duration:** Longer calls (>506s) significantly boost deposit rates.")
            st.write(f"  - 💰 **Balance:** Customers with a balance >$1,857 are more likely to deposit.")
            st.write(f"  - 📅 **Month:** December (90%), September (85%), and October (82%) have peak deposit rates.")
            st.write(f"  - 🔄 **Contacts:** Customers contacted 2-3 times have better engagement.")
            st.write(f"  - 📊 **Previous Interactions:** Past interactions positively impact deposits.")

            st.success("### 🎯 Key Suggestions:")
            st.write("💡 **Focus on upselling premium products**.")
            st.write("✉️ **Use email or SMS reminders for follow-ups**.")
            st.write("📈 **Offer personalized financial planning services**.")

            # Show toast message and balloons
            st.toast("✅ Customer analysis complete! Key insights identified to improve subscription likelihood.")
            st.balloons()


    except Exception as e:
        raise CustomException(e,sys)
