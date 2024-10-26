import streamlit as st
import pickle
import torch
import numpy as np
from cxchurn import CxChurn
from torch import nn


df = pickle.load(open("df1.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

model = CxChurn()
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval() 


# st.title("Customer Churn Prediction Model Using ANN and PyTorch")
title = "Customer Churn Prediction Model Using ANN and 💳👨🏻‍💼💁🏻‍♀️ PyTorch 👨🏻‍💻🕵️‍♂️💳"
st.markdown(f'<div style="text-align: center; color: #ba4120; font-size: 28px; font-weight: bold; margin-top: 2px; background-color: #20baba; font-family: Comic Sans MS, monospace; font-weight: bold;">{title}</div>',
                        unsafe_allow_html=True)
st.write("")
# st.table(df.head())

# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary

col1, col2, col3 = st.columns(3)

with col1: 
    CreditScore = int(st.number_input("Enter the credit score:"))
    input_geography = st.selectbox("Enter the Country: ", ["Germany", "Spain", "France"])
    Gender = st.selectbox("Enter the Gender: ", ["Male", "Female"])

with col2:
    Age = int(st.number_input("Enter the Age: "))
    Tenure = int(st.number_input("Enter the Tenure Period: "))
    Balance = float(st.number_input("Enter the Bank Balance: "))

with col3:
    NumOfProducts = int(st.number_input("Enter the number of products: "))
    HasCrCard = st.selectbox("if cx has credit card or not: ", ["Yes", "No"])
    IsActiveMember =  st.selectbox("Cx is active member or not: ", ["Yes", "No"])

EstimatedSalary = float(st.number_input("Enter the Cx Salary: "))


input_CrCredit = HasCrCard
credit_value = None
if input_CrCredit == "Yes":
    credit_value = 1
elif input_CrCredit == "No":
    credit_value = 0


input_member = IsActiveMember
active_member = None
if input_member == "Yes":
    active_member = 1
elif input_member == "No":
    active_member = 0


input_gender = Gender
# Initialize a variable for gender
gender_value = None
# Use if-else to set gender_value based on input
if input_gender == "Female":
    gender_value = 0
elif input_gender == "Male":
    gender_value = 1


geography_germany = 0
geography_spain = 0

# Use if-else to determine the appropriate values
if input_geography == "Germany":
    geography_germany = 1
elif input_geography == "Spain":
    geography_spain = 1
elif input_geography == "France":
    geography_germany = 0  # France is not represented, so keep it 0
    geography_spain = 0     # Likewise, Spain should be 0

input_data = np.array([CreditScore, Age, Tenure, Balance, NumOfProducts, credit_value, active_member, EstimatedSalary, geography_germany, geography_spain, gender_value])

if st.button("Predict"):
    # st.write(input_data)
    new_data = input_data.reshape(1, -1)
    new_input = scaler.transform(new_data)

    input_data = torch.tensor(new_input, dtype=torch.float32)

    with torch.inference_mode():
        output = model(input_data)
        probabilities = torch.sigmoid(output)
        prediction = (probabilities >= 0.5).float()

        # Create a custom message based on the probability
        if probabilities >= 0.75:
            justification = "The model is highly confident that the customer will churn."

        elif probabilities >= 0.55:
            justification = "The model suggests the customer is likely to churn, but there is some uncertainty."
        elif probabilities >= 0.35:
            justification = "The model indicates uncertainty: the customer might churn or stay."
        else:
            justification = "The model is confident that the customer is unlikely to churn."

        # Print the results

        st.markdown(f'<div style="text-align: center; color: #cfcf02; font-size: 20px; margin-top: 2px; background-color: #706e6d; font-family: Serif, monospace; font-weight: bold;">💳 {justification} 💳</div>',
                        unsafe_allow_html=True)
        st.write("")
        st.text(f"Predicted Churn Class: {'Churn (Left)' if prediction == 1 else 'Not Churn (Not Left)'}")
        st.text(f"Churn Probability: {probabilities.item():.4f}")
        # st.write(f"Justification: {justification}")

        