import streamlit as st 
import requests

# add title
st.title('Credit Default Prediction')
st.subheader('Plese enter the input below then click the Predict button.')

# create the input form
with st.form(key='credit_form'):

    # create dropdown input
    person_age = st.number_input(label='1.\tEnter person age:')
    person_income = st.number_input(label='2.\tEnter person income:')
    person_home_ownership = st.selectbox(
        label='3. \tEnter the home ownership status:',
        options = (
            "MORTGAGE",
            "RENT",
            "OWN",
            "OTHER"
        )
    )
    person_emp_length = st.number_input(label='4.\tEnter person employment length:')
    loan_intent = st.selectbox(
        label='5. \tEnter the intention of loan:',
        options = (
            'EDUCATION', 
            'PERSONAL', 
            'MEDICAL', 
            'VENTURE', 
            'HOMEIMPROVEMENT',
            'DEBTCONSOLIDATION'
        )
    )
    loan_grade = st.selectbox(
        label="6. \tEnter the loan grade:",
        options = (
            'A', 
            'B', 
            'C', 
            'D', 
            'E', 
            'F', 
            'G'
        )
    )
    loan_amnt = st.number_input(label='7.\tEnter loan amount:')
    loan_int_rate = st.number_input(label='8.\tEnter loan interest rate:')
    loan_percent_income = st.number_input(label='9.\tEnter loan percent income:')
    cb_person_default_on_file = st.selectbox(
        label='10. \tDoes the person has had default loan historically?',
        options=(
            "N",
            "Y"
        )
    )
    cb_person_cred_hist_length = st.number_input(label='11.\tEnter the number of years that the person has had a recorded credit history:')

    # create button to submit the form
    submitted = st.form_submit_button("Predict!")

    # condition when form has already submitted
    if submitted:
        # create dict of all data from the input form
        raw_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }

        # create a loading animation
        with st.spinner('Sending input data to prediction server. . .'):
            res = requests.post("http://localhost:8080/predict", json=raw_data).json()
        
        # show the response from api
        if res['label'] != "Default":
            st.write(f'Probability of Default: {res['probability']} or {round(res['probability']*100, 2)}%')
            st.success("Predicted Credit Risk: Non-Default Loan")
        else:
            st.write(f'Probability of Default: {res['probability']} or {round(res['probability']*100, 2)}%')
            st.error("Predicted Credit Risk: Default Loan")