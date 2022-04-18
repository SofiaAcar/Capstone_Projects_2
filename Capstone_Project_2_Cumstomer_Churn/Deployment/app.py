import streamlit as st
import pandas as pd
import pickle
import joblib
st.title("Churn PAGE")
randomf_model = open("RandomForestClassifier.pkl","rb")
randomf_columns = open("my_columns.pkl","rb")
randomf_scaler = open('my_scaler_knn.pkl',"rb")
randomf_model = joblib.load(randomf_model)
randomf_columns = joblib.load(randomf_columns)
randomf_scaler = joblib.load(randomf_scaler)




st.sidebar.title('Configure Your Customer')
satisfaction_level = st.sidebar.number_input("satisfaction_level", min_value =0.09, max_value = 1.0, value=0.53)
last_evaluation = st.sidebar.number_input("last_evaluation", min_value =0.36, max_value = 1.0, value=0.42)
number_project = st.sidebar.number_input("number_project", min_value =2, max_value = 7, value=4)
average_montly_hours = st.sidebar.number_input("average_montly_hours", min_value =96, max_value = 310, value=120)
time_spend_company = st.sidebar.number_input("time_spend_company", min_value =2, max_value = 10, value=3)
Work_accident = st.sidebar.selectbox("Work_accident", [0,1])
promotion_last_5years = st.sidebar.selectbox("promotion_last_5years", [0,1])
Departments = st.sidebar.selectbox("Departments", ['RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng',  'sales', 'support', 'technical', 'IT'])
salary = st.sidebar.selectbox("salary", ['low','medium','high'])

data = {}
data["satisfaction_level"]=satisfaction_level
data["last_evaluation"]=last_evaluation
data["number_project"]=number_project
data["average_montly_hours"]=average_montly_hours
data["time_spend_company"]=time_spend_company
data["Work_accident"]=Work_accident
data["promotion_last_5years"]=promotion_last_5years
data["Departments"]=Departments
data["salary"]=salary
st.write(randomf_columns)
predict = st.sidebar.button("P R E D I C T")

if predict:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df).reindex(columns=randomf_columns, fill_value=0)
    df = randomf_scaler.transform(df)
    result = randomf_model.predict(df)
    st.table(pd.DataFrame([data]))
    st.write(result)
    if result == 0:
        st.markdown("<h2 style='text-align: center; color: green;'>He/she will stay.</h2>", unsafe_allow_html=True)
    elif result == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>He/she will not stay.</h2>", unsafe_allow_html=True)


