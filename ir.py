
import pandas as pd
import streamlit as st
import pickle
import numpy as np




with open('svm_binary.pkl','rb')as file:
    svm_binary=pickle.load(file)

with open('svm_multi.pkl','rb')as file:
    svm_multi=pickle.load(file)
with open('logistics_binary.pkl','rb')as file:
    logistics_binary=pickle.load(file)
with open('logistic_multi_ovr_model.pkl','rb')as file:
    logistics_ovr=pickle.load(file)
with open('logistic_multi_softmax_model.pkl','rb')as file:
    logistics_multinomial=pickle.load(file)


     
st.title("Model Selection and Prediction App")

model_option = st.selectbox(
    "Choose a model for prediction",
    ["SVC Binary", "SVC Multi", "Logistic Regression Binary", "Logistic Regression Multi (OVR)", "Logistic Regression Multi (Softmax)"]
)

sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

input=np.array([[sepal_length,sepal_width ,petal_length,petal_width]])
def get_prediction(model, input):
    
    prediction = model.predict(input)
    return prediction

if st.button('predict'):
    if model_option == "SVC Binary":
        prediction = get_prediction(svm_binary, input)
    elif model_option == "SVC Multi":
        prediction = get_prediction(svm_multi, input)
    elif model_option == "Logistic Regression Binary":
        prediction = get_prediction(logistics_binary, input)
    elif model_option == "Logistic Regression Multi (OVR)":
        prediction = get_prediction(logistics_ovr, input)
    elif model_option == "Logistic Regression Multi (Softmax)":
        prediction = get_prediction(logistics_multinomial, input)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    predicted_class_name = class_names[prediction[0]]  
    st.success(f'{predicted_class_name}')
   



if __name__ == '__main__':
    st.write("Result accrding to model selection!")