import streamlit as st
import pandas as pd
import pickle

st.header('Cars 24 price predictor')

col1, col2, col3, col4 = st.columns(4)

with col1:
    ft= st.selectbox(
        "Select Fuel type",
        ["Petrol", "Diesel", "CNG","Electric"])
with col2:
    tt=st.radio('select transmission',["Manual","Automatic"])
    
with col3:
    seats=st.radio('select number of seats',[1,2,3,4])
    
engine=st.slider('set engine power',500,5000,100)

encd = { "fuel_type": { "Petrol" :1,
"Diesel" :2,
"CNG" :3,
"Electric" :4 }, "tt": { "Manual" :1,
"Automatic" :2 }
      }
        
def model_pred(fuel_encoded,tt_encoded,seats,engine):
    with open("car_pred","rb") as file:
        reg_model=pickle.load(file) 
        input_features=[[2012,1,120000,fuel_encoded,tt_encoded,19.7,engine,46.3,seats]]
        return reg_model.predict(input_features)


if st.button("Predict"):
    fuel_encoded=encd["fuel_type"][ft]
    tt_encoded=encd["tt"][tt]
    price=model_pred(fuel_encoded,tt_encoded,seats,engine)
    st.text("Predicted price is"+ str(price))



