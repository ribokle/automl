import streamlit as st
import pandas as pd
import os
### import profiling capability
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
### import ML codes
from pycaret.regression import compare_models as compare_models, setup as setup, pull as pull, save_model as save_model
from pycaret.classification import compare_models as compare_modelsc, setup as setupc, pull as pullc, save_model as save_modelc


st.set_page_config(page_title="AutoML")
with st.sidebar:
    st.image("https://unilever.sharepoint.com/_api/siteiconmanager/getsitelogo?type=%271%27")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Regression","Classification","Download"])
    st.info("This app enables you to build an automated ML pipeline using Streamlit, ydata profiling and Pycaret")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice== "Upload":
    st.title("Upload your data for modeling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)



if choice== "Profiling":
    st.title("Automated Exploratory Data Analysis") 
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice== "Regression":
    st.title("Automated ML")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train model"):
        setup(df,target=target)
        setup_df = pull()
        st.info("This is the experimental settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")

if choice== "Classification":
    st.title("Automated ML")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train model"):
        setupc(df,target=target)
        setup_df = pullc()
        st.info("This is the experimental settings")
        st.dataframe(setup_df)
        best_model = compare_modelsc()
        compare_df = pullc()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_modelc(best_model,"best_model")
    



if choice== "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model",f,"trained_model.pkl")

# invoke python from C:\Users\Gourav.Kumar\reg\automl