import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.title("Medical Expense Batch File Predictor")

## Load sample file
sample_file = pd.read_csv('sample_file.csv')

st.header("Overview")
st.markdown("This web app runs a ML algorithm on an input csv and returns the uploaded file with predicted medical expenses per client appended as the last feature/column. The input file should be formatted as the sample below.")
st.dataframe(sample_file)

## Load model
#filename = 'random_forest_regressor_model.sav'
model = pickle.load(open('rf-model.pkl', 'rb'))

st.header("Upload File")
st.markdown("Upload your CSV file for prediction below")
## Load Dataset

pred_df = st.file_uploader("upload file", type={"csv", "txt"})
if pred_df is not None:
    pred_df = pd.read_csv(pred_df)

## Main function

def make_predictions():
  
  categorical_features = ["sex","smoker","region"]

  pred_df['sex_male'] = np.where(pred_df['sex'] == "male", 1,0)
  pred_df['smoker_yes'] = np.where(pred_df['smoker'] == "yes", 1,0)

  pred_df['region_northwest'] = np.where(pred_df['region'] == "region_northwest", 1,0)
  pred_df['region_southeast'] = np.where(pred_df['region'] == "region_southeast", 1,0)
  pred_df['region_southwest'] = np.where(pred_df['region'] == "region_southwest", 1,0)

  pred_df.drop(categorical_features,axis=1,inplace=True)
   
  predictions = model.predict(pred_df)

  pred_df.loc[:,'predicted_expense'] = predictions

  return pred_df


if st.button('predict'):
    predicted_file = make_predictions()
    st.dataframe(predicted_file)

    total_expense = round(predicted_file['predicted_expense'].sum(),2)

    cohort_number = len(predicted_file)

    @st.cache
    def convert_df():
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
      return predicted_file.to_csv().encode('utf-8')

    final_data = convert_df()

    st.download_button(
      label="Download data as CSV",
      data=final_data,
      file_name='predictions.csv',
      mime='text/csv')

    st.markdown('')
   

    st.markdown('The predicted total expense for this cohort of {} customers is ${}'.format(cohort_number,total_expense))
    









