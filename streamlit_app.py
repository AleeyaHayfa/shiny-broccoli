import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

with st.expander('Data'):
    # Load the dataset
    df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
    st.write('Dataset:')
    st.dataframe(df)

    # Separate features (X) and target (Y)
    st.write('*X (Features)*')
    x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    st.dataframe(x_raw)

    st.write('*Y (Target)*')
    y_raw = df['AtRiskOfOrExperiencingHomelessnessFlag']
    st.write(y_raw)

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='PeopleonApplication', y='FamilyType', color='AtRiskOfOrExperiencingHomelessnessFlag')

# Data prepration (user input)
with st.sidebar:
  st.header('Input features')
  Family = st.selectbox('Family Type', (
    'Single Person', 
    'Single Parent, 1 Child', 
    'Single Parent, 2 Children', 
    'Single Parent, >2 Children', 
    'Couple Only', 
    'Couple, 1 Child', 
    'Couple, 2 Children', 
    'Couple, >2 Children', 
    'Single Person Over 55', 
    'Couple Only Over 55', 
    'Other'
))
  DisabilityFlag = st.selectbox('Disability', ('Yes', 'No'))
  TotalPeople = st.slider('Total People on application', 1, 12, 2)
  TotalMonths = st.slider('Total months you have been register', 0, 239, 23)
  # 1.0 mininum, 12.0 maximum, 1.681 average (same as other)

# user input into dataframe
# Create a DataFrame for the input features
  data = {'FamilyType': Family,
          'MonthsonHousingRegister': TotalMonths,
          'DisabilityApplicationFlag': DisabilityFlag,
          'PeopleonApplication': TotalPeople,
         }
  input_df = pd.DataFrame(data, index=[0])
  input_details = pd.concat([input_df, x_raw], axis=0) #combine input features with penguin features
  # input details is user input

with st.expander('Input features'):
  st.write('*Input User*')
  input_df
  st.write('*Combined Housing data*') #combine original dgn user input
  input_details

# Encode X 
encode = ['FamilyType', 'DisabilityApplicationFlag']
df_house = pd.get_dummies(input_details, prefix=encode)

x = df_house[1:] #ignore first row but use everything after first row
input_row = df_house[:1] #use only first row

# Encode y
target_mapper = {'Yes': 1,
                 'No': 0
                 }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)
#y #ni yang encoded
#y_raw # original


with st.expander('Data preparation'):
  st.write('*Encoded X (input housing)*')
  input_row
  st.write('*Encoded y*')
  y
  # y_raw #compare dgn original nk tgk betul ke tak

# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(x, y) #use fit function to train it

## Apply model to make predictions
prediction = clf.predict(input_row) #predict value, input_row is input features
prediction_proba = clf.predict_proba(input_row) #do probability

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Yes', 'No']
df_prediction_proba.rename(columns={0: 'No',
                                 1: 'Yes'
                                 })
#df_prediction_proba #to see if it works

# Display predicted species
st.subheader('Predicted Homelessness')
st.dataframe(df_prediction_proba,
             column_config={
               'Yes': st.column_config.ProgressColumn( #progressColumn tu ui macam slider
                 'Yes',
                 format='%.2f',
                 width='medium',
                 #probability (0-1)
                 min_value=0,
                 max_value=1
               ),
               'No': st.column_config.ProgressColumn(
                 'No',
                 format='%.2f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


housing_homeless = np.array(['Yes', 'No'])
st.success(str(housing_homeless[prediction][0]))
