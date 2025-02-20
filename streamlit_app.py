import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title('🤖 ThinkTankers ML App')

st.write('This app builds a machine learning model')

# Load Data
with st.expander('Data'):
    df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
    st.write('Dataset:')
    st.dataframe(df)

    x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    y_raw = df['AtRiskOfOrExperiencingHomelessnessFlag']

    st.write('*X (Features)*')
    st.dataframe(x_raw)

    st.write('*Y (Target)*')
    st.write(y_raw)

# Data Visualization
with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='PeopleonApplication', y='FamilyType', color='AtRiskOfOrExperiencingHomelessnessFlag')

# User Input
with st.sidebar:
    st.header('Enter Client Details')
    Family = st.selectbox('Family Type', (
        'Single Person', 'Single Parent, 1 Child', 'Single Parent, 2 Children', 
        'Single Parent, >2 Children', 'Couple Only', 'Couple, 1 Child', 
        'Couple, 2 Children', 'Couple, >2 Children', 'Single Person Over 55', 
        'Couple Only Over 55', 'Other'
    ))
    DisabilityFlag = st.selectbox('Disability', ('Yes', 'No'))
    TotalPeople = st.slider('Total People on application', 1, 12, 2)
    TotalMonths = st.slider('Total months you have been registered', 0, 239, 23)

    data = {'FamilyType': Family,
            'MonthsonHousingRegister': TotalMonths,
            'DisabilityApplicationFlag': DisabilityFlag,
            'PeopleonApplication': TotalPeople}
    input_df = pd.DataFrame(data, index=[0])
    input_details = pd.concat([input_df, x_raw], axis=0)

# Display User Input
with st.expander('Input features'):
    st.write('*Input User*')
    st.write(input_df)
    st.write('*Combined Housing data*')
    st.write(input_details)

# Encode X
encode = ['FamilyType', 'DisabilityApplicationFlag']
df_house = pd.get_dummies(input_details, prefix=encode)
x = df_house[1:]
input_row = df_house[:1]

# Encode y
target_mapper = {'Yes': 1, 'No': 0}
y = y_raw.apply(lambda val: target_mapper[val])

with st.expander('Data preparation'):
    st.write('*Encoded X (input housing)*')
    st.write(input_row)
    st.write('*Encoded y*')
    st.write(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on Test Data
y_pred = clf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display Model Performance
with st.expander('Model Performance'):
    st.write(f'**Accuracy:** {accuracy:.2f}')
    st.write(f'**Precision:** {precision:.2f}')
    st.write(f'**Recall:** {recall:.2f}')
    st.write(f'**F1 Score:** {f1:.2f}')

# Predict User Input
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display Prediction
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
st.subheader('Predicted Homelessness')
st.dataframe(df_prediction_proba, column_config={
    'Yes': st.column_config.ProgressColumn('Yes', format='%.2f', width='medium', min_value=0, max_value=1),
    'No': st.column_config.ProgressColumn('No', format='%.2f', width='medium', min_value=0, max_value=1)
}, hide_index=True)

housing_homeless = np.array(['No', 'Yes'])
st.success(str(housing_homeless[prediction][0]))
