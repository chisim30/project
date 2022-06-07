import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import streamlit as st 
import streamlit.components.v1 as components

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




# title
original_title = '<p style="font-family:sans; color:black; font-size: 80px; text-align:center">Cancern</p>'
st.markdown(original_title, unsafe_allow_html=True)


#Get the data
cell_df = pd.read_csv('cell.csv')
# data cleaning
cell_df['BareNuc'].replace('?', np.nan, inplace =True)

cell_df['BareNuc'].replace(np.nan, '1', inplace =True)


#model building
X = cell_df[['Clump', 'UnifSize', 'UnifShape','MargAdh', 
      'SingEpiSize','BareNuc', 'BlandChrom', 'NormNucl']]

y = cell_df['Class']

# Split the dataset into 70% Training set and 30% Testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                       test_size= 0.3, random_state = 400)

name = st.text_input('What is your name?').capitalize()

#Get the feature input from the user
def get_user_input():

    Clump = st.number_input('Clump thickness:', min_value=1, max_value=10)
    UnifSize = st.number_input('Uniformity of cell size:', min_value=1, max_value=10)
    UnifShape = st.number_input('Unifority of cell shape:', min_value=1, max_value=10)
    MargAdh = st.number_input('Marginal adhesion:', min_value=1, max_value=10)
    SingEpiSize = st.number_input('Single epithelial cell size:', min_value=1, max_value=10)
    BareNuc = st.number_input('Bare nuclei:', min_value=1, max_value=10)
    BlandChrom = st.number_input('Bland chromatin:', min_value=1, max_value=10)
    NormNucl = st.number_input('Normal nucleoli:', min_value=1, max_value=10)
  
    user_data = {'Clump': Clump,
                'UnifSize': UnifSize,
                'UnifShape': UnifShape,
                'MargAdh': MargAdh,
                'SingEpiSize': SingEpiSize,
                'BareNuc': BareNuc,
                'BlandChrom': BlandChrom,
                'NormNucl': NormNucl
                 }
                 
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()

bt = st.button('Get Result')



if bt:
    parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

    KNN = KNeighborsClassifier()
    Grid_knn = GridSearchCV(KNN, parameters, cv=10)
    Grid_knn.fit(X_train, y_train)

    yhat_knn = Grid_knn.predict(user_input)
    
    if yhat_knn == 2:
        st.write(name, ', your tumor is benign.')
    else:
        st.write(name, ', your tumor is malignant.')






