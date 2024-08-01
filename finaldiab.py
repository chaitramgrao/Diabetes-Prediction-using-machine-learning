#!/usr/bin/env python
# coding: utf-8

# In[53]:

#importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the dataset
data_frame = pd.read_csv('diabetes.csv')

data_frame.head(10)

data_frame.head()

#number of rows and columns in the dataset data_frame
data_frame.shape

#getting the statistical measure of the data
data_frame.describe()

data_frame['Outcome'].value_counts()

"""0---> Non Diabetic



1----> Diabetic
"""


# In[56]:

#grouping the data
data_frame.groupby('Outcome').mean()

X =data_frame.drop(columns='Outcome', axis=1)
Y= data_frame['Outcome']

print(X)

print(Y)

scaler =StandardScaler()

standardize = scaler.fit(X)

standardized_data = scaler.transform(X)
print(standardized_data)

X= standardized_data
Y= data_frame['Outcome']

print(X)

print(Y) ##this serves as the label

X_train , X_test, Y_train, Y_test,= train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)

print(X.shape, X_train.shape, Y.shape, Y_train.shape)


# In[57]:


"""Training our model



"""
#model training
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# accuracy test
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(training_data_accuracy)

##accuracy test for test data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy= accuracy_score(X_test_prediction, Y_test)

print('The accuracy for test data is :', testing_data_accuracy)

input_data =(10,168,74,0,0,38,0.537,34)
input_data_np= np.asarray(input_data)

#reshaping the data because we only want data for one instance
final_input_data= input_data_np.reshape(1,-1)

#standardizing the input data since our model got trained on standardized data and not general raw data
std_data= scaler.transform(final_input_data)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0]==0):    ##prediction is a list . svm returns a list with only one value(0 or 1)
  print("The Person does not have diabetes")
else:
  print("The person is  diabetic")

#saving the trained model
import pickle

filename = r'C:\Users\CHAITRA M G\Desktop\Chaitra\trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

#loading the saved model
loading_model=pickle.load(open('trained_model.sav', 'rb'))


import numpy as np
import pickle 
import streamlit as st


# In[59]:


loaded_model = pickle.load((open(r'C:\Users\CHAITRA M G\Desktop\Chaitra\trained_model.sav','rb')))


# In[60]:


def diabetes_prediction(input_data):
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping the data because we only want data for one instance


    #standardizing the input data since our model got trained on standardized data and not general raw data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):    ##prediction is a list . svm returns a list with only one value(0 or 1)
      return "The Person does not have diabetes"
    else:
      return "The person is  diabetic"


# In[ ]:





# In[66]:


def main():
    
    
    #giving a title 
    st.title('Diabetes Predictor')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure= st.text_input('Blood Pressure value')
    SkinThickness= st.text_input('Skin Thickness value')
    Insulin= st.text_input('Insulin Level')
    BMI= st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age= st.text_input('Age of the Person')
    
    #code for prediction
    
    diagnosis =''
    
    #creating an input button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()


# In[ ]:




