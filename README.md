# Diabetes-Prediction-using-machine-learning

PROBLEM STATEMENT
 Diabetes is a serious health issue where the blood sugar levels hence the early diagnosis
is highly needed.  Elevated blood sugar levels cause various health issues and difficulties. 
SOLUTION TO THE PROBLEM
 The project is designed to predict Diabetes in the early stage.  This project helps to identify if the person is diabetic or not so the necessary
medication can be taken.  The proposed system uses the Support Vector Machine Classification for predicting
the person is diabetic or not. 
METHODOLOGY
 Installation of packages - 1. Include the following libraries: numpy for numerical
operations and pandas for data manipulation. 1. sklearn.preprocessing is used to standardise data. 2. sklearn.model_selection is used to separate data. 3. For the Support Vector Machine model, use sklearn.svm. 4. sklearn.metrics for assessment
5. pickle for model saving and loading
6. streamlit for web application development
 Collection of Data Sets - the diabetes dataset is loaded into a DataFrame, pd.read_csv() is used for loading the dataset
 Standardization of data - the data is standardized using the head(), shape(), description(), and value_counts() functions. 
Training Test Split - The standardized data is then split into train and test data.
 Model Training - The SVM classifier is employed to train the model. 80% of thedata is used in training
 Model evaluation - The model accuracy is evaluated using test data. 20% of the datais used to test the model
 Model Saving - The model is saved so that it need not to be retrained.  
Model Deployment - The code creates a user friendly interface where user can enter their medical data. 
 Prediction - When the diabetes test result button is pressed, the code retrieves user
inputs, standardizes it and uses the saved model to predict the likelihood of diabetes

WORKING:
The PIMA Indian Diabetes Dataset, a widely used benchmark, is utilized in this project. Relevant features are extracted from the dataset, representing factors like age, glucose levels,BMI and so on. Data is then standardized using a scaling technique to ensure consistent measurements across features. SVM aims to find the optimal hyperplane (decision boundary) that effectively separates thetwo classes. This hyperplane maximizes the margin—the distance between it and the closest points from each class. A wider margin generally leads to better generalization and a more robust model. The SVM algorithm iteratively adjusts the hyperplane's position, aiming to minimize misclassifications in the training data. It focuses on the "support vectors"—the closest points to the hyperplane—as they are crucial for defining the decision boundary
A linear kernel SVM is employed due to its effectiveness in high-dimensional spaces and interpretability. The model is trained on the prepared data, aiming to identify patterns that differentiate diabetic and non-diabetic individuals. Hyperparameter optimization is performed to enhance the model's generalization. The trained model's accuracy is rigorously assessed on both the training and separate testingdatasets. Metrics like accuracy, precision, and recall are used to evaluate the model's ability to correctly predict diabetic and non-diabetic cases. Cross-validation techniques are utilized to further validate the model's generalizability.

The trained SVM model is saved for future use and potential integration into other applications. Additionally, a user-friendly web interface is built using Streamlit. This interface allows individuals to input their health data and receive the model's prediction on their potential diabetes risk, empowering them to make informed decisions regarding their health. When a new individual's health data is presented, SVM maps their point onto the same high- dimensional space. It then determines on which side of the hyperplane the point falls, classifying them as diabetic or non-diabetic accoringly.
