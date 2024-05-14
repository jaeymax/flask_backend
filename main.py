from flask import Flask, request, jsonify
import json
import pandas as pd
# Load the dataset
data = pd.read_csv('heart.csv')
# Display the first few rows of the dataset
#print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns
# Set the aesthetics for the plots
#sns.set(style="whitegrid")
# Histograms for all numeric columns
#data.hist(figsize=(15, 12))
#plt.show()

#print(data.describe())

#print("Correlation with the target variable")
#print(data.corr()['target'].sort_values())

#plt.figure(figsize=(8, 6))
#sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt="0.1f", linewidths= 0.5)
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Preparing data for modeling
X = data.drop('target', axis=1)

feature_names = data.drop('target', axis=1).columns


y = data['target']
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predicting on the test set
y_pred = model.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score
# Calculating performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#print("Accuracy:", accuracy)
#print("Confusion Matrix:\n", conf_matrix)
#print("Classification Report:\n", classification_report(y_test, y_pred))

#plt.figure(figsize=(5, 5))
#sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, cmap='Blues', square=True, cbar=False)
#plt.xlabel('Predicted label')
#plt.ylabel('True label')
#plt.title('Confusion Matrix')
#plt.xticks([0.5, 1.5], ['No Heart Disease', 'Heart Disease'])
#plt.yticks([0.5, 1.5], ['No Heart Disease', 'Heart Disease'], rotation=0)
#plt.show()

performance_data = {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1-Score': [f1]}
performance_df = pd.DataFrame(performance_data)
# Plotting the data
#plt.figure(figsize=(6, 4))
#sns.barplot(data=performance_df)
#plt.title('Model Performance Metrics')
#plt.ylabel('Score')
#plt.ylim(0, 1)  
#plt.xticks(range(len(performance_data)), list(performance_data.keys()))
#plt.show()




# Function to capture user input from terminal
def get_user_input():
    user_input = {}
    for feature in feature_names:
        value = input(f"Enter value for {feature}: ")
        user_input[feature] = value
    return user_input

# Function to preprocess user input
def preprocess_input(user_input):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    # Perform any necessary preprocessing (e.g., encoding categorical variables, scaling numerical features)
    # Note: Ensure preprocessing steps match those used during training
    return input_df


def make_predictions(input_df):
    # Make predictions using the trained model
    predictions = model.predict(input_df)
    return predictions

# Function to display results
def display_results(predictions):
    if predictions[0] == 0:
        print("Prediction: No Heart Disease")
    else:
        print("Prediction: Heart Disease")









app = Flask(__name__)


@app.route('/')
def home():
    response = {'message':'Hello World'}
    
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from request
    user_input = request.json
    
    # Preprocess input data
    input_data = pd.DataFrame(user_input, index=[0])
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Prepare response
    if predictions[0] == 0:
        prediction_result = "No Heart Disease"
    else:
        prediction_result = "Heart Disease"
    
    response = {'prediction': prediction_result}
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

