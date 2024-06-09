import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Check if input data is passed
if len(sys.argv) < 2:
    print(json.dumps({"error": "Usage: python app.py '<input_data_json>'"}))
    sys.exit(1)

# Parse input data from command-line argument
try:
    input_data = json.loads(sys.argv[1])
except json.JSONDecodeError as e:
    print(json.dumps({"error": "Error decoding JSON: " + str(e)}))
    sys.exit(1)

# Extract input features
age = input_data.get("age", 0)
sex = input_data.get("sex", 0)
cp = input_data.get("cp", 0)
trestbps = input_data.get("trestbps", 0)
chol = input_data.get("chol", 0)
fbs = input_data.get("fbs", 0)
restecg = input_data.get("restecg", 0)
thalach = input_data.get("thalach", 0)
exang = input_data.get("exang", 0)
oldpeak = input_data.get("oldpeak", 0)
slope = input_data.get("slope", 0)
ca = input_data.get("ca", 0)
thal = input_data.get("thal", 0)

# Load the dataset
heart_data = pd.read_csv('./db2.csv')

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Building a Predictive System
input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Standardize the input data
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_input_data = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = model.predict(std_input_data)

response = "The Person does not have a Heart Disease" if prediction[0] == 0 else "The Person has Heart Disease"

# Output the prediction and response as JSON
print(json.dumps({"prediction": int(prediction[0]), "response": response}))
