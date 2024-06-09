import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Check if input data is passed
if len(sys.argv) < 2:
    print("Usage: python app.py <input_data>")
    sys.exit(1)

# Parse input data from command-line argument
input_data = json.loads(sys.argv[1])

# Extract input features
pregnancies = input_data.get("Pregnancies", 0)
glucose = input_data.get("Glucose", 0)
blood_pressure = input_data.get("BloodPressure", 0)
skin_thickness = input_data.get("SkinThickness", 0)
insulin = input_data.get("Insulin", 0)
bmi = input_data.get("BMI", 0)
diabetes_percentage = input_data.get("DiabetesPrecentage", 0)
age = input_data.get("Age", 0)

# Load dataset
diabetes_dataset = pd.read_csv('./db.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']










# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Make predictions on new data
input_data_array = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_percentage, age]])
std_data = scaler.transform(input_data_array)
prediction = classifier.predict(std_data)

# Output prediction
if prediction[0] == 0:
    result = 'The person is not diabetic'
else:
    result = 'The person is diabetic'

# Print result as JSON array
response = json.dumps([prediction[0], result])
print(response)
