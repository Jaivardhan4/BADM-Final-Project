# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ay_1FBadIFYcGCOklSkzcO-02fvboC-T
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import gradio as gr
import joblib

# Define file paths
data_path = '/content/breast-cancer-wisconsin-data.csv'
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'

# Load the dataset
data = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Encode the target variable
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Drop unnecessary columns if they exist
if 'Unnamed: 32' in data.columns:
    data = data.drop(columns=['Unnamed: 32'])
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Separate features and target variable
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Check if model and scaler files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(xgb_model, model_path)
else:
    # Load the scaler and model
    scaler = joblib.load(scaler_path)
    xgb_model = joblib.load(model_path)

# Function for making predictions
def predict(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean):
    input_data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]]
    input_data = scaler.transform(input_data)
    prediction = xgb_model.predict(input_data)
    return 'Malignant' if prediction[0] == 1 else 'Benign'

# Create Gradio interface
inputs = [
    gr.Number(label="Radius Mean"),
    gr.Number(label="Texture Mean"),
    gr.Number(label="Perimeter Mean"),
    gr.Number(label="Area Mean"),
    gr.Number(label="Smoothness Mean"),
    gr.Number(label="Compactness Mean"),
    gr.Number(label="Concavity Mean"),
    gr.Number(label="Concave Points Mean"),
    gr.Number(label="Symmetry Mean"),
    gr.Number(label="Fractal Dimension Mean")
]

outputs = gr.Textbox(label="Prediction")

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Breast Cancer Prediction", description="Predict whether a breast tumor is malignant or benign based on its characteristics.").launch()

!pip install xgboost
!pip install scikit-learn
!pip install joblib
!pip install matplotlib
!pip install seaborn

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
data_path = '/content/breast-cancer-wisconsin-data.csv'
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'

# Load the dataset
data = pd.read_csv(data_path)

# Encode the target variable
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Separate features and target
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize and perform Grid Search
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, model_path)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy, Precision, Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Plot Confusion Matrix with clear labels and annotations
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Precision and Recall
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})

plt.figure(figsize=(10, 7))
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics')
plt.show()

##v1

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
data_path = '/content/breast-cancer-wisconsin-data.csv'
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'

# Load the dataset
data = pd.read_csv(data_path)

# Encode the target variable
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Check the columns in the dataset
print(data.columns)

# Separate features and target
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Check if model and scaler files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    # Split the data into training (90%) and testing (10%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    # Initialize and train the XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(xgb_model, model_path)
else:
    # Load the scaler and model
    scaler = joblib.load(scaler_path)
    xgb_model = joblib.load(model_path)

    # Standardize the features
    X_scaled = scaler.transform(X)

    # Split the data into training (90%) and testing (10%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy, Precision, Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Precision and Recall
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})

plt.figure(figsize=(10, 7))
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics')
plt.show()