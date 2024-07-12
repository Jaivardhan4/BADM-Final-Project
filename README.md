## Project Overview

Breast cancer is a disease in which abnormal breast cells grow out of control and form tumors. If left unchecked, the tumors can spread throughout the body and become fatal. This project aims to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) using machine learning techniques.

### Objective

The main objective is to develop a classification model to predict the target variable, indicating whether a tumor is malignant (M) or benign (B).

## Dataset

The dataset used in this project contains several features related to breast tumors. Below are the key features:

1. **Radius Mean**: Mean of distances from the center to points on the perimeter.
2. **Texture Mean**: Standard deviation of gray-scale values.
3. **Perimeter Mean**: Mean size of the core tumor.
4. **Area Mean**: Mean area of the core tumor.
5. **Smoothness Mean**: Mean of local variation in radius lengths.
6. **Compactness Mean**: Mean of perimeter^2 / area - 1.0.
7. **Concavity Mean**: Mean severity of concave portions of the contour.
8. **Concave Points Mean**: Mean number of concave portions of the contour.
9. **Symmetry Mean**: Mean symmetry of tumor.
10. **Fractal Dimension Mean**: Mean "coastline approximation" - 1.

## Model Selection and Implementation

### Model Used: XGBoost

We selected the XGBoost (Extreme Gradient Boosting) classifier for this project due to its superior performance and efficiency. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. 

#### Advantages of XGBoost:

1. **Performance**: XGBoost is known for its high performance in terms of accuracy and speed.
2. **Regularization**: It includes L1 and L2 regularization to prevent overfitting.
3. **Handling Missing Data**: XGBoost can handle missing data internally, which is beneficial for real-world datasets.
4. **Parallel Processing**: It supports parallel processing which speeds up the training process.
5. **Tree Pruning**: The model prunes trees to prevent overfitting, resulting in a more generalized model.

### Implementation Steps:

1. **Data Preprocessing**:
   - Loading the dataset.
   - Encoding the target variable.
   - Standardizing the features.

2. **Model Training**:
   - Splitting the data into training (90%) and testing (10%) sets.
   - Performing hyperparameter tuning using Grid Search with Cross-Validation to find the best model.

3. **Evaluation**:
   - Generating classification reports and confusion matrices.
   - Calculating accuracy, precision, and recall.

## Results

### Classification Report:

     precision       recall    f1-score   

       Benign          0.97      0.97              
       Malignant       0.94      0.94           

       accuracy        0.96

### Confusion Matrix:
             Benign    Malignant
    Benign     39         1
    Malignant   1         16

### Accuracy: 
96.49 %

### Precision: 
94.12 %

### Recall: 
94.12 %

## Critical Analysis

The XGBoost model achieved an accuracy of 96.49%, which is considered excellent for a binary classification problem. Here are some key observations and critical evaluations:

1. **High Accuracy**: The model correctly predicted 96.49% of the test samples, indicating its high reliability.
2. **Precision and Recall**: Both precision and recall scores are high (0.9412), indicating that the model is effective in predicting both malignant and benign tumors. High precision reduces the chances of false positives, and high recall reduces the chances of false negatives.
3. **Balanced Performance**: The confusion matrix shows a balanced performance across both classes with minimal misclassifications (1 false positive and 1 false negative).
4. **Model Efficiency**: XGBoost's ability to handle large datasets efficiently and provide high performance with lower computation time played a crucial role in achieving these results.

### Conclusion

The use of XGBoost provided a significant advantage in terms of accuracy and efficiency. The model's high performance in predicting breast cancer makes it a valuable tool for medical professionals in early diagnosis and treatment planning. The balance between precision and recall ensures that the model is reliable and minimizes the risk of both false positives and false negatives.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-link>

2. **Navigate to the Project Directory**:

   ```bash
    cd breast-cancer-classification

3. **Install the Required Packages**:

   ```bash
    pip install -r requirements.txt
    
4. **Run the Script**: 

   ```bash
    python main.py
