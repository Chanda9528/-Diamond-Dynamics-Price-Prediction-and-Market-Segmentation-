# Diamond Dynamics - Price Prediction and Market Segmentation

This project predicts diamond prices and classifies diamonds into market segments using Machine Learning models. It also includes a Streamlit web application for real-time predictions.

## Problem Statement
Diamond price depends on quality parameters such as carat, cut, color, clarity and physical dimensions.  
The objective of this project:
1. Predict diamond price accurately.
2. Group diamonds into suitable market categories.

## Dataset Details
- Total Records: 53,940
- Features: carat, cut, color, clarity, depth, table, price, x, y, z

## Data Preprocessing
- Removed outliers using IQR method.
- Handled skewness using log transformation.
- Label encoding for categorical features.
- Feature scaling for clustering.

## Feature Engineering
New features created:
- Volume = x * y * z
- Price per Carat = price / carat
- Dimension Ratio = (x + y) / (2 * z)
- Carat Category (Light / Medium / Heavy)

## Machine Learning Models (Regression)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
- Artificial Neural Network (ANN)

Evaluation Metrics:
MAE, MSE, RMSE, R2 Score

Best model saved as .pkl file.

## Clustering (Market Segmentation)
- K-Means Clustering applied
- Optimal cluster selection using Elbow Method
- Price column removed before clustering
- PCA applied for visualization

Cluster labels assigned based on characteristics:
- Heavy High Price Diamonds
- Medium Price Balanced Diamonds
- Small Low Price Diamonds

## Streamlit Application
Features:
- Takes diamond input values from user
- Predicts diamond price in INR
- Predicts cluster category (market group)

## Technologies Used
Python, Pandas, NumPy, Scikit-Learn, TensorFlow/PyTorch, Matplotlib, Seaborn, Streamlit, Pickle

## Project Outputs
- Jupyter Notebook containing full analysis and model development
- Saved trained models (.pkl)
- Streamlit app for deployment


