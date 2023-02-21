# Heart-Failure-Prediction-Modelling

This is a repository for the heart failure prediction model that I created as part of my data science coursework in university. The program uses machine learning algorithms to predict whether a patient is likely to experience heart failure based on several medical features.

## Project Overview

The goal of this project was to create a predictive model for heart failure based on medical data. I used a dataset from Kaggle and applied various machine learning algorithms to build a model that could accurately predict whether a patient was likely to experience heart failure based on several medical features.

I chose this project because it allowed me to apply several key data science skills, including data cleaning and preprocessing, feature engineering, model selection and evaluation, and visualization.

## Technical Details

This project is written in Python and uses several libraries and frameworks, including:

pandas and numpy for data manipulation and preprocessing.
scikit-learn for machine learning algorithms and model evaluation.
matplotlib and seaborn for data visualization.
The program consists of the following files:

predict_heart_failure.py: the main script for running the heart failure prediction program.

Library.py: a Python module containing utility functions used in the heart failure prediction program.

Ann_model: a trained artificial neural network model for heart failure prediction.

Knn_model.joblib: a trained k-nearest neighbors model for heart failure prediction.

Logreg_model.joblib: a trained logistic regression model for heart failure prediction.

Xgb_model.joblib: a trained XGBoost model for heart failure prediction.

Ann_thresholds.csv: a file containing threshold values used in the artificial neural network model.

Knn_thresholds.csv: a file containing threshold values used in the k-nearest neighbors model.

Logreg_thresholds.csv: a file containing threshold values used in the logistic regression model.

Xgb_thresholds.csv: a file containing threshold values used in the XGBoost model.

Heart.csv: the dataset used for heart failure prediction.
Results


After experimenting with several different machine learning algorithms and model architectures, I was able to achieve an accuracy of 0.85 in predicting heart failure using the XGBoost model.

I also performed feature importance analysis to determine which features were most strongly correlated with heart failure. This analysis revealed that age, serum creatinine levels, and ejection fraction were the most important predictors.

## Future Work

In the future, I plan to further optimize the model performance by using more advanced hyperparameter tuning methods, exploring ensemble learning techniques, and evaluating the model's performance on larger and more diverse datasets.

## Conclusion

Overall, this project was an excellent opportunity for me to apply my data science skills in a real-world context. By building a predictive model for heart failure, I was able to gain valuable experience in data preprocessing, feature engineering, machine learning algorithms, and model evaluation. I am excited to continue working on this project and exploring new ways to improve the accuracy and performance of the heart failure prediction model.
