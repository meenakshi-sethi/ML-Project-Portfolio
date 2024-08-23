# Diabetes Prediction Model

## Project Overview

This project focuses on developing a machine learning model to predict the likelihood of diabetes in patients based on a given set of features. While accuracy is an important metric, this project emphasizes the importance of **recall**—the model's ability to correctly identify all relevant instances, particularly those with diabetes. This approach prioritizes minimizing the cost of false negatives, which, in this context, would be failing to detect diabetes when it is actually present.

## Models Evaluated

Several models were evaluated, both before and after hyperparameter tuning, to determine the best approach for this task:

- **Naive Bayes (Untuned)**
- **Logistic Regression (Untuned and Tuned)**
- **Random Forest (Untuned and Tuned)**
- **XGBoost (Untuned and Tuned)**

### Key Findings

- **Logistic Regression (Untuned)** demonstrated the highest recall score of **0.7428** for both versions, making it the most effective model for predicting diabetes with minimal false negatives.
- **XGBoost (Untuned)** performed well with a recall score of **0.7428**, but its performance declined after tuning, with a recall score dropping to **0.1172**.
- **Random Forest (Untuned)** showed a moderate recall score of **0.3356**, which further declined after tuning to **0.0800**.
- **Naive Bayes (Untuned)** had the lowest recall score of **0.0323**, indicating it is less reliable for this specific prediction task.

### Conclusion

Given the critical nature of accurately identifying diabetic patients, **Logistic Regression** was chosen as the best model for this task due to its superior recall performance, both before and after tuning.
