# Extensive Report on Diabetes Risk Prediction Using Machine Learning

### 1. Objective

The primary goal of this project was to develop models that can predict which individuals are at high risk for developing diabetes. By doing this, we hope to enable early intervention, which could prevent the onset of diabetes or lessen its impact. Additionally, understanding the key factors that contribute to diabetes risk can help target prevention efforts more effectively.

---

### 2. Data Overview and Preparation

#### 2.1 Dataset Description

The dataset I used includes information from 223,243 individuals, with 22 features that cover a range of health indicators, lifestyle habits, and demographic information. The target variable, `Diabetes_binary`, indicates whether an individual has diabetes (1) or not (0). Some of the key features include blood pressure (HighBP), cholesterol levels (HighChol), BMI, smoking habits, physical activity, and alcohol consumption, along with demographic data like age, sex, education level, and income.

#### 2.2 Data Cleaning and Preprocessing

**Handling Missing and Duplicate Values:**
Before diving into analysis, I checked for missing and duplicate data. Thankfully, the dataset was clean with no missing values or duplicates, so I didn't need to perform any imputation or data removal.

**Optimizing Data Types:**
To make the data easier to work with and to improve computational efficiency, I optimized the data types. For example, I converted the `BMI` column to `float32` and the `Age` column to `int8`. This helped in reducing the memory usage and made the analysis faster.

### 3. Exploratory Data Analysis (EDA)

#### 3.1 Summary Statistics

I started by calculating summary statistics to get a sense of the data's distribution. This helped in understanding the central tendencies, variability, and range of each feature.

**Key Insights:**
1. **Diabetes Prevalence:** About 15% of the individuals in the dataset have diabetes, which is relatively low.
2. **Cardiovascular Risk Factors:** High blood pressure (43%) and high cholesterol (41%) are quite common, indicating a significant presence of cardiovascular risk factors.
3. **Health Behaviors:** Most individuals are physically active (77%) and have had their cholesterol checked (96%), which shows good preventive health behavior. However, smoking (43%) and heavy alcohol consumption (7%) are still concerns.

#### 3.2 Univariate Analysis

To further explore the data, I performed univariate analysis on each feature individually.

**Key Findings:**
- **Binary Variables:** Variables like `HighBP` and `HighChol` show a clear distinction between those with and without these conditions.
- **Continuous Variables:** For instance, BMI has a right-skewed distribution, indicating most individuals have a BMI in the lower range, but some have very high values.
- **Demographics:** The dataset is almost evenly split between genders, and most individuals fall into middle or higher education and income brackets, which often correlate with better health outcomes.

#### 3.3 Bivariate Analysis

Next, I examined the relationships between each feature and the target variable, `Diabetes_binary`.

**Observations:**
- **Binary Variables:** While some features like `HighBP` and `HighChol` show possible associations with diabetes, it’s clear that more complex analyses, such as chi-square tests, are needed to confirm these relationships.
- **Continuous Variables:** For features like BMI and mental health days, scatter plots suggested potential correlations with diabetes. However, the relationships aren’t straightforward and may require non-linear modeling techniques to fully understand.

### 4. Data Preparation

#### 4.1 Handling Outliers

Outliers can skew the results of the analysis, so I carefully examined features like BMI, physical health, and mental health for extreme values. Using box plots, I identified and handled outliers to ensure that they wouldn’t distort the results. This step helped in improving the data quality for better model performance.

#### 4.2 Feature Selection and Engineering

To build a reliable predictive model, I needed to focus on the most relevant features.

**Correlation Matrix:**
I created a heatmap to visualize how each feature correlates with diabetes. Some features like age, education, and income showed a stronger relationship with diabetes, which suggested they should be prioritized in the model.

**Chi-Square Test:**
To further refine the features, I used the chi-square test. This statistical method helped me identify the most impactful features, such as physical health, BMI, and high blood pressure. Focusing on these features would likely improve the model’s accuracy.

#### 4.3 Scaling and Balancing the Data

Before training the models, I split the data into training and test sets and applied scaling techniques to ensure that all numerical features were on the same scale. This step was essential because many machine learning algorithms work better when the data is standardized.

**Handling Class Imbalance:**
Given that the dataset was imbalanced (with more non-diabetic cases), I used class weights to give more importance to the minority class (diabetic cases). This approach helped in training models that could better predict diabetes.

### 5. Model Development

You're right; including ROC-AUC and RMSE, as well as the cross-validation accuracy, will provide a more complete picture of the model's performance. Below is the revised **Model Development** section that includes these metrics:

---

### 5. Model Development

In this project, I experimented with several machine learning models to determine which one would be the most effective for predicting diabetes risk. The models I tested included Naive Bayes, Logistic Regression, Random Forest, and XGBoost. For each model, I evaluated their performance both before and after hyperparameter tuning.

| **Performance Metric** | **Naive Bayes (Untuned)** | **Logistic Regression (Untuned)** | **Logistic Regression (Tuned)** | **Random Forest (Untuned)** | **Random Forest (Tuned)** | **XGBoost (Untuned)** | **XGBoost (Tuned)** |
|------------------------|---------------------------|-----------------------------------|---------------------------------|-----------------------------|-------------------------|-----------------------|---------------------|
| **Accuracy**           | 85.53%                    | 71.26%                           | 85.84%                          | 78.91%                      | 85.32%                  | 71.26%                | 86.01%              |
| **Precision**          | 0.4875                    | 0.30                             | 0.55                            | 0.297                       | 0.592                   | 0.300                 | 0.577               |
| **Recall**             | 0.0323                    | 0.74                             | 0.11                            | 0.336                       | 0.080                   | 0.743                 | 0.117               |
| **F1-Score**           | 0.0606                    | 0.43                             | 0.18                            | 0.315                       | 0.141                   | 0.427                 | 0.195               |
| **ROC-AUC**            | 0.5133                    | 0.73                             | 0.55                            | 0.601                       | 0.5353                  | 0.725                 | 0.551               |
| **RMSE**               | 0.3803                    | 0.5361                           | 0.3763                          | 0.4592                      | 0.3752                  | 0.5361                | 0.3740              |
| **Key Takeaway**       | Good accuracy but poor recall, indicating it missed many diabetic cases. High RMSE and low ROC-AUC suggest better prediction for non-diabetic cases. | Slight improvement in accuracy and ROC-AUC, but still limited in detecting diabetic cases due to model assumptions. | Strong recall but low precision, leading to more false positives. Reasonably good ROC-AUC, with a high RMSE indicating room for improvement. | Significant improvement in accuracy and precision, but recall decreased. ROC-AUC dropped, indicating reduced class discrimination. RMSE improved. | Balanced accuracy and recall, better at predicting positive cases than Naive Bayes. Moderate ROC-AUC, with RMSE slightly better than Logistic Regression. | Improved precision and accuracy but reduced recall. Slight drop in ROC-AUC, with better RMSE indicating overall prediction accuracy improvement. | High recall but low precision, similar to untuned Logistic Regression. Good ROC-AUC with high RMSE, indicating potential for enhancement. | Improved accuracy and precision after tuning, but recall decreased. Reduced ROC-AUC with improved RMSE, reflecting better overall prediction accuracy. |

The models evaluated—Naive Bayes, Logistic Regression, Random Forest, and XGBoost—exhibited varying performance metrics, with each model showing strengths and weaknesses across accuracy, precision, recall, F1-score, ROC-AUC, and RMSE. Untuned versions generally favored either precision or recall, while tuning often resulted in trade-offs, such as improved accuracy at the expense of recall or vice versa. Despite tuning, changes in ROC-AUC and RMSE highlighted shifts in each model’s ability to discriminate between classes and predict accurately.

**Summary of Model Performance:**
- **Untuned Models:** Logistic Regression and XGBoost showed the highest recall, making them better at identifying diabetic cases. However, their lower precision indicates they had more false positives.
- **Tuned Models:** After tuning, Random Forest and XGBoost achieved higher accuracy and precision, but their ability to identify all diabetic cases (recall) decreased. This trade-off highlights the importance of deciding whether it’s more critical to correctly identify all diabetic cases (high recall) or to ensure that positive predictions are more likely to be correct (high precision).

The choice between using the untuned or tuned version of the models depends on the specific goals of the project. For a healthcare application where it’s crucial not to miss any diabetic cases, the untuned Logistic Regression or XGBoost models might be preferred. However, if the goal is to reduce the number of false positives, the tuned versions of Random Forest or XGBoost would be more suitable.

---

### Cross-Validation Accuracy

To ensure the models were not overfitting and their performance was consistent across different subsets of the data, I performed cross-validation.

**Average Cross-Validation Accuracy:**
- **Naive Bayes (Untuned):** 78.84%
- **Logistic Regression (Untuned):** 85.77%
- **Random Forest (Untuned):** 84.09%
- **XGBoost (Untuned):** 85.76%
- **Logistic Regression (Tuned):** 85.77%
- **Random Forest (Tuned):** 85.89%
- **XGBoost (Tuned):** 85.93%

**Key Takeaway:** The cross-validation results confirmed that the tuned models of Random Forest and XGBoost consistently provided higher accuracy across different subsets of the data, validating their overall performance. Logistic Regression, both untuned and tuned, also showed stable performance, with the untuned version excelling in recall. The slightly lower cross-validation accuracy of Naive Bayes indicates that it may not generalize as well as the other models.

---

### Model Selection and Decision-Making Rationale

In this project, the goal was to develop a model capable of accurately predicting diabetes risk among individuals, with the overarching aim of facilitating early intervention and improving healthcare outcomes. Given the critical nature of the task, selecting the right model involved careful consideration of various performance metrics, including accuracy, precision, recall, ROC-AUC, and RMSE, as well as cross-validation results to ensure model reliability.

#### 1. Defining the Objective

The primary objective was to ensure that the model could identify as many potential diabetic cases as possible, minimizing the chances of false negatives. In healthcare, particularly in disease prediction, recall (sensitivity) often takes precedence because missing a case could lead to serious consequences, including delayed treatment and deteriorating health conditions. However, precision was also important to avoid unnecessary treatments or anxiety caused by false positives.

#### 2. Evaluating Model Performance

**Untuned Models:**
- **Logistic Regression (Untuned):** This model demonstrated the highest recall (0.74), making it exceptionally effective at identifying diabetic cases. This is crucial in healthcare, where the cost of missing a diagnosis is high. However, it had a lower precision, meaning that while it caught most cases, it also produced more false positives.
- **XGBoost (Untuned):** Similarly, the untuned XGBoost model also showed strong recall (0.743) and reasonable precision, making it another strong candidate for scenarios where identifying all possible cases is the priority.

**Tuned Models:**
- **XGBoost (Tuned):** After tuning, XGBoost achieved the highest overall accuracy (86.01%) with improved precision but at the cost of recall. This model became more balanced, reducing false positives while still maintaining reasonable recall.
- **Random Forest (Tuned):** The tuned Random Forest model also showed a good balance of accuracy and precision, although its recall decreased. This model is suitable for contexts where precision is more critical, such as when minimizing false positives is a priority.

#### 3. Considering the Context

Given the healthcare context of this project, where the early detection of diabetes is vital, models that maximize recall were prioritized. The untuned Logistic Regression and XGBoost models were particularly effective in this regard, ensuring that nearly all potential diabetic cases were identified, even if it meant accepting a higher number of false positives.

However, in scenarios where resources are limited or where false positives could lead to unnecessary anxiety or treatment, the precision of the model becomes more important. In such cases, the tuned XGBoost or Random Forest models, which offer a better balance between precision and recall, may be more appropriate.

#### 4. Cross-Validation Insights

The cross-validation results further validated the robustness of the tuned models, with XGBoost and Random Forest consistently showing high accuracy across different data subsets. This consistency indicates that these models are less likely to overfit and are reliable for real-world application.

#### 5. Final Decision

Based on the above analysis, the final model selection is guided by the specific needs of the healthcare context:

- **If the primary goal is to ensure that nearly all diabetic cases are detected (high recall),** the **untuned Logistic Regression** or **untuned XGBoost** models are recommended. These models prioritize recall, ensuring that few cases are missed.
- **If the goal is to balance recall with reducing false positives (high precision),** the **tuned XGBoost** or **tuned Random Forest** models are recommended. These models offer a more balanced approach, making them suitable for contexts where precision is also highly valued.
- **For a well-rounded approach,** the **tuned XGBoost** model, which consistently provided high accuracy and a good balance of precision and recall, is a strong candidate for further development and deployment.

#### 6. Future Considerations

For even more refined outcomes, an ensemble approach combining the strengths of both the untuned (high recall) and tuned (high precision) models could be explored. Additionally, further hyperparameter tuning and feature engineering might yield models that can even better balance the trade-offs between recall, precision, and accuracy, ensuring optimal performance in predicting diabetes risk.
