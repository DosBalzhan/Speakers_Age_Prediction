Abstract—In this report, we present a methodology for estimatiing the age of a speaker based on their speech characteristics. Our approach integrates feature engineering, including the analysis of
correlations to uncover dependencies between features and the target variable, along with hyperparameter tuning and model comparison to identify the optimal pipeline for this regression
problem. The results present the effectiveness of this approach, achieving predictive performance with Root Mean Square Error as the primary evaluation metric.

I. PROBLEM OVERVIEW
The proposed project focuses on the Predicting Speaker’s Age Dataset, a collection of audio recordings from various speakers. The goal is to develop a regression pipeline that
accurately predicts the age of a speaker for each spoken sentence. The dataset includes the following components:
• Development Set: 2,933 labeled samples with the
speaker’s age provided as the target variable.
• Evaluation Set: 691 unlabeled samples reserved for testing the model’s predictions.

II. PROPOSED APPROACH
A. Preprocessings
Firstly, the initial analysis showed that the sampling rate column was constant across all records and was therefore removed. The gender and ethnicity columns, being categorical features, were transformed using the One-Hot Encoding method. To visualize these relationships, we generated scatter plots mapping different features against age, with gender represented by distinct colors. 

B. Model Selection
After preprocessing, the dataset was split into training and testing sets using an 80/20 ratio to effectively evaluate model performance. The following models were tested:
• Linear Regression. A simple yet effective model that assumes a linear relationship between the features and the target variable.
• Lasso and Ridge (with ElasticNet). A combination of Lasso and Ridge that balances L1 and L2 regularization. The models articularly effective when features are highly
correlated, as it retains multiple correlated features while still performing feature selection.
• Random Forest Regressor. An ensemble method that constructs multiple decision trees and averages their predictions to improve robustness and accuracy. Random forests handle non-linear relationships and interactions well and are less prone to overfitting compared to single decision trees.
Among them, Lasso Regression showed the best results in its best configuration, effectively capturing key relationships between features and the target variable. This highlights its
strength in feature selection and regularization. The bestworking configuration of hyperpa-rameters has been identified through a grid search, as explained in the following section.

C. Hyperparameters tuning
The hyperparameters of Lasso Regression and Random Forest were tuned as part of this work to optimize model performance. To improve performance the Polynomial Features were added to enhance default models by introducing nonlinear relationships between features. Also Recursive Feature Elimination (RFE) was applied to select the most relevant features ( by iteratively removing the least important ones). All
dependent hyperparameter tuning was conducted using Grid Search CV with 5-fold cross-validation. The dataset is split into 5 parts, and training occurs 5 times. Final DS pipeline contains on 4 steps:
• Standard Scaler
• RFE selector (num of features to select = 33)
• Polynomial Features (degree=2)
• Lasso regression (alpha=0.1)

III. RESULTS
In this project, we explored the performance of models such as LinearRegression, Ridge, Lasso (ElasticNet with their combination) and RandomForestRegressor to select the best performing model based on RMSE (Root Mean Square Error) and runtime. To further enhance the performance of models, the following measures were implemented:
• Optimization of regularization alpha for Lasso to balance
overfitting and underfitting
• Polynomial feature generation to capture non-linear relationships in the data
• Ensembling Lasso and Random Forest models to improve
their overall efficienc

After implementing the necessary measures, we determined the optimal hyperparameters using Grid Search. Following this
optimization, we trained the models on all available development data, where they demonstrated improved performance.
Lasso Regression demonstrated better performance on the test data compared to other models. The best configuration for the Lasso was determined to be
alpha = 0.1 and nfeatures = 33, achieving an RMSE of 9.777.
For the evaluation dataset, we selected the Lasso Regression model, as it outperformed the ensemble approach on the training set. The public score achieved after implementing the
measures was 9.987
