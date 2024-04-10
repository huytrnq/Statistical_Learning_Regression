# Challenge Description

The challenge consists of solving a regression problem using both a parametric and a nonparametric approach. 

## Parametric Approach

For the parametric approach, you need to estimate a linear regression model using the training data from the `train_ch.csv` file. 

## Nonparametric Approach

For the nonparametric approach, you need to use KNN (properly tuned) to predict the response values for the test observations.

## Task

Your task is to build models using the training data and test them on the `test_ch.csv` observations. 

You will receive a training set (`train_ch.csv`) with 1000 observations and a test set (`test_ch.csv`) with 100 observations. Both datasets have nine independent variables.

## Submission Requirements

Your submission should include the following three files:

1. A comma-separated file named `StudentRegistrationNumber_FamilyName_chal1.csv`, with 100 rows and 2 columns, containing the output values estimated using linear regression in the first column and the estimates obtained with KNN in the second column. The file should have no header.

2. A presentation in PDF format named `StudentRegistrationNumber_FamilyName_chal1.PDF`, with up to 10 pages. In the presentation, you should describe the model and explain how you obtained it.

3. An R/Python script named `StudentRegistrationNumber_FamilyName_chal1` (with the appropriate file extension) that was used to obtain the results.

## Evaluation Criteria

Your results will be ranked based on the RMSE test and the similarity of your proposed model to the one used to generate the data.

## Additional Analysis

In addition to building the models, you are also required to analyze the following aspects:

- Possible presence of non-linearity in the response-predictor relationships
- Correlation of error terms
- Non-constant variance of error terms
- Presence of outliers and/or high-leverage points
- Collinearity/multicollinearity of predictors
- Presence of a single interaction term

### Linear Regression Results
========================= Linear Regression Model =========================
Number of rows before filtering: 1000
Number of rows after filtering: 987

| Number of Features | Features                      | Average RMSE | Average R2    |
|--------------------|-------------------------------|--------------|---------------|
| 1                  | ['v3']                        | 14.9405      | 0.99596       |
| 2                  | ['v3', 'v9']                  | 14.9402      | 0.99596       |
| 3                  | ['v3', 'v7', 'v9']            | 14.9123      | 0.99598       |
| 4                  | ['v3', 'v5', 'v7', 'v9']      | 14.9281      | 0.99597       |
| 5                  | ['v1', 'v3', 'v5', 'v7', 'v9']| 14.9567      | 0.99595       |
| 6                  | ['v1', 'v3', 'v4', 'v5', 'v7', 'v9'] | 14.9562 | 0.99595 |
| 7                  | ['v1', 'v3', 'v4', 'v5', 'v7', 'v8', 'v9'] | 14.9777 | 0.99594 |
| 8                  | ['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v8', 'v9'] | 14.9897 | 0.99594 |
| 9                  | ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'] | 14.9923 | 0.99593 |

