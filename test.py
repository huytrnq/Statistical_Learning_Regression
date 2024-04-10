import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load libraries
from ISLR2 import *
from scipy.stats import t

# Load Boston dataset
Boston = Boston.load()
Boston = pd.DataFrame(Boston.data, columns=Boston.feature_names)
Boston['medv'] = Boston.target

# Simple Linear Regression
lm_fit = smf.ols('medv ~ lstat', data=Boston).fit()
print(lm_fit.summary())

# Confidence interval and prediction interval
new_data = pd.DataFrame({'lstat': [5, 10, 15]})
pred_conf = lm_fit.get_prediction(new_data).summary_frame(alpha=0.05)
print(pred_conf)

# Plot
plt.scatter(Boston['lstat'], Boston['medv'], facecolors='none', edgecolors='b')
plt.plot(Boston['lstat'], lm_fit.predict(), color='red')
plt.xlabel('lstat')
plt.ylabel('medv')
plt.show()

# Multiple Linear Regression
lm_fit_multiple = smf.ols('medv ~ lstat + age', data=Boston).fit()
print(lm_fit_multiple.summary())

# Interaction Terms
lm_interaction = smf.ols('medv ~ lstat * age', data=Boston).fit()
print(lm_interaction.summary())

# Non-linear Transformations of the Predictors
lm_fit2 = smf.ols('medv ~ lstat + I(lstat**2)', data=Boston).fit()
print(lm_fit2.summary())

# Qualitative Predictors
Carseats = Carseats.load()
Carseats = pd.DataFrame(Carseats.data, columns=Carseats.feature_names)
Carseats['Sales'] = Carseats.target

lm_fit_qualitative = smf.ols('Sales ~ Income + Advertising + Price + Age + Income:Advertising + Price:Age + C(ShelveLoc)', 
                            data=Carseats).fit()
print(lm_fit_qualitative.summary())
