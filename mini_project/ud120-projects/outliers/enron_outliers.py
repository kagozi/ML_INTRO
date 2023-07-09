#!/usr/bin/python3
import os
import joblib
import sys
import numpy
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("../tools/"))

from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )

data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

salary, bonus =  targetFeatureSplit(data)
salary = numpy.array(salary)
bonus  = numpy.array(bonus)

### Reshape the data to have a single feature
salary = salary.reshape(-1, 1)
bonus = bonus.reshape(-1, 1)

salary_train, salary_test, bonus_train, bonus_test = train_test_split(salary, bonus, test_size=0.3, random_state=42)


reg = LinearRegression()
reg.fit(salary_train, bonus_train)
bonus_pred = reg.predict(salary_test)
### Calculate the mean squared error and R-squared score
mse = mean_squared_error(bonus_test, bonus_pred)
r2 = r2_score(bonus_test, bonus_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

### Plot the scatter plot of the data and the regression line
plt.scatter(salary_test, bonus_test, color='b', label='Actual')
plt.plot(salary_test, bonus_pred, color='r', label='Predicted')
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Regression Model: Salary vs Bonus")
plt.legend()
plt.show()