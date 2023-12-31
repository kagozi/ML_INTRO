#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
# features_list = [poi, feature_1, feature_2, feature_3]

data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
### Feature Scaling ###
scaler = MinMaxScaler()
finance_features = scaler.fit_transform(finance_features)

for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
# Perform k-means clustering
kmeans = KMeans(n_clusters=2,init='k-means++', n_init='warn', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
kmeans.fit(finance_features)
pred = kmeans.labels_


### What are the maximum and minimum values taken by the “exercised_stock_options” feature
# Initialize variables for maximum and minimum values
max_exercised_stock_options = float('-inf')
min_exercised_stock_options = float('inf')

# Iterate over the data_dict dictionary
for person, features in data_dict.items():
    exercised_stock_options = features['exercised_stock_options']
    # Check if the value is not NaN and update maximum and minimum values
    if exercised_stock_options != 'NaN':
        if exercised_stock_options > max_exercised_stock_options:
            max_exercised_stock_options = exercised_stock_options
        if exercised_stock_options < min_exercised_stock_options:
            min_exercised_stock_options = exercised_stock_options

# Print the maximum and minimum values
print("Maximum exercised_stock_options:", max_exercised_stock_options)
print("Minimum exercised_stock_options:", min_exercised_stock_options)

### What are the maximum and minimum values taken by the “salaries” feature
# Initialize variables for maximum and minimum values
max_salaries = float('-inf')
min_salaries = float('inf')

# Iterate over the data_dict dictionary
for person, features in data_dict.items():
    salaries = features['salary']
    # Check if the value is not NaN and update maximum and minimum values
    if salaries != 'NaN':
        if salaries > max_salaries:
            max_salaries = salaries
        if salaries < min_salaries:
            min_salaries = salaries

# Print the maximum and minimum values
print("Maximum salaries:", max_salaries)
print("Minimum salaries:", min_salaries)
### Calculate rescaled values ###
salary = 200000.0
exercised_stock_options = 1000000.0

rescaled_salary = scaler.transform([[salary, 0]])[0][0]
rescaled_exercised_stock_options = scaler.transform([[0, exercised_stock_options]])[0][1]

print("Rescaled Salary:", rescaled_salary)
print("Rescaled Exercised Stock Options:", rescaled_exercised_stock_options)

## rename the "name" parameter when you change the number of features
## so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
