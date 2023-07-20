#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Step 1: Split the data into training and testing sets
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!  

# Step 2: Create a classifier (e.g., DecisionTreeClassifier)
clf = DecisionTreeClassifier()

# Fit the classifier on the full data
clf.fit(features, labels)

# Make predictions on the same data (training data)
labels_pred = clf.predict(features)

# Evaluate the performance of the classifier on the full data
accuracy = accuracy_score(labels, labels_pred)
precision = precision_score(labels, labels_pred)
recall = recall_score(labels, labels_pred)
f1 = f1_score(labels, labels_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Split the data into training and testing sets, holding out 30% of the data for testing
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# # Fit the classifier on the training data
# clf.fit(features_train, labels_train)

# # Make predictions on the testing data
# labels_pred = clf.predict(features_test)

# # Evaluate the accuracy of the classifier on the testing data
# accuracy = accuracy_score(labels_test, labels_pred)

# print("Accuracy:", accuracy)
