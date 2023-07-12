#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "./your_word_data.pkl"
authors_file = "./your_email_authors.pkl" 
word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
print(type(word_data))
print(type(authors))
print(len(word_data))
print(len(authors))

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
# Get the feature names (associated words)
feature_names = vectorizer.get_feature_names_out()

# ...

# Print all the associated words
print(feature_names)

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
num_features = len(features_train[0])
# print(num_features)
# print(len(labels_train))
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
feature_importances = clf.feature_importances_

imp = numpy.max(feature_importances)
print(imp)
index_of_maximum = numpy.where(feature_importances == imp)
print("Maximum Index position: ",index_of_maximum)
# Find the index of the most important feature
max_importance_index = numpy.argmax(feature_importances)

# Get the associated word
most_powerful_word = feature_names[max_importance_index]

# Print the most powerful word
print("Most Powerful Word:", most_powerful_word)
labels_pred = clf.predict(features_test)
accuracy = metrics.accuracy_score(labels_pred,labels_test)
print(accuracy)