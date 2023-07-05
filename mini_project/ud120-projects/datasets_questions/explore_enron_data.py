#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import pandas as pd

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))
enron_names_file = "../final_project/poi_names.txt"

# print(enron_data.items())
# print(len(enron_data.items()))
count = 0

for person_name, person_data in enron_data.items():
    if person_data["poi"] == 1:
        count += 1

with open(enron_names_file, 'r') as file:
    next(file)
    
    poi = 0
    poi_count = sum(1 for line in file)
    # print(line for line in file)
# print("Total number of POIs:", poi_count)

## STOCK BELONGING TO James Prentice
enron_data["PRENTICE JAMES"]["total_stock_value"]

## Emails from Wesley Colwell to PoI
enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

## What’s the value of stock options exercised by Jeffrey K Skilling?
# print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

## Executive with most money
executives = [
    "SKILLING JEFFREY K",
    "FASTOW ANDREW S",
    "LAY KENNETH L"
]
for executive in executives:
    if executive in enron_data:
        total_payments = enron_data[executive]["total_payments"]
        # print(f"Total payments for {executive}: {total_payments}")
    else:
        print(f"No data available for {executive}")


salary_count = 0
email_count = 0

for person_data in enron_data.values():
    if person_data.get('salary'):
        salary_count += 1
    if person_data.get('email_address'):
        email_count += 1
# print(enron_data.keys())
# print("Number of folks with a quantified salary:", salary_count)
# print("Number of folks with a known email address:", email_count)
total_people = len(enron_data)
nan_payments_count = 0

for person_data in enron_data.values():
    if person_data.get('total_payments') == 'NaN':
        nan_payments_count += 1

percentage_nan_payments = (nan_payments_count / total_people) * 100

print("Number of people with 'NaN' for total payments:", nan_payments_count)
print("Percentage of people with 'NaN' for total payments:", percentage_nan_payments)

poi_count = 0
poi_nan_payments_count = 0

for person_data in enron_data.values():
    if person_data.get('poi'):
        poi_count += 1
        if person_data.get('total_payments') == 'NaN':
            poi_nan_payments_count += 1

percentage_poi_nan_payments = (poi_nan_payments_count / poi_count) * 100
print(poi_count)
# print("Number of POIs with 'NaN' for total payments:", poi_nan_payments_count)
# print("Percentage of POIs with 'NaN' for total payments:", percentage_poi_nan_payments)
