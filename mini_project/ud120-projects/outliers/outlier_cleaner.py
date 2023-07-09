#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    ### your code goes here
    errors = (net_worths - predictions) ** 2  # Calculate the squared residual errors
    threshold = np.percentile(errors, 90)  # Determine the threshold for the top 10% errors

    cleaned_data = []
    for age, net_worth, error in zip(ages, net_worths, errors):
        if error <= threshold:
            cleaned_data.append((age, net_worth, error))
    print(cleaned_data)
    return cleaned_data
