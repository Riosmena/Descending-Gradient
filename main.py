"""
File: main.py
==========================================================================
Description:
This file contains the implementation of a linear regression
algorithm using the gradient descent method. The dataset used
is the 'imports-85.data' file, which contains information about
cars and their prices. The goal is to predict the price of a car
based on its features.

==========================================================================
Date                    Author                   Description
21/08/2024         J. Riosmena          First implementation
22/08/2024         J. Riosmena          Second attempt to improve the model

==========================================================================
Comments:

==========================================================================
To run:

$ python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create lists to store errors
train_errors = []
val_errors = []

# Define the hypothesis function for linear regression
def hypothesis(params, samples):
    """
    This function calculates the hypothesis (prediction) for linear regression.
    
    :param params(list): list of parameters (theta)
    :param samples(list): list of sample features
    :return: predicted value (hypothesis)
    """
    return sum(p * s for p, s in zip(params, samples))

# Define the mean squared error function
def mean_square_error(params, samples, y):
    """
    This function calculates the mean squared error (MSE) of the model.
    
    :param params(list): list of parameters (theta)
    :param samples(list): list of sample features
    :param y(list): list of actual results
    :return: mean squared error
    """
    acum = 0
    for i in range(len(samples)):
        hyp = hypothesis(params, samples[i])
        error = hyp - y[i]
        acum += error ** 2
    return acum / len(samples)

# Define the gradient descent function
def descending_gradient(params, samples, y, alpha):
    """
    This function performs the gradient descent optimization.
    
    :param params(list): list of parameters (theta)
    :param samples(list): list of sample features
    :param y(list): list of actual results
    :param alpha(float): learning rate
    :return: updated parameters after one step of gradient descent
    """
    temp = params.copy()
    for j in range(len(params)):
        acum = sum((hypothesis(params, sample) - y[i]) * sample[j] for i, sample in enumerate(samples))
        temp[j] = params[j] - alpha * (1 / len(samples)) * acum
    return temp

# Define the feature scaling function
def scaling(samples):
    """
    This function scales the features to avoid overflow during gradient descent.
    
    :param samples(list): list of sample features
    :return: scaled features
    """
    samples = np.array(samples).T
    for i in range(1, len(samples)):
        average = np.mean(samples[i])
        max_value = np.max(samples[i])
        samples[i] = (samples[i] - average) / max_value
    return samples.T.tolist()

# Calculate the R-squared value
def r_squared(predicted_y, real_y):
    """
    This function calculates the R-squared value to evaluate the model's performance.
    
    :param predicted_y(list): predicted values from the model
    :param real_y(list): actual values from the data
    :return: R-squared value
    """
    mean_y = np.mean(real_y)
    ss_total = sum((real_y - mean_y) ** 2)
    ss_res = sum((real_y - predicted_y) ** 2)
    return 1 - (ss_res / ss_total)

# Load and prepare the dataset
def load_dataset():
    """
    This function loads the dataset from a CSV file and returns it as a DataFrame.
    
    :return: pandas DataFrame containing the dataset
    """
    columns = ['symboling', 'normalized_losses', 'make', 'fuel_type', 
               'aspiration', 'num_doors', 'body_style', 'drive_wheels', 
               'engine_location', 'wheel_base', 'length', 'width', 
               'height', 'curb_weight', 'engine_type', 'num_cylinders', 
               'engine_size', 'fuel_system', 'bore', 'stroke', 
               'compression_ratio', 'horsepower', 'peak_rpm', 
               'city_mpg', 'highway_mpg', 'price']
    
    data = pd.read_csv('data/imports-85.data', names=columns)
    
    return data

if __name__ == "__main__":
    # Load the dataset
    df = load_dataset()

    # Replace "?" with 0 and convert necessary columns to numeric types
    df.replace("?", 0, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    # Create lists for the columns to use
    price_vals = []
    wheel_base_vals = []
    curb_weight_vals = []
    engine_size_vals = []
    horsepower_vals = []
    city_mpg_vals = []
    highway_mpg_vals = []

    # Create the features and results lists
    features = []
    results = []

    # Store the values of the columns
    for i in range(len(df)):
        row = df.iloc[i]
        price_vals.append(float(row['price']))
        wheel_base_vals.append(float(row['wheel_base']))
        curb_weight_vals.append(float(row['curb_weight']))
        engine_size_vals.append(float(row['engine_size']))
        horsepower_vals.append(float(row['horsepower']))
        city_mpg_vals.append(float(row['city_mpg']))
        highway_mpg_vals.append(float(row['highway_mpg']))

    # Calculate the mean of the columns
    price_mean = np.mean(price_vals)
    wheel_base_mean = np.mean(wheel_base_vals)
    curb_weight_mean = np.mean(curb_weight_vals)
    engine_size_mean = np.mean(engine_size_vals)
    horsepower_mean = np.mean(horsepower_vals)
    city_mpg_mean = np.mean(city_mpg_vals)
    highway_mpg_mean = np.mean(highway_mpg_vals)

    # Create the features and results lists
    for i in range(len(df)):
        row = df.iloc[i]
        price_val = float(row['price'])
        wheel_base_val = float(row['wheel_base'])
        curb_weight_val = float(row['curb_weight'])
        engine_size_val = float(row['engine_size'])
        horsepower_val = float(row['horsepower'])
        city_mpg_val = float(row['city_mpg'])
        highway_mpg_val = float(row['highway_mpg'])
        
        # Handle missing values
        if row['price'] == 'nan':
            continue

        if row['wheel_base'] == 'nan' or row['curb_weight'] == 'nan' or row['engine_size'] == 'nan' or row['horsepower'] == 'nan' or row['city_mpg'] == 'nan' or row['highway_mpg'] == 'nan':
            price_val = price_mean
            wheel_base_val = wheel_base_mean
            curb_weight_val = curb_weight_mean
            engine_size_val = engine_size_mean
            horsepower_val = horsepower_mean
            city_mpg_val = city_mpg_mean
            highway_mpg_val = highway_mpg_mean

        # Append features and result to their respective lists
        features.append([1, wheel_base_val, curb_weight_val, engine_size_val, horsepower_val, city_mpg_val, highway_mpg_val])
        results.append(price_val)

    # Convert the lists to numpy arrays
    x = np.array(features)
    y = np.array(results)

    # Split the data into training, validation, and test sets
    train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.4, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

    # Apply feature scaling
    scaled_train_x = scaling(train_x)
    scaled_val_x = scaling(val_x)
    scaled_test_x = scaling(test_x)

    # Initialize parameters, learning rate, and epoch counter
    parameters = np.zeros(len(features[0]))
    alpha = 0.5
    epoch = 0

    # Train the model using gradient descent
    while True:
        parameters = descending_gradient(parameters, scaled_train_x, train_y, alpha)
        train_error = mean_square_error(parameters, scaled_train_x, train_y)
        train_errors.append(train_error)
        val_error = mean_square_error(parameters, scaled_val_x, val_y)
        val_errors.append(val_error)
        
        print(f"Epoch: {epoch} \tTrain Error: {train_error} \tValidation Error: {val_error}")

        if epoch == 1000:
            break
    
        epoch += 1 

    # Output the final parameters
    print("\nParameters:")
    for param in parameters:
        print(param)

    # Predict an example price
    example_features = [1, 95.9, 2337, 109, 90, 24, 30]
    scaled_example_features = scaling([example_features])[0]
    example_price = hypothesis(parameters, scaled_example_features)
    print(f"\nExample prediction price: {example_price}")

    # Calculate R-squared for training and validation sets
    train_predicted_y = [hypothesis(parameters, sample) for sample in scaled_train_x]
    val_predicted_y = [hypothesis(parameters, sample) for sample in scaled_val_x]

    print(f"\nTrain R^2: {r_squared(train_predicted_y, train_y)}")
    print(f"Validation R^2: {r_squared(val_predicted_y, val_y)}")

    # Plot the training and validation errors
    plt.plot(range(epoch + 1), train_errors, label='Train Error')
    plt.plot(range(epoch + 1), val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Train and Validation Error')
    plt.legend()
    plt.grid(True)
    plt.show()