"""
File: main.py
==================================================================
Description:
This file contains the implementation of a linear regression
algorithm using the gradient descent method. The dataset used
is the 'imports-85.data' file, which contains information about
cars and their prices. The goal is to predict the price of a car
based on its features.

==================================================================
Date                    Author                   Description
21/08/2024         J. Riosmena          First implementation

==================================================================
Comments:

==================================================================
To run:

$ python main.py
"""

# Libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a list to store the errors
errors = []

def hypothesis(params, samples):
    """
    This function calculates the hypothesis of the linear regression
    algorithm.

    :param params(list): list of parameters (theta)
    :param samples(list): list of samples (features)
    """
    return sum(p * s for p, s in zip(params, samples))

def mean_square_error(params, samples, y):
    """
    This function calculates the mean square error of the linear
    regression algorithm.

    :param params(list): list of parameters (theta)
    :param samples(list): list of samples (features)
    :param y(list): list of results
    """

    # Accumulator to store the mean error
    acum = 0

    # Calculate the hypothesis and the error
    for i in range(len(samples)):
        hyp = hypothesis(params, samples[i])
        error = hyp - y[i]
        acum += error ** 2

        # Show the hypothesis and the real value
        print(f"Hypothesis: {hyp:.6f}, Real: {y[i]:.6f}")

    # Calculate the mean error and store it
    mean_error = acum / len(samples)
    errors.append(mean_error)

def descending_gradient(params, samples, y, alpha):
    """
    This function calculates the gradient descent of the linear
    regression algorithm.

    :param params(list): list of parameters (theta)
    :param samples(list): list of samples (features)
    :param y(list): list of results
    :param alpha(float): learning rate
    """

    # Create a copy of the parameters
    temp = params.copy()

    # Calculate the gradient descent for each parameter and store it
    for j in range(len(params)):
        acum = sum((hypothesis(params, sample) - y[i]) * sample[j] for i, sample in enumerate(samples))
        temp[j] = params[j] - alpha * (1 / len(samples)) * acum

    return temp

def scaling(samples):
    """
    This function scales the samples to avoid the overflow of the
    gradient descent.

    :param samples(list): list of samples (features)
    """

    # Get the transpose of the samples
    samples = np.array(samples).T

    # Calculate the average and the max value of each feature
    for i in range(1, len(samples)):
        average = np.mean(samples[i])
        max_value = np.max(samples[i])

        # Scale the samples
        samples[i] = (samples[i] - average) / max_value

    return samples.T.tolist()

def load_dataset():
    """
    This function loads the dataset from the 'imports-85.data' file.

    :return: dataset
    """

    # Get the columns of the dataset
    columns = ['symboling', 'normalized_losses', 'make', 'fuel_type', 
               'aspiration', 'num_doors', 'body_style', 'drive_wheels', 'engine_location', 
               'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 
               'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 
               'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
    
    # Read and store the dataset
    data = pd.read_csv('data/imports-85.data', names=columns)
    
    return data

if __name__ == "__main__":
    # Load the dataset
    df = load_dataset()

    # Create the features and results lists
    features = []
    results = []

    # Clean and prepare the dataset
    df.replace("?", pd.NA, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    df.dropna(subset=['horsepower'], inplace=True)

    # Create the features and results lists
    for i in range(len(df)):
            row = df.iloc[i]
            if row['price'] != '?':
                features.append([1.0, float(row['wheel_base']), float(row['curb_weight']), float(row['engine_size']), float(row['horsepower']), float(row['city_mpg']), float(row['highway_mpg'])])
                results.append(float(row['price']))

    # Convert the lists to numpy arrays
    x = np.array(features)
    y = np.array(results)

    # Apply the scaling to the features
    scaled_x = scaling(x)

    # Initialize the parameters, alpha and epoch
    parameters = np.zeros(len(features[0]))
    alpha = 0.5
    epoch = 0

    # Train the model
    while True:

        # Store the old parameters and calculate the mean square error
        old_params = list(parameters)
        mean_square_error(parameters, scaled_x, y)

        # Calculate the gradient descent
        parameters = descending_gradient(parameters, scaled_x, y, alpha)

        # Increase the epoch
        epoch += 1

        # Break the loop if the epoch is 2000 or the parameters are the same
        if epoch == 2000 or np.array_equal(old_params, parameters):
            break

    # Show the results
    print("===========================================================")
    print(f"Epochs: {epoch}")
    print("===========================================================")
    print("Parameters \t\tOld Parameters")
    for p, op in zip(parameters, old_params):
        print(f"{p} \t{op}")
    print("===========================================================")
    print(f"Error: {errors[-1]}")
    
    # Plot the errors
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()