"""
File: main_framework.py
==========================================================================
Description:
This file contains the implementation of a linear regression
algorithm using the gradient descent method, using a framework. 
The dataset used is the 'imports-85.data' file, which contains 
information aboutcars and their prices. The goal is to predict the 
price of a car based on its features.

==========================================================================
Date                    Author                   Description
02/09/2024         J. Riosmena          First implementation

==========================================================================
Comments:

==========================================================================
To run:

$ python main_framework.py
"""

# Libraries needed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
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
    df = load_dataset()

    # Replace "?" with NaN and drop rows with missing 'price'
    df.replace("?", np.nan, inplace=True)
    df.dropna(subset=['price'], inplace=True)

    # Convert necessary columns to numeric
    df['price'] = pd.to_numeric(df['price'])
    df['horsepower'] = pd.to_numeric(df['horsepower'])

    # Select features and target
    features = ['wheel_base', 'curb_weight', 'engine_size', 'horsepower', 'city_mpg', 'highway_mpg']
    X = df[features]
    y = df['price']

    # Handle missing values by filling with the mean
    X.fillna(X.mean(), inplace=True)

    # Split the data into training, validation, and test sets
    train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.4, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    # Define initial values for epochs and learning rate
    epochs = 1000
    learning_rate = 0.03

    # Train the model using scikit-learn's SGDRegressor
    model = SGDRegressor(max_iter=epochs, eta0=learning_rate, tol=0.0001)
    model.fit(train_x, train_y)

    # Predict on training and validation sets
    train_pred = model.predict(train_x)
    val_pred = model.predict(val_x)

    # Calculate R^2 and MSE
    train_r2 = r2_score(train_y, train_pred)
    val_r2 = r2_score(val_y, val_pred)
    train_mse = mean_squared_error(train_y, train_pred)
    val_mse = mean_squared_error(val_y, val_pred)

    # Calculate the bias
    bias =model.intercept_

    # Print the results
    print(f"Bias: {bias}")
    print(f"Train R^2: {train_r2}")
    print(f"Validation R^2: {val_r2}")
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")

    # Plot the training and validation errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.plot(range(len(train_pred)), train_y - train_pred, label='Train Error', color='blue')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Train Error')
    ax1.set_title('Training Error')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(range(len(val_pred)), val_y - val_pred, label='Validation Error', color='green')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Validation Error')
    ax2.set_title('Validation Error')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
