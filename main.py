import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

errors = []

def hypothesis(params, samples):
    return sum(p * s for p, s in zip(params, samples))

def mean_square_error(params, samples, y):
    acum = 0
    for i in range(len(samples)):
        hyp = hypothesis(params, samples[i])
        error = hyp - y[i]
        acum += error ** 2
        print(f"Hypothesis: {hyp:.6f}, Real: {y[i]:.6f}")

    mean_error = acum / len(samples)
    errors.append(mean_error)

def descending_gradient(params, samples, y, alpha):
    temp = params.copy()
    for j in range(len(params)):
        acum = sum((hypothesis(params, sample) - y[i]) * sample[j] for i, sample in enumerate(samples))
        temp[j] = params[j] - alpha * (1 / len(samples)) * acum
    return temp

def scaling(samples):
    samples = np.array(samples).T
    for i in range(1, len(samples)):
        average = np.mean(samples[i])
        max_value = np.max(samples[i])
        samples[i] = (samples[i] - average) / max_value
    return samples.T.tolist()

def load_dataset():
    columns = ['symboling', 'normalized_losses', 'make', 'fuel_type', 
               'aspiration', 'num_doors', 'body_style', 'drive_wheels', 'engine_location', 
               'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 
               'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 
               'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
    
    data = pd.read_csv('data/imports-85.data', names=columns)
    return data

if __name__ == "__main__":
    df = load_dataset()

    features = []
    results = []

    df.replace("?", pd.NA, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    df.dropna(subset=['horsepower'], inplace=True)

    for i in range(len(df)):
            row = df.iloc[i]
            if row['price'] != '?':
                features.append([1.0, float(row['wheel_base']), float(row['curb_weight']), float(row['engine_size']), float(row['horsepower']), float(row['city_mpg']), float(row['highway_mpg'])])
                results.append(float(row['price']))

    x = np.array(features)
    y = np.array(results)

    scaled_x = scaling(x)

    parameters = np.zeros(len(features[0]))
    alpha = 0.03
    epoch = 0

    while True:
        mean_square_error(parameters, scaled_x, y)
        parameters = descending_gradient(parameters, scaled_x, y, alpha)
        epoch += 1

        if epoch == 1000:
            break

    print(f"Parameters: {parameters}")
    print(f"Errors: {errors[-5:]}")

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()