import numpy as np
import matplotlib.pyplot as plt

errors = []

def hypothesis(params, samples):
    return sum(p * s for p, s in zip(params, samples))

def show_errors(params, samples, y):
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

if __name__ == "__main__":
    pass