import numpy as np
import logging

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# Theses are the outputs that are supposed to appear
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

# These are the synaptic weights that calculate the output
synaptic_weights = 2 * np.random.random((3, 1)) - 1

# These are the first randomly picked synaptic weights
print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(2000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Syanptic weights after training')
print(synaptic_weights)

print('Outputs after training')
print(outputs)
