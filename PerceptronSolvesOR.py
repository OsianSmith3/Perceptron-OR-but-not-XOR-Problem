#Part 2-----------------------------------------------------------------
# Show that a perceptron can solve the OR problem but not the XOR problem.

import numpy as np 

# OR / XOR input data
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# OR output data
outputs = np.array([0, 1, 1, 1])

# XOR output data
outputs_xor = np.array([0, 1, 1, 0])

# Perceptron training function
def train_perceptron(inputs, outputs):
    # Initialise weights and bias
    weights = np.random.rand(inputs.shape[1])
    bias = np.random.rand(1)

    # Set learning rate
    learning_rate = 0.1

    # Train the perceptron
    for _ in range(1000):
        for i in range(len(inputs)):
            # Compute predicted output
            predicted_output = np.dot(inputs[i], weights) + bias

            # Update weights and bias
            update = learning_rate * (outputs[i] - (predicted_output > 0).astype(int))
            weights += update * inputs[i]
            bias += update  

    return weights, bias.item() 

# Train the perceptron
weights, bias = train_perceptron(inputs, outputs)

# Train the perceptron with XOR inputs and outputs
weights_xor, bias_xor = train_perceptron(inputs, outputs_xor)

# Test the trained perceptron
def test_perceptron(inputs, weights, bias, outputs):
    for i in range(len(inputs)):
        predicted_output = np.dot(inputs[i], weights) + bias
        predicted_output_scalar = predicted_output.item()
        trained_output = (predicted_output > 0).astype(int)
        print(f"Input: {inputs[i]}, Trained Output: {trained_output}, Expected Output: {outputs[i]}")
       

# Test the trained perceptron on XOR inputs
def test_perceptron_xor(inputs, weights, bias, outputs):
    for i in range(len(inputs)):
        predicted_output = np.dot(inputs[i], weights) + bias
        predicted_output_scalar = predicted_output.item()  
        trained_output = (predicted_output > 0).astype(int)
        print(f"Input: {inputs[i]}, Trained Output: {trained_output}, Expected Output: {outputs[i]}")
        

# Test the trained perceptron on OR inputs
print("==========Testing OR Problem==========")
print("")
test_perceptron(inputs, weights, bias, outputs)

# Test the trained perceptron on XOR inputs
print("")
print("==========Testing XOR Problem==========")
print("")
test_perceptron_xor(inputs, weights_xor, bias_xor, outputs_xor)
print("Shows failure to solve the XOR problem with a single layer perceptron.")
