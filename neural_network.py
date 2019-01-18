import numpy as np
import math

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1 / (1 + math.exp(gamma))
  else:
    return 1 / (1 + math.exp(-gamma))

v_sigmoid = np.vectorize(sigmoid)

def deriv_sigmoid(a):
    return a * (1 - a)

v_deriv_sigmoid = np.vectorize(deriv_sigmoid)

# Cost function for individual neuron
def cost_function(output, target):
    # Add 0.0001 to avoid having to evaluate math.log(0)
    return target * math.log(output + 0.0001) + (1 - target) * math.log(1 - output + 0.0001)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.learning_rate = learning_rate
        self.layers = []
        for size in layer_sizes:
            self.layers.append(np.zeros(size))

        self.weights = []
        self.biases = []
        for i in range(0, len(self.layers) - 1):
            # Initialises weights to random values between -1 and 1
            random_weights = np.random.rand(self.layers[i + 1].size, self.layers[i].size)
            random_weights = 2 * random_weights - 1
            self.weights.append(random_weights)

            # Initialises biases to random values between -1 and 1
            random_biases = np.random.rand(self.layers[i + 1].size)
            random_biases = 2 * random_biases - 1
            self.biases.append(random_biases)

    # Computes the output from the given input using the current weights and biases
    def compute(self, input):
        self.layers[0] = np.array(input)
        for i in range(0, len(self.layers) - 1):
            self.layers[i + 1] = np.matmul(self.weights[i], self.layers[i]) + self.biases[i]
            self.layers[i + 1] = v_sigmoid(self.layers[i + 1])
        # Returns the output layer as a list
        return self.layers[-1].tolist()

    # Calculates the value of the cost function over the supplied data set
    # Useful for diagnosing underfitting/overfitting and also to check for convergence
    def calculate_cost(self, data, targets):
        sum = 0
        for i in range(0, len(data)):
            output = self.compute(data[i])
            for j in range(0, len(output)):
                sum += cost_function(output[j], targets[i][j])
        return sum / len(data) * -1

    def backprop(self, target):
        # Each element of error corresponds to the errors for one layer
        # Here 0 is simply used as a placeholder
        error = [0] * (len(self.layers) - 1)

        # Calculate error for output layer
        error[-1] = (target - self.layers[-1])

        # Calculate error for hidden layers
        for i in range(len(error) - 2, -1, -1):
            transposed = np.transpose(self.weights[i + 1])
            error[i] = np.matmul(transposed, error[i + 1]) * v_deriv_sigmoid(self.layers[i + 1])

        # Calculate the changes that need to be made to the weights and biases
        # in order to reduce the cost function value.
        delta_weights = []
        delta_biases = []
        for i in range(0, len(self.weights)):
            error_reshape = np.reshape(error[i], (error[i].size, 1))
            layer_reshape = np.reshape(self.layers[i], (1, self.layers[i].size))
            delta_weights.append(np.matmul(error_reshape, layer_reshape))
            delta_biases.append(error[i])

        return delta_weights, delta_biases

    # Performs stochastic gradient descent on a single piece of data
    def train(self, input, target):
        self.compute(input)
        dw, db = self.backprop(target)
        for j in range(0, len(self.weights)):
            self.weights[j] += self.learning_rate * dw[j]
            self.biases[j] += self.learning_rate * db[j]
