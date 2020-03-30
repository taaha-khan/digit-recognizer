# Imports
from digit_matrix import Matrix
import math, random

# Activation Function
def sigmoid(x):
    if x < 0: return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

# Derivative of Sigmoided Value
def dsigmoid(y):
    # return sigmoid(x) * (1 - sigmoid(x))
    return y * (1 - y)

# Network
class NeuralNetwork:

    # Constructor
    def __init__(self, a, b, c):
        
        # Copy Same Neural Network
        if isinstance(a, NeuralNetwork):
            self.input_nodes = a.input_nodes
            self.hidden_nodes = a.input_nodes
            self.output_nodes = a.input_nodes

            self.weights_ih = a.weights_ih.copy()
            self.weights_ho = a.weights_ho.copy()

            self.bias_h = a.bias_h.copy()
            self.bias_o = a.bias_o.copy()


        else:  # New Neural Network
            # Network Node Amounts
            self.input_nodes = a
            self.hidden_nodes = b
            self.output_nodes = c

            # Weight Matrices
            self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
            self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
            self.weights_ih.randomize()
            self.weights_ho.randomize()

            # Bias Matrices
            self.bias_h = Matrix(self.hidden_nodes, 1)
            self.bias_o = Matrix(self.output_nodes, 1)
            self.bias_h.randomize()
            self.bias_o.randomize()
            self.learning_rate = 0.1


    # Feedforward Algorithm
    def predict(self, input_array):

        # Generating Hidden Outputs
        inputs = Matrix.fromArray(Matrix, input_array)
        hidden = Matrix.static_multiply(Matrix, self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation Function
        hidden.map(sigmoid)
        
        # Generating Output
        output = Matrix.static_multiply(Matrix, self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)

        return output.toArray()

    # Normalizing Outputs to 100%
    def softmax(self, outputs, A):
        outputs = outputs.exp(A)
        return outputs / outputs.sum()
    
    # Backpropagation Training Algorithm
    def train(self, input_array, target_array):

        # FEEDFORWARD ALGORITHM ---------

        # Generating Hidden Outputs
        inputs = Matrix.fromArray(Matrix, input_array)
        hidden = Matrix.static_multiply(Matrix, self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation Function
        hidden.map(sigmoid)
        
        # Generating Output
        outputs = Matrix.static_multiply(Matrix, self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)

        # -------------------------------

        # Converting Arrays to Matrix Objects
        targets = Matrix.fromArray(Matrix, target_array)

        # Calculating Errors
        output_errors = Matrix.static_subtract(Matrix, targets, outputs)
        
        # Calculating Gradient
        gradients = Matrix.static_map(Matrix, outputs, dsigmoid)
        gradients.multiply(output_errors)
        gradients.multiply(self.learning_rate)

        # Calculate Deltas
        hidden_T = Matrix.static_transpose(Matrix, hidden)
        weigths_ho_deltas = Matrix.static_multiply(Matrix, gradients, hidden_T)

        # Adjusting Weights & Bias (hidden -> output)
        self.weights_ho.add(weigths_ho_deltas)
        self.bias_o.add(gradients)

        # Calculating Hidden Layer Errors
        who_t = Matrix.static_transpose(Matrix, self.weights_ho)
        hidden_errors = Matrix.static_multiply(Matrix, who_t, output_errors)
        
        # Calculate Hidden Gradient
        hidden_gradient = Matrix.static_map(Matrix, hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)

        # Calculate (Input -> Hidden) Deltas
        inputs_T = Matrix.static_transpose(Matrix, inputs)
        weight_ih_deltas = Matrix.static_multiply(Matrix, hidden_gradient, inputs_T)

        # Adjusting Weights (input -> hidden)
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hidden_gradient)


    # Neuro-evolution Functions

    # Copying same Neural Network
    def copy(self):
        return NeuralNetwork(self, None, None)
    
    # Mutating Child Neural Network Weights
    def mutate(self, rate):

        def change(self):
            if random.random() < rate:
                return (random.random() * 2) - 1
            else: return self

        self.weights_ih.map(change)
        self.weights_ho.map(change)
        self.bias_h.map(change)
        self.bias_o.map(change)
