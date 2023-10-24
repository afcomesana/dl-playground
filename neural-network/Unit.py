import numpy as np
import random

from activation import Activation, Identity

class Unit:
    """
    - inputs: should be an list or numpy array of integers or float numbers
    - activation: if it is None that means that the activation is the identity function.
    - weights: are not usually passed as arguments when initializing the Unit, so the
    most of the time will be randomly executed to small positive numbers (0,1)
    - bias: same as weights
    """
    
    def __init__(self, activation = Identity, weights = np.array([]), bias = 0):
        self.activation = activation
        self.weights    = weights
        self.bias       = bias
        self.gradient   = None
        
        if not issubclass(self.activation, Activation):
            raise Exception("Activation function not valid.")
            
    def assert_inputs_type(self, inputs):
        """
        Check input for the unit has the proper type.
        If the type has not a numerical value raise an error
        """
        
        if isinstance(inputs, list): return True
        if isinstance(inputs, np.ndarray): return True
        
        if isinstance(inputs, float):
            inputs = np.array(inputs)
            return True
        
        if isinstance(inputs, int):
            inputs = np.array(inputs)
            return True
        
        raise "Input type is not valid."
        
    def compute(self, inputs):
        """
        Compute the output of the neuron.
        1. Performs dot product over inputs and weights.
        2. Add the bias.
        3. Apply activation function.
        """
        
        self.assert_inputs_type(inputs)
        
        self.inputs = self.output = inputs
        
        
        if len(self.weights) > 0:
            
            if len(self.weights) != len(self.inputs):
                raise Exception("Dimension mismatch, input vector sized %s while current unit has %s weights." % (len(self.inputs), len(self.weights)))
            
            self.output = np.dot(self.inputs, self.weights)
        
        self.output += self.bias
        
        self.linear_sum = self.output
        
        self.output  = self.activation.apply(self.output)
        
        return self.output
    
    def initialize_parameters(self, previous_layer_units_amount):
        self.weights = np.random.rand(previous_layer_units_amount)
        self.bias    = np.random.rand()
        
    def update_parameters(self, learning_rate = 0.1):
        if self.gradient is None: return
        # Update bias:
        self.bias -= self.gradient*learning_rate
        
        # Update weights
        self.weights -= np.array(self.gradient*learning_rate)