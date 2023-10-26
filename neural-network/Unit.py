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
    
    def __init__(self, activation = Identity, weights = np.array([]), bias = None):
        self.activation = activation
        self.weights    = weights
        self.bias       = bias
        self.gradient   = None
        
        if not issubclass(self.activation, Activation):
            raise Exception("Activation function not valid.")
            
        
    def compute(self, inputs, apply_activation=True, verbose=False):
        """
        Compute the output of the neuron.
        1. Performs dot product over inputs and weights.
        2. Add the bias.
        3. Apply activation function.
        """
        
        
        self.inputs = self.output = inputs
        
        
        if len(self.weights) > 0:
            if len(self.weights) != len(self.inputs):
                raise Exception("Dimension mismatch, input vector sized %s while current unit has %s weights." % (len(self.inputs), len(self.weights)))
            
            self.output = np.dot(self.inputs, self.weights)
        

        if not self.bias is None:        
            self.output += self.bias
        
        self.linear_sum = self.output

        if apply_activation:
            self.output  = self.activation.apply(self.output)
    
        return self.output
    
    def initialize_parameters(self, previous_layer_units_amount):
        
        if len(self.weights) == 0:
            self.weights = np.random.rand(previous_layer_units_amount)
            
        if self.bias is None:
            self.bias = np.random.rand()
        
    def update_parameters(self, learning_rate = 0.1):
        if self.gradient is None: return
        
        
        
        # Update bias:
        self.bias -= self.gradient*learning_rate
        # Update weights
        self.weights -= np.array(self.inputs*self.gradient*learning_rate)