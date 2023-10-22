import random
import numpy as np
import random

from activation import ReLU, Activation

class Unit:
    """
    - inputs: should be an list or numpy array of integers or float numbers
    - activation: if it is None that means that the activation is the identity function.
    - weights: are not usually passed as arguments when initializing the Unit, so the
    most of the time will be randomly executed to small positive numbers (0,1)
    - bias: same as weights
    """
    
    def __init__(self, activation = None, inputs = None, weights = None, bias = None):
        self.activation = activation
        self.inputs     = inputs
        self.weights    = weights
        self.bias       = bias
        
        if not self.activation is None and not isinstance(self.activation, Activation):
            raise Exception("Activation function not valid.")

        # if weights is None:
        #     self.weights = np.random.rand(len(self.inputs))
            
        # if bias is None:
        #     self.bias = np.random.rand()
            
    
    def assert_inputs_type(self):
        """
        Check input for the unit has the proper type.
        If the type has not a numerical value raise an error
        """
        
        if isinstance(self.inputs, list): return True
        if isinstance(self.inputs, np.ndarray): return True
        
        if isinstance(self.inputs, float):
            self.inputs = np.array(self.inputs)
            return True
        
        if isinstance(self.inputs, int):
            self.inputs = np.array(self.inputs)
            return True
        
        raise "Input type is not valid."
        
    def compute_output(self):
        """
        Compute the output of the neuron.
        1. Performs dot product over inputs and weights.
        2. Add the bias.
        3. Apply activation function.
        """
        
        self.output = np.dot(self.inputs, self.weights) + self.bias
        
        if not self.activation is None:
            self.output = self.activation(self.output).apply()
        
        return self.output
    