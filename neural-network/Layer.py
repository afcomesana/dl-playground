import numpy as np

from activation import Activation, Identity
from Unit import Unit

class Layer:
    def __init__(self, units_amount, activation = Identity):
        self.units_amount = units_amount
        self.activation = activation

        if not isinstance(self.units_amount, int):
            raise Exception("Units amounts must be an intenger.")

        if issubclass(self.activation, Activation):
            self.activation = [self.activation]*units_amount

        self.units = [Unit(activation=act) for act in self.activation]
        
        
class InputLayer(Layer):
        
    def compute(self, inputs, apply_activation=True, verbose=False):
        output = [unit.compute(value) for value, unit in zip(inputs, self.units)]
        return np.array(output) if isinstance(inputs, np.ndarray) else output

class DenseLayer(Layer):
    def compute(self, inputs, apply_activation=True, verbose=False):
        output = [unit.compute(inputs, apply_activation=apply_activation, verbose=verbose) for unit in self.units]
        
        if len(output) == 1: return output[0]
        
        return np.array(output) if isinstance(inputs, np.ndarray) else output
        
    def connect(self, layer):
        [unit.initialize_parameters(layer.units_amount) for unit in self.units]