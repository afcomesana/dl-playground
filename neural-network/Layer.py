import numpy as np

from activation import Activation
from Unit import Unit

class Layer:
    def __init__(self, units_amount, activation = None):
        self.units_amount = units_amount
        self.activation = activation

        if not isinstance(self.units_amount, int):
            raise Exception("Units amounts must be an intenger.")
        
        if not self.activation is None:
            self.activation = self.activation()

        if self.activation is None or isinstance(self.activation, Activation):
            self.activation = [self.activation]*units_amount

        
        self.units = [Unit(activation=act) for act in self.activation]