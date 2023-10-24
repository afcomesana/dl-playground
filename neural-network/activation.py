import numpy as np
import math

class Activation:
    """
    Generic class for activation functions for all them to share the apply method.
    This methods intends to simplify the process of compute an operation either it
    is a single value or it has to be applied over an array of values.
    """
    @staticmethod
    def compute(value):
        """
        Mathematical function applied over the specified value.
        Default method is Identity.
        """
        return value
    
    @staticmethod
    def gradient(value):
        return 1
        
    @classmethod
    def apply(cls, inputs):
        if isinstance(inputs, list) or isinstance(inputs, np.ndarray):
            output = map(lambda value: cls.compute(value), inputs)
            return list(output) if isinstance(inputs, list) else np.array(output)
        
        if isinstance(inputs, int) or isinstance(inputs, float):
            return cls.compute(inputs)
        
        raise "Don't know how to process input."

class ReLU(Activation):
    @staticmethod
    def compute(value):
        return max(0, value)
    
    def gradient(value):
        if value <= 0: return 0
        return 1
    
class Sigmoid(Activation):
    @staticmethod
    def compute(value):
        return 1/(1 + pow(math.e,-value))
    
    @staticmethod
    def gradient(value):
        return pow(math.e, -value) / pow(1 + pow(math.e, -value), 2)
    
class Identity(Activation):
    pass