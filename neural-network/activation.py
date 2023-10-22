import numpy as np

class Activation:
    def __init__(self, inputs = None):
        self.inputs = inputs
        
    def compute(self):
        return self.inputs
        
    def apply(self):
        if isinstance(self.inputs, list) or isinstance(self.inputs, np.ndarray):
            output = map(lambda value: self.compute(value), self.inputs)
            
            return list(output) if isinstance(self.inputs, list) else np.array(output)
        
        if isinstance(self.inputs, int) or isinstance(self.inputs, float):
            return self.compute(self.inputs)
        
        raise "Don't know how to process input."

class ReLU(Activation):
    def compute(self, value):
        return max(0, value)
    