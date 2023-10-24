import numpy as np

from Layer import Layer, InputLayer

class Model:
    def __init__(self, layers=[]):
        self.layers = []
        [self.add(layer) for layer in layers]
        
    def add(self, layer):
        """
        If there is any layer in the model, we have to connect the new layer with the last one.
        
        - layer: has to be an instance of the class Layer.
        """
        
        if not isinstance(layer, Layer):
            raise Exception("Can not add an non-Layer object to a model.")
        
        if len(self.layers) > 0:
            layer.connect(self.layers[-1])

        self.layers += [layer]
        
    def predict(self, inputs):
        
        output = np.array(inputs)
        
        for layer in self.layers:
            output = layer.compute(output)
            
        return np.array(output)
    
    def train(self, inputs, target, loss, learning_rate=0.1):
        # Batch training:
        if inputs.ndim > 1:
            pass
        
        y_hat = self.predict(inputs)
        self.backpropagation(np.matrix(loss(target, y_hat).gradient))
        # print("Target:", target, "\nPrediction:", y_hat, "\nLoss:", loss(target, y_hat).loss, "\nGradient:",loss(target, y_hat).loss)
        # print()
        
        [unit.update_parameters(learning_rate=learning_rate) for layer in self.layers for unit in layer.units]
        
    def backpropagation(self, gradient):
        # Iterate over the layers of the model in reversed order (do not iterate over the input layer)
        for layer in list(reversed(self.layers))[:-1]:
            layer_gradient = []
            for index, unit in enumerate(layer.units):
                unit.gradient = gradient[:, index]
                unit.gradient = np.sum(unit.gradient * unit.activation.gradient(unit.linear_sum))
                layer_gradient += [unit.gradient*unit.weights]
                
            gradient = np.matrix(layer_gradient)
            

    def info(self):
        
        print("\n\nTotal layers in the model:", len(self.layers), "\n")
        
        print(" ==> ".join(["%s (%s units)" % (type(layer).__name__, layer.units_amount) for layer in self.layers]))
        
        for index, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer): continue
            print()
            print(index, type(layer).__name__, ":")
            
            for index, unit in enumerate(layer.units):
                print()
                print(index+1, "Unit(", type(unit.activation).__name__,")")
                print("Weights:", unit.weights)
                print("Bias", unit.bias)
            
        print("\n")