import numpy as np
import re

from .Layer import Layer, InputLayer
from .activation import Sigmoid
from .loss_functions import CrossEntropy

class Model:
    def __init__(self, layers=[]):
        self.min_loss = 1e-07
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
    
    def get_digits_amount(value):
        value = str(value)
        digits = re.search(r"e(\+|\-)[0-9]+", value)
        if digits is None:
            return len(str(int(float(value))))
        
        digits = digits.group(0)
        digits = int(digits[1:])
        
        return digits
    
    def train(self, inputs, target, loss_class, learning_rate=0.1, verbose=False):
        # TODO: Generalize special case for more-than-one-unit output layers

        shortcut = False
        
        output_activations = [unit.activation for unit in self.layers[-1].units]
        
        # Special case:
        # - Loss function is cross entropy
        # - Last layer activation function is Sigmoid
        if issubclass(loss_class, CrossEntropy) and all(issubclass(activation, Sigmoid) for activation in output_activations):
            
            shortcut = True
            
            y_hat = inputs
            for index, layer in enumerate(self.layers):
                y_hat = layer.compute(y_hat, apply_activation=len(self.layers) != index + 1)

        else:
            y_hat = self.predict(inputs)

        
        if len(output_activations) == 1:
            output_activations = output_activations[0]
            
        loss = loss_class(target, y_hat, output_activations)
        gradient = loss.gradient
        loss     = loss.loss
        
        if loss < self.min_loss: return
        
        if verbose:
            print("\n=======================================================\n")
            print(inputs, y_hat)
            print(loss, gradient)
            self.info()
            
        self.backpropagation(np.matrix(gradient), shortcut=shortcut)
        
        [unit.update_parameters(learning_rate=learning_rate) for layer in self.layers for unit in layer.units]
        if verbose: self.info()
        
    def backpropagation(self, gradient, shortcut=False):
        """
        Naïve implementation of the backpropagation algorithm.
        
        Parameters:
        - gradient: gradient of the loss function with respect to the output of the model.
        - shortcut: in case a special derivative calculation has been made, this parameter must
          be set to True, in order to not compute again the derivative of the activation function
          in the first layer (last layer going forward)
          
        """
        
        # Iterate over the layers of the model in reversed order (do not iterate over the input layer)
        for layer_index, layer in enumerate(list(reversed(self.layers))[:-1]):
            
            layer_gradient = []
            
            for unit_index, unit in enumerate(layer.units):
                # Incoming gradient
                unit.gradient = gradient[:, unit_index]
                
                if not shortcut or layer_index > 0:
                    # Gradient of the incoming gradient with respect to the activation function of the unit
                    unit.gradient = unit.gradient * unit.activation.gradient(unit.linear_sum)
                
                unit.gradient = np.sum(unit.gradient)
                
                # Gradient to pass on to the next layer (previous layer actually since we are going backwards):
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