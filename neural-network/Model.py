from Layer import Layer

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
        output = inputs
        for layer in self.layers:
            output = layer.compute(output)
            
        return output