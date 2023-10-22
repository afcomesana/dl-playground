class Model:
    def __init__(self, layers=[]):
        self.layers = layers
        
    def add(self, layer):
        self.layers += [layer]