from Layer import Layer
from Model import Model

from activation import ReLU, Activation

nn = Model()

nn.add(Layer(5))
nn.add(Layer(5, ReLU))

print(nn)