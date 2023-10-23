from Layer import InputLayer, DenseLayer
from Model import Model
from loss_functions import CrossEntropy

from activation import ReLU, Activation, LogisticSigmoid
import numpy as np

dataset = {
    "inputs": [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ],
    "targets": [0, 1, 1, 0]
}

model = Model()
model.add(InputLayer(2))
model.add(DenseLayer(2, ReLU))
model.add(DenseLayer(1, LogisticSigmoid))

y_hat = model.predict(dataset["inputs"][0])

ce = CrossEntropy(dataset["targets"][0], y_hat)
print("Target:",dataset["targets"][0], "\nOutput:", y_hat,"\nCalculated loss:",ce.loss,"\nGradient:", ce.gradient)