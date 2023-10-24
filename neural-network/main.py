from Layer import InputLayer, DenseLayer
from Model import Model
from loss_functions import CrossEntropy, MeanSquaredError

from activation import ReLU, Activation, Sigmoid
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = {
    "inputs": [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ],
    "targets": [0, 1, 1, 0]
}

X_train = np.round(np.random.rand(100, 2))
X_train = np.column_stack((X_train, list(map(lambda sample: int(sample[0])^int(sample[1]), X_train))))

batch = X_train
np.random.shuffle(batch)
batch = batch[:10]

model = Model()
model.add(InputLayer(2))
model.add(DenseLayer(4, Sigmoid))
model.add(DenseLayer(1, Sigmoid))

model.train(batch[0:5, :-1], batch[0, -1], CrossEntropy)

# print("Predictions pre-training:")
# for sample, target in zip(dataset["inputs"], dataset["targets"]):
#     y_hat = model.predict(sample)
#     print("Prediction:", y_hat, "\nTarget:", target)
#     print()
    

# for i in range(1000):
#     for sample, target in zip(dataset["inputs"], dataset["targets"]):
#         model.train(sample, target, CrossEntropy)

# print("Predictions post-training:")
# for sample, target in zip(dataset["inputs"], dataset["targets"]):
#     y_hat = model.predict(sample)
#     print("Prediction:", y_hat, "\nTarget:", target)
#     print()

# # print("Target:",dataset["targets"][0], "\nOutput:", y_hat,"\nCalculated loss:",ce.loss,"\nGradient:", ce.gradient)

# plt.figure(figsize=(10, 10))
# plt.axis("scaled")
# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 1.1)

# color = {
#     0: "ro",
#     1: "go"
# }

# for sample, target in zip(dataset["inputs"], dataset["targets"]):
#     plt.plot(sample[0], sample[1], color[target], markersize=20)
    
# x_range = np.arange(-0.1, 1.1, 0.01)
# y_range = np.arange(-0.1, 1.1, 0.01)

# xx, yy = np.meshgrid(x_range, y_range, indexing="ij")
# Z = np.array([[model.predict([x, y]) for x in x_range] for y in y_range])

# plt.contourf(xx, yy, Z, levels=2, colors=["red", "green"], alpha=0.4)
# plt.show()