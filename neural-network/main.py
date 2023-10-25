from Layer import InputLayer, DenseLayer
from Model import Model
from loss_functions import CrossEntropy, MeanSquaredError

from activation import ReLU, Activation, Sigmoid
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import inspect
df = pd.read_csv("diabetes.csv")

# Remove all rows with potentially missing values:
for col in ["Pregnancies","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]:
    df = df.loc[df[col] != 0, :]
    
X_train = df.loc[:300, :]
X_test  = df.loc[300:,:]

X_train_data   = np.array(X_train.loc[:,X_train.columns != "Outcome"])
X_train_target = np.array(X_train["Outcome"])

X_test_data   = np.array(X_test.loc[:,X_test.columns != "Outcome"])
X_test_target = np.array(X_test["Outcome"])

rows, cols = X_train_data.shape

model = Model()
model.add(InputLayer(cols))
model.add(DenseLayer(4, ReLU))
model.add(DenseLayer(1, Sigmoid))

model.train(X_train_data[0,:], X_train_target[0], CrossEntropy)

# for i in range(10):
#     print(model.predict(X_train_data[0, :]))
#     model.train(X_train_data[0, :], X_train_target[0], CrossEntropy)
    

# for epoch in range(100):
#     for row in range(rows):
#         model.train(X_train_data[row, :], X_train_target[row], CrossEntropy)
        
# for row in range(len(X_test_data)):
#     y_hat = model.predict(X_test_data[row, :])
#     print("Predicted: %s (%s)" % (y_hat, X_test_target[row]))



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