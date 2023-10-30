from model.Layer import InputLayer, DenseLayer
from model.Model import Model
from model.loss_functions import CrossEntropy, MeanSquaredError
from model.activation import ReLU, Sigmoid


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import center, standarize

# X_train_data = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# X_train_target = np.array([0, 1, 1, 0])


df = pd.read_csv("diabetes.csv")
# Remove all rows with potentially missing values:
for col in ["Pregnancies","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]:
    df = df.loc[df[col] != 0, :]

X_train = df.loc[:300, :]
X_test  = df.loc[300:,:]

X_train_data   = np.array(X_train.loc[:,X_train.columns != "Outcome"])
X_train_target = np.array(X_train["Outcome"])
X_train_data   = standarize(X_train_data)

rows, cols = X_train_data.shape

for col in range(cols):
    print("mean:", np.mean(X_train_data[:, col]), " - std:", np.std(X_train_data[:, col]))

X_test_data   = np.array(X_test.loc[:,X_test.columns != "Outcome"])
X_test_target = np.array(X_test["Outcome"])


model = Model()
model.add(InputLayer(cols))
model.add(DenseLayer(5, ReLU))
model.add(DenseLayer(5, ReLU))
model.add(DenseLayer(4, ReLU))
model.add(DenseLayer(4, ReLU))
model.add(DenseLayer(3, ReLU))
model.add(DenseLayer(1, Sigmoid))


# print("Pre-training:")
# for data, target in zip(X_train_data, X_train_target):
#     print(target, model.predict(data))
    

# for i in range(2000):
#     for data, target in zip(X_train_data, X_train_target):
#         model.train(data, target, CrossEntropy)

# print()
# print("Post-training:")
# for data, target in zip(X_train_data, X_train_target):
#     print(target, model.predict(data))


for data, target in zip(X_train_data, X_train_target):
    model.train(data, target, CrossEntropy)


# plt.figure(figsize=(10, 10))
# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 1.1)

# color = {0: "ro",1: "go"}

# for sample, target in zip(X_train_data, X_train_target):
#     plt.plot(*sample, color[target], markersize=20)
    
# x_range = np.arange(-0.1, 1.1, 0.01)
# y_range = np.arange(-0.1, 1.1, 0.01)

# # Default indexing returns the first matrix having each row as the original array
# # and the second matrix where each i-th row is the i-th element repeated as many 
# # times as elements has the first array

# # Indexing "ij" returns the opposite: in the first matrix, each row consist of the
# # i-th element repeated as many times as elements there are in the second array
# xx, yy = np.meshgrid(x_range, y_range, indexing="ij")
# print(model.predict(np.array([0, 1])))
# Z = np.array([[model.predict([x, y]) for x in x_range] for y in y_range])
# Z = Z.round()

# plt.contourf(xx, yy, Z, colors=["red", "blue", "green", "yellow"], alpha=0.4)
# plt.show()



# df = pd.read_csv("diabetes.csv")

# # Remove all rows with potentially missing values:
# for col in ["Pregnancies","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]:
#     df = df.loc[df[col] != 0, :]
    
# X_train = df.loc[:300, :]
# X_test  = df.loc[300:,:]

# X_train_data   = np.array(X_train.loc[:,X_train.columns != "Outcome"])
# X_train_target = np.array(X_train["Outcome"])

# X_test_data   = np.array(X_test.loc[:,X_test.columns != "Outcome"])
# X_test_target = np.array(X_test["Outcome"])

# rows, cols = X_train_data.shape