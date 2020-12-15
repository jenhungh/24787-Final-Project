import numpy as np
import pandas as pd
from MLP import MLP

# Load data
data = pd.read_csv("ML Final Project Dataset.csv")
dataX = data.iloc[:,8:11]
# print(dataX.values)
# print(dataX.shape)
data_Kp = data.iloc[:,11]
data_Ki = data.iloc[:,12]
# print(data_Kp.values)
# print(data_Kp.shape)

# Create self-testing data
self_testX = np.array([[0.048, 0.115, 0.16], #05 Kp=0.15, Ki=0.1
                       [-0.04, 0.15, 0.17], #11 Kp=0.05, Ki=0.1
                       [0.005, 0.15, 0.13], #16 Kp=0.07, Ki=0.3
                       [-0.014, 0.12, 0.13], #20 Kp=0.07, Ki=1.2
                       [0.64, 0.12, 0.14]]) #25 Kp=0.07, Ki=0.001

# Run the MLP for "Kp"
mlp = MLP(dataX.values, data_Kp.values)
trainX, testX, trainY, testY = mlp.split()
self_test_Kp = mlp.MLP_model(trainX, testX, trainY, testY, self_testX)
print(self_test_Kp)

# Run the MLP for "Ki"
mlp = MLP(dataX.values, data_Ki.values)
trainX, testX, trainY, testY = mlp.split()
self_test_Ki = mlp.MLP_model(trainX, testX, trainY, testY, self_testX)
print(self_test_Ki)