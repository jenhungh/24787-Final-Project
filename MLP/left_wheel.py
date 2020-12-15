import numpy as np
import pandas as pd
from MLP import MLP

# Load data
data = pd.read_csv("ML Final Project Dataset.csv")
dataX = data.iloc[:,2:5]
# print(dataX.values)
# print(dataX.shape)
data_Kp = data.iloc[:,5]
data_Ki = data.iloc[:,6]
# print(data_Kp.values)
# print(data_Kp.shape)

# Create self-testing data
self_testX = np.array([[0.05, 0.115, 0.13], #05 Kp=0.2, Ki=0.2
                       [-0.03, 0.17, 0.2], #11 Kp=0.05, Ki=0.1
                       [-0.02, 0.14, 0.15], #16 Kp=0.07, Ki=0.3
                       [0.006, 0.13, 0.12], #20 Kp=0.07, Ki=1.2
                       [0.56, 0.16, 0.17]]) #25 Kp=0.07, Ki=0.001

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