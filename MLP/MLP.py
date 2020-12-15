# Import package
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

class MLP:

    # Initilaize
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY

    # Split the data into training set and testing set 
    def split(self):
        trainX, testX, trainY, testY = train_test_split(self.dataX, self.dataY, test_size=0.1, random_state=0, shuffle=True)
        return trainX, testX, trainY, testY
    
    # Create the MLP model
    def MLP_model(self, trainX, testX, trainY, testY, self_testX):
        # Build the model
        model = Sequential()

        # Set up the first layer and the input size
        model.add(Dense(32,input_shape=(3,)))

        # Add layers to the model 
        model.add(Dense(64, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['mean_squared_error'])
        print(model.summary())

        # Set up training parameters
        batch_size = 128
        epochs = 1000

        # train network by calling fit function
        model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY))

        # Predict the testing data
        self_testY = model.predict(self_testX)
        
        return self_testY