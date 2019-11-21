# Imports
import keras as kr
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import gzip
# For encoding categorical variables.
import sklearn.preprocessing as pre

def createModel():
    (train_img, train_lbl), (test_img, test_lbl) = mnist.load_data()
        
    # Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
    model.add(kr.layers.Dense(units=400, activation='relu'))
    # Add a three neuron output layer.
    model.add(kr.layers.Dense(units=10, activation='softmax'))

    # Build the graph.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    # Initialize train images and labels
    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    #Set up input value for training
    inputs = train_img.reshape(60000, 784)

    # Set up encoder and output values for training
    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    model.fit(inputs, outputs, epochs=10, batch_size=100)
        
    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

    (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()

    # Test the model score and accuracy
    score = model.evaluate(inputs, outputs, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save the entire model to a HDF5 file. - https://jovianlin.io/saving-loading-keras-models/
    # The '.h5' extension indicates that the model shuold be saved to HDF5.
    model.save('model.h5')
    
    return model

def predictImage(predictionImage):
    # Try/catch equivelant in python, adapted from https://www.pythonforbeginners.com/error-handling/python-try-and-except
    try:
        # Try to load model saved - https://jovianlin.io/saving-loading-keras-models/
        model = load_model("model.h5")
    # When no model exists call createModel() function to create new model
    except:
        # Call the create model function
        model = createModel()
    
    # Make a prediction on what the number drawn is
    predict = model.predict(predictionImage)
    
    # returns the largest value in the label
    return predict.argmax()





