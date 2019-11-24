# Imports

# API used for building the neural network
import keras as kr
# Used for building the model
from keras.layers import Dense, Dropout, Flatten
# Used for initialization of test and training images/labels (Conversion to numpy arrays)
import numpy as np
# Used for unzipping data sets
import gzip
# For encoding categorical variables.
import sklearn.preprocessing as pre

# Function to build the model
def createModel():
    
    # Read in all MNIST Data from dataset in folder /MNIST Data Files , for training and test images
    with gzip.open('MNIST Data Files/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('MNIST Data Files/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    with gzip.open('MNIST Data Files/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('MNIST Data Files/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
        
    # Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
    model.add(kr.layers.Dense(units=400, activation='relu'))
    # Add a 10 neuron output layer, representing 0-9
    model.add(kr.layers.Dense(units=10, activation='softmax'))

    # Build the graph.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    # Initialize train images and labels as numpy arrays
    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    # Set up input value for training
    inputs = train_img.reshape(60000, 784)

    # Set up encoder and output values for training
    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    # Train the model for a certain number of Epochs
    model.fit(inputs, outputs, epochs=10, batch_size=100)
        
    # Initialize test images and labels as numpy arrays
    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

    # Calculate the amount of correct results by comparing test images to the labels
    (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()

    # Test the model accuracy - https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    scores = model.evaluate(inputs, outputs, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Save the entire model to a HDF5 file. - https://jovianlin.io/saving-loading-keras-models/
    # The '.h5' extension indicates that the model shuold be saved to HDF5.
    model.save('model.h5')
    
    # Return the model
    return model

# Function to predict a specified image
def predictImage(predictionImage):
    # Try/catch equivelant in python, adapted from https://www.pythonforbeginners.com/error-handling/python-try-and-except
    try:
        # Try to load model saved - https://jovianlin.io/saving-loading-keras-models/
        model = kr.models.load_model("model.h5")
    # When no model exists call createModel() function to create new model
    except:
        # Call the create model function
        model = createModel()
    
    # Make a prediction on what the number drawn is
    predict = model.predict(predictionImage)
    
    # returns the largest value in the label
    return predict.argmax()





