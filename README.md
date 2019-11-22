# Emerging-Technologies-Project
4th Year emerging technologies project

## Project Objective: 
Create,document,and train a model that recognises hand-written digits using the MNIST dataset, also create a web application that allows a user to draw a digit using their mouse or touch screen device. This drawing should then be then be submitted for recognition to the model you have trained above.

## Python Packages in Implementation: 
- Keras
- Jupyter
- Flask
- numpy
- matplotlib
- Pillow
- CV2
### Other packages used
- re, Base64. Math , io, gzip, sklearn

## How to Run/Use
- Clone/Download repository
- cd into repository in a command line tool (e.g. cmder)
- pip/conda install all necessary packages depending on command line tool you prefer to use. (e.g. conda install keras)
- cd into Web Application directory.
- Run command: python flaskapp.py 
- (Note this is when youll be prompted to install necessary packages if you don't have them installed)
- Navigate to the url link provided by the flask server in a brower (http://127.0.0.1:5000/).
- Draw desired number you wish to predict in the canvas and press the predict button.


## Research
Note: All refernces to websites etc.. are displayed in references section.

1. I started my research by refreshing myself on python as it had been a while since I had looked at some python code. I did this by looking at the docs/tutorials, as well as the videos which were uploaded to moodle e.g. Collatz Revision. I also looked over my project from 3rd year from graph theory which was carried out in python. I viewed the videos on jupyter notebooks and tested those myself to get a feel for how they work. Afterwards, I did some research on deep learning to get a better understanding of it as well as looking into Keras to find out what it was, as our lecturer mentioned we would be using this down the line. I then did some research into areas, such as simple linear regression and Newton's method as these were topics posted on moodle. 

2. I started my research on keras by  watching the videos that were posted on moodle as well as looking and testing the models that were in the jupyter notebook's on moodle e.g. single neuron networks vs multi neuron networks . I also read some of the Keras website to get a better understanding of what the package done to help improve the previous research I had carried out on Keras. I then looked into Tensorflow as this is what Keras runs on top of to carry out the deep learning. I did this by looking at their website seeing how it is used for machine learning and found out it's used by many large corporations/companies e.g. Google, SAP, IBM, Paypal etc. So I knew learning about this would be a useful skill for myself in the future.

3. We were then introduced to the MNIST dataset, a set of 60000 examples of images of numbers 0-9 and 10000 images that are used as a test set. I did some research into this by looking at their website and understanding more about the dataset as well as writing a program in python which could read in the mnist dataset and display some of them, as reading them would be a vital skill needed to complete this project. I also converted the images to an array using numpy and displayed the image in array format on screen using zeros where the pixels were coloured in and dots where they were not as seen on the video which was posted to moodle doing the same but in C programming language. 

4. Now that I was more familiar with both Keras and MNIST and the correlation that I needed between them for the project I wanted to jump more in depth as to how they work in relation to my project and what would be the best thing to use to compliment their features to get an efficient training model. I started by looking at the different optimizers that could be used for training the model. I did this by looking at a website which went indepth into how the learning rates of optimizers work and which would be the best to use for different situations, it talked about optimizers such as Adam, RMSProp and Adagrad etc..., it displayed in graphs the data from their testing and how each of optimizers tested performed in different situations. From this article I decided that adam was the best optimizer to use due to it's stability and it was the also the fastest to learn. I also looked into the keras layers such as Dense, Flatten, MaxPooling2D and Dropout and how they work in a model. I did this using the Keras documentation as well as some other websites which explained them in depth. I also did some research into epochs and batch size and how they differ. I aslo tried to figure out what the best amount of epochs is, but figured out it was different depending on the dataset, which sent me down a trail of searching for the best epochs for the MNIST dataset to find different results, though a concensus between 8-12 seemed to be the most commonly used. I also looked at some of the activation functions used e.g. (softmax, relu, sigmoid) and the differences between them.

5. I viewed the pandas videos on moodle and youtube by the creator of pandas and looked into how it worked with the iris dataset and reading the jupyter notebooks available. I looked at useful the data can be when creating neural networks and how important they would be in industry standard projects and how useful the pandas package would be, I also remembered how easy the graphs with data made it to understand the perfomance of optimizers that I had previously carried out above. 

6. I then did some research on flask by looking at their website and creating a simple flask app that displayed a sample webpage, as this would be needed for the web application needed for the project. I did extra research into Flask as I planned on using it for my other year long project. After I did some research into canvas's in html and how they work as they were needed to allow the user to draw a number and send the canvas as an image. The next step was carrying this out in my actual project and passing it to my python code, I did some research into ajax requests to pass the image in base64 and read a few articles on http to refresh my knowledge on the topic. I also watched the videos on moodle after I had carried these out to ensure that I had done them correctly or in a similar way, which I had. I found the Pillow package through my research and used this to carry out the converion of the base64 image back to an image once sent to the flask application. This image then needed to be converted to greyscale, so I did some research into how I would convert an image from coloured to grey and found that the CV2 package would be best to carry out this task.

7. Through re-reading the MNIST website, I realised I was converting my image to 28x28, which is the size of the MNIST dataset images, but forgot that the MNIST data is stored as a 20x20 image within a 28x28 container. So I did some research and read some MNIST examples which looked about how you would go about this. I also realised that my images where not a high enough threshold and therefore was effecting the results I was getting back and found a solution on the same article that I used to convert to (20x20 within 28x28). 
