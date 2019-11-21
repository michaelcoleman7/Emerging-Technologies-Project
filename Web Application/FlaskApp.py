from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import math
from io import BytesIO
import re, base64
import matplotlib.pyplot as plt
import cv2
from Model import predictImage

app = Flask(__name__)

# Set up app.route decorator so runs webpage() function when user loads http://127.0.0.1:5000/
@app.route('/')
def webpage():
    # Render the HTML page Webpage.html from the templates folder
    return render_template('/Webpage.html')

# Map Route to get image data from canvas
@app.route('/image', methods=['POST'])
def getImage():
    # Convert base64 string to image - Adapted from https://github.com/python-pillow/Pillow/issues/3400

    # Request image from html in base64 format
    image_b64 = request.values['imageBase64']
    base64_data = re.sub('^data:image/.+;base64,', '', image_b64)
    # Decode from base64
    byte_data = base64.b64decode(base64_data) 
    image_data = BytesIO(byte_data)
    # Convert to PIL Image
    img = Image.open(image_data)

    img = img.save("predictImage.png")
    imgcv2 = cv2.imread("predictImage.png")

    # Convert img returned to greyscale
    grayImg = cv2.cvtColor(imgcv2, cv2.COLOR_BGR2GRAY)

    # Added threshold to the image in order to turn coloured image pixels fully white - https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    # (was grey colour previously causing prediction issues)
    (thresh, grayImg) = cv2.threshold(grayImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Remove the rows and columns that are completly black from around the sides of the image, adapted from - https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    while np.sum(grayImg[0]) == 0:
        grayImg = grayImg[1:]

    while np.sum(grayImg[:,0]) == 0:
        grayImg = np.delete(grayImg,0,1)

    while np.sum(grayImg[-1]) == 0:
        grayImg = grayImg[:-1]

    while np.sum(grayImg[:,-1]) == 0:
        grayImg = np.delete(grayImg,-1,1)
    rows,cols = grayImg.shape

    # Refactor in order to fit into a 20x20 image, adapted from - https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        grayImg = cv2.resize(grayImg, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        grayImg = cv2.resize(grayImg, (cols, rows))

    # Resize the image back to a 28x28 image by adding black rows and columns, adapted from - https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    grayImg = np.lib.pad(grayImg,(rowsPadding,colsPadding),'constant')

    # Convert grey image to array for prediction
    grayArray = ~np.array(list(grayImg)).reshape(1, 784).astype(np.uint8) /255


    arrayElementCount = 0
    #Print array out for testing by looping through each element
    for i in grayArray[0]:
        if i == 1:
            # print without going to a new line - adapted from https://www.stechies.com/python-print-without-newline/
            #print . if a one exists in array
           print(".",end="")
        else:
            # Print 0 if array element not = 1
            print("0",end="")
        
        arrayElementCount +=1

        # If the count 28 then go to new line as image is 28x28
        if arrayElementCount == 28:
            print("\n")
            arrayElementCount = 0
    
    prediction = predictImage(grayArray)

    print(prediction)

    # Test image in greyscale
    #cv2.imshow('image',grayImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return str(prediction)

if __name__ == '__main__':
    # Run the application 
    app.run()