from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import math
from io import BytesIO
import re, base64
import matplotlib.pyplot as plt
import cv2

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


    return ''

if __name__ == '__main__':
    # Run the application 
    app.run()