from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import pickle
from keras.preprocessing import image
import numpy as np
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

model = pickle.load(open('model2.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    img_pred = image.load_img(image_path,target_size=(250,750))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred,axis=0)
    load_model = pickle.load(open("model2.pkl",'rb'))
    result=load_model.predict(img_pred)
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    if result[0][0]>result[0][1]:
        prediction="Bacterial leaf blight"
    else:
        prediction="Brown spot"

    return render_template('index.html',image_path=image_path,prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)