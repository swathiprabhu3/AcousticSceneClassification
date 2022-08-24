#import os
#import tensorflow as tf
#import numpy as np
#from tensorflow import keras
#from skimage import io
#from tensorflow.keras.preprocessing import image
#import os


# Flask utils
#from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask import *

import json

# import luyin as ly
import librosa
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import librosa.display

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation , Dropout

import IPython.display as ipd


import matplotlib.pyplot as plt
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('cnnmodel.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

le = LabelEncoder()
            



   



    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

    #print("Number of training samples = ", x_train.shape[0])
    #print("Number of testing samples = ",x_test.shape[0])
    #num_labels = y.shape[1]

def extract_feature(file_name):
                audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
                fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
                scaled = np.mean(fea.T,axis=0)
                return np.array([scaled])

# function to predict the feature
def print_prediction(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # get the feature
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    # scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # array of features
    prediction_feature = np.array([feature_scaled])
    # reshaping the features
    final_prediction_feature = prediction_feature.reshape(
        prediction_feature.shape[0], 10,5,1
    )
    # get the id of label using argmax
    predicted_vector = np.argmax(model.predict(final_prediction_feature), axis=-1)
    # get the class label from class id
    predicted_class = to_categorical(le.fit_transform(predicted_vector))
    # display the result
    return predicted_vector[0]

def print_prediction1(file_name):
                pred_fea = extract_feature(file_name) 
                pred_fea=np.array([pred_fea],int)
                print('pred_fea',pred_fea.shape)
                pred_fea=pred_fea.reshape(pred_fea,10,5,1)  
                print('pred_fea',pred_fea.shape)              
                pred_vector = np.argmax(model.predict(pred_fea), axis=-1)
                print('pred_vector',pred_vector)
                pred_class = to_categorical(le.fit_transform(pred_vector))
                
                #pred_class = le.inverse_transform(pred_vector)
                print(pred_class)
                return pred_vector


    #fold5/159701-6-5-0.wav
    #fold7/107357-8-1-11.wav
    # Python program to convert a list to string
        
    # Function to convert  
   


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print('file_path',file_path)
        print(f)
        print("basepath",basepath)
        metadata=pd.read_csv('C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/metadata/UrbanSound8K.csv')
        metadata.head(10)

        metadata['class'].value_counts()
        #le = LabelEncoder()
        # Make predicti
        
        #f ='C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/audio/fold5/159701-6-5-0.wav'
        preds = print_prediction(file_path)
        print('preds',preds)
        #output=preds[0]
        #print(output)

        # x = x.reshape([64, 64]);
        label = ['air conditioner','car horn','children playing','dog bark','drilling','engine idling','gunshot','jack hammer','siren',
        'street music']
        a = preds
        print('Prediction:', label[a])
        result=label[a]
        return result
    return None


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    #app.run()
