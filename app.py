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


app = Flask(__name__)

@app.route("/runfirst",methods=['POST'])
def runfirst():
    if request.method == "POST":
        name = request.files.get('fileSelect')
        print(name)
        import IPython.display as ipd
        from keras.models import load_model
        model=load_model('mymodel.h5')

        import pandas as pd
        from tqdm import tqdm
        path ='C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/audio/'
        df = pd.read_csv('C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/metadata/UrbanSound8K.csv')
        # df

        #sample = "UrbanSoundog.wav" # single bark
        #x,freq = librosa.load(sample)
        #sr=freq

        #print(x.shape)
        #print(type(x))
        #print(freq)
        #print(type(freq))

        import IPython.display as ipd
                #ipd.Audio(sample)

        import matplotlib.pyplot as plt
        import librosa.display
        # plt.figure(figsize=(10,3))
        # plt.title("Single Bark Wave Plot")
        #librosa.display.waveshow(x,sr=freq)

        #sample2 = "UrbanSound2.wav"
        #x,sr = librosa.load(sample2)
        #ipd.Audio(x,rate=sr)


        # plt.figure(figsize=(10,3))
        # plt.title("Multi Bark Wave Plot")
        #librosa.display.waveshow(x,sr=freq)

        #X=librosa.stft(x) #stft -> Short-time Fourier transform
        #X_db=librosa.amplitude_to_db(abs(X)) #Translation from amplitude to desibel(db) value
        # plt.figure(figsize=(20,8))
        #librosa.display.specshow(X_db, sr=sr,x_axis="time",y_axis="hz")
        # plt.title("Multi Bark Sound Spectogram")
        # plt.colorbar()

        #sample3 = "UrbanSoundbird.wav"
        #x,sr = librosa.load(sample3)
        #ipd.Audio(x,rate=sr) 

        #data_h, data_p = librosa.effects.hpss(x)
        #spec_h = librosa.feature.melspectrogram(data_h, sr=sr)
        #spec_p = librosa.feature.melspectrogram(data_p, sr=sr)
        #db_spec_h = librosa.power_to_db(spec_h,ref=np.max)
        #db_spec_p = librosa.power_to_db(spec_p,ref=np.max)

        #ipd.Audio(data_h,rate=sr)

        #librosa.display.specshow(db_spec_h,y_axis='mel', x_axis='s', sr=sr)
        # plt.title("Harmonic Mel Spectogram")
        # plt.colorbar()


        #ipd.Audio(data_p,rate=sr) 

        #librosa.display.specshow(db_spec_p,y_axis='mel', x_axis='s', sr=sr)
        # plt.title("Percuisive Mel Spectogram")
        # plt.colorbar()


        #mfcc=librosa.feature.mfcc(x,sr=sr)
        #print("shape of mfcc:" ,mfcc.shape)


        # plt.figure(figsize=(15,6))
        
        #librosa.display.specshow(mfcc,x_axis="s")
        # plt.title("Mel-Frequency Cepstral Coefficients")
        # plt.colorbar()

        #zero_crossing=librosa.zero_crossings(x)
        #print("Type of Zero Crossing Rate",type(zero_crossing))
        #print(zero_crossing, " --> it contains booleans")
        #print("Total Number of Zero Crossing is: ",sum(zero_crossing))

        # plt.figure(figsize=(15,5))
        # plt.title("Zero Crossing Rate")
        # plt.plot(x[4000:5100])
        # plt.grid()

        #y, sr = librosa.load(sample3)
        #chroma=librosa.feature.chroma_stft(y=y, sr=sr)
        # plt.figure(figsize=(10, 4))
        #librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        # plt.colorbar()
        # plt.title('Chromagram')
        # plt.tight_layout()


        metadata=pd.read_csv('C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/metadata/UrbanSound8K.csv')
        metadata.head(10)

        metadata['class'].value_counts()

        #audio_file_path='UrbanSounddrill.wav'
        #librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)


        #mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)
        #print(mfccs.shape)
        #print(librosa_audio_data)

        ### Lets plot the librosa audio data
        # import matplotlib.pyplot as plt
        # Original audio with 1 channel 
        # plt.figure(figsize=(12, 4))
        # plt.plot(librosa_audio_data)

        #mfccs






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


        def print_prediction(file_name):
            pred_fea = extract_feature(file_name) 
            pred_vector = np.argmax(model.predict(pred_fea), axis=-1)
            print('pred_vector',pred_vector)
            pred_class = to_categorical(le.fit_transform(pred_vector))
            
            #pred_class = le.inverse_transform(pred_vector)
            print(pred_class)
            return pred_class[0]


            #file_name ='C:/Users/ACER/OneDrive/Desktop/APP/UrbanSound8K/audio/fold9/7975-3-0-0.wav'
            file_name=name
            output=print_prediction(file_name)
            print(output)
            return render_template('index.html',data=output)
        #ipd.Audio(file_name)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)