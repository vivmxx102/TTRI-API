#.-*-coding: UTF-8 -*-

import app as app
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def Trim(Signal,Sr,Len = 5):
    Signal = np.array(Signal[:(Len*Sr)])
    return Signal

def Pad(Signal,Len,Sr = 22050):
    if len(Signal) < (Len*Sr):
      Pad_Size = (Len*Sr)-len(Signal)
      Padded = np.pad(Signal, (0,Pad_Size ), 'constant')
      return Padded
    return Signal

def Save_MFCC(Signal,Sr,n_mfcc=13,n_fft=2048,hop_length=512):
    MFCC = librosa.feature.mfcc(Signal,sr=Sr,n_fft=n_fft,hop_length = hop_length,n_mfcc=n_mfcc)
    return MFCC/303.0



@app.route('/emotion_predict',methods=['POST'])
def postInput():
    insertValues = request.get_json()
    data = insertValues['ECG']
    input = np.array(data)
    working_data, measures = hp.process(input, sample_rate=256.0)
    
    bpm = np.log1p(measures['bpm'])
    ibi = np.log1p(measures['ibi'])
    sdnn = np.log1p(measures['sdnn'])
    sdsd = np.log1p(measures['sdsd'])
    rmssd = np.log1p(measures['rmssd'])
    pnn20 = np.log1p(measures['pnn20'])
    hr_mad = np.log1p(measures['hr_mad'])
    sd1 = np.log1p(measures['sd1'])
    s = np.log1p(measures['s'])
    sd1sd2 = np.log1p(measures['sd1/sd2'])
    breathingrate = np.log1p(measures['breathingrate'])
    input = np.array([[bpm,ibi,sdnn,sdsd,rmssd,pnn20,hr_mad,sd1,s,sd1sd2,breathingrate]])
    print(input)

    Vlance_result = app.Vlance_predict(input)
    Arousal_result = app.Arousal_predict(input)
    if Vlance_result > 0.0 :
        Vlance = "HV"
    else:
        Vlance = "LV"

    if Arousal_result > 0.0 :
        Arousal = "HA"
    else:
        Arousal = "LA"
    print(str(Vlance)+str(Arousal))
    return jsonify({'return': 'Emotion：'+str(Vlance)+str(Arousal)})

@app.route('/Cry_predict',methods=['POST'])
def postInput_cry():
    Sr = 22050
    insertValues = request.get_json()
    data = insertValues['Audio']
    input = np.array(data)
    Trimed = (Trim(Pad(input,5,Sr),Sr)) 
    MFCC = Save_MFCC(Trimed,Sr)
    Audio_predict = app.Cry_predict(MFCC)
    predict = np.where(Audio_predict[0]==max(Audio_predict[0]))
    if predict[0][0] == 0:
        return jsonify({'return': 'Audio_predict：Baby_Pain'})
    elif predict[0][0] == 1:
        return jsonify({'return': 'Audio_predict：Baby_Uncomfortable'})
    else:
        return jsonify({'return': 'Audio_predict：Audio_Environment'})
  

