import pickle
import gzip
from tensorflow.keras.models import load_model
import numpy as np

# 載入模型
with gzip.open('app/model/emotion_test.pgz', 'r') as f:
    emotion_Vlance = pickle.load(f)
with gzip.open('app/model/emotion_Arousal.pgz', 'r') as f:
    emotion_Arousal = pickle.load(f)

Cry_model = load_model('app/model/cry_demo.h5')

# 將模型預測寫成一個 function 
def Vlance_predict(input):
    pred=emotion_Vlance.predict(input)[0]
    print(pred)
    return pred
def Arousal_predict(input):
    pred=emotion_Arousal.predict(input)[0]
    print(pred)
    return pred
def Cry_predict(input):
    Input = []
    Input.append(input)
    Input = np.array(Input)

    return Cry_model.predict(Input)*100
