import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np 
import pylab
import os
import tensorflow as tf
import onnxruntime

diag_model_path = "respiratory_diseases.onnx"
cw_model_path = "crack_wheezles.onnx"
cw_dict = {'cracks' : 0, 'wheezles': 0}

def slice(sig, n):
    for i in range(0, len(sig), n):
        segments = yield sig[i:i + n]
    return segments


def predictResp(audio_path):
    out = []
    d_sess = onnxruntime.InferenceSession(diag_model_path)
    cw_sess = onnxruntime.InferenceSession(cw_model_path)
    d_input_name = d_sess.get_inputs()[0].name
    cw_input_name = cw_sess.get_inputs()[0].name
    sig, s = librosa.load(audio_path)
    
    if (len(sig) > 5*s):
        slices = slice(sig, 5*s)   
    else:
        slices = [sig]

    for i, segment in enumerate(slices):
        padded = librosa.util.pad_center(segment, 7*s)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        S = librosa.feature.melspectrogram(y=padded, sr=s)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig('temp_pic.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        img = tf.keras.preprocessing.image.load_img('temp_pic.jpg', target_size=(224, 224))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = input_arr[None, :, :, :]
        res = d_sess.run(None, {d_input_name: input_arr})
        cw = cw_sess.run(None, {cw_input_name: input_arr})
        os.remove('temp_pic.jpg')
        prob = res[0]
        prob1 = cw[0]
        out.append(prob.argmax(axis = 1)[0])
        cw_var = prob1.argmax(axis =1)[0]
        if cw_var == 0: 
            cw_dict['cracks'] += 1
        elif cw_var == 1:
            cw_dict['wheezles'] += 1
        elif cw_var == 2:
            cw_dict['cracks'] += 1
            cw_dict['wheezles'] += 1
        
    diag = max(set(out), key=out.count)
    if diag==0:
        return 'Healthy'
    elif diag == 1:
        return 'URTI'
    elif diag == 2:
        return 'Asthma'
    elif diag == 3:
        return 'COPD'
    elif diag == 4:
        return 'LRTI'
    elif diag == 5:
        return 'Bronchietasis'
    elif diag == 6:
        return 'Pneumonia'
    elif diag == 7:
        return 'Bronchiolitis'


def cracks_wheezles() :
    return cw_dict