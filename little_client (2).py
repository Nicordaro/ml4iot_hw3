import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import zlib
import base64
import requests
import json
from preprocessing import preprocess

url = 'http://localhost:8080/big'

zip_path = tf.keras.utils.get_file(
                origin = 'http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
                fname = 'mini_speech_commands.zip',
                extract = True,
                cache_dir = '.',
                cache_subdir = 'data')
    
data_dir = os.path.join('.','data', 'mini_speech_commands')

def filenames(path):
    with open(path, "rb") as fp:
        arr = np.array(fp.read().splitlines())
    return arr

def success_checker(score_margin):
    if score_margin <= 32:
        return True
    return False
    
test_files = filenames("kws_test_split.txt")
LABELS = np.array(['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go'])

sampling_rate = 16000
frame_length = 256
frame_step = 128

with open('small_model_stft.tflite.zlib', 'rb') as fp:
    model_zip = zlib.decompress(fp.read())
    interpreter = tf.lite.Interpreter(model_content=model_zip)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details()

predictions = []
true_labels = []
communication_cost = 0

for file_path in test_files:
    audio_binary = tf.io.read_file(file_path)
    # stft preprocessing
    x = preprocess(audio_binary, sampling_rate, frame_length, frame_step)
    
    # retrieve the true label
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]  
    y_true = tf.argmax(label == LABELS)
    true_labels.append(y_true)
    
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    
    # prediction of the little model 
    logits = interpreter.get_tensor(output_details[0]["index"])
    probs = tf.nn.softmax(logits)#.numpy()[0]*100
    prob = tf.reduce_max(probs).numpy()*100
    y_pred = tf.argmax(probs, 1).numpy()[0]

    probs = probs.numpy()[0]*100
    probs[::-1].sort()
    two_biggest = probs[:2]
    
    #difference between the top 2 biggest probabilities to see the level of uncertainity in the prediction
    score_margin = two_biggest[0] - two_biggest[1]
    
    if score_margin <= 50:
        # define the payload
        f = open(file_path, 'rb') 
        audio_b64bytes = base64.b64encode(f.read())
        audio_string = audio_b64bytes.decode()
        
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        
        senML_json = {
                    'bn' : 'http://169.254.37.210/',
                    'bt' : timestamp,
                    'e' : {'n' : 'audio', 'u' : '/', 't' : 0, 'vd' : audio_string}
                    }
        
        # compute the communication cost of this message
        temp = json.dumps(senML_json)
        communication_cost += len(temp)
        
        r = requests.put(url, json=senML_json)
        
        if r.status_code == 200:
            body = r.json()
            y_pred = int(body['prediction'])
            predictions.append(y_pred)
        else:
            print('Error:', r.status_code)
            print(r.text)
        
    else:
        predictions.append(y_pred)    
        
accuracy = sum(1 for x,y in zip(predictions, true_labels) if x == y)/float(len(predictions))
print('Accuracy: '+str(accuracy))
print('Communication cost: '+str(communication_cost))
       
        
    
    


