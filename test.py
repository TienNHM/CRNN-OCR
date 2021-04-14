#%%
import os
import fnmatch
import cv2
import numpy as np
import string
import time

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
import tensorflow as tf

#ignore warnings in the output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#%%
# char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# total number of our output classes: len(char_list)
char_list = string.ascii_letters+string.digits
 
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

#%%
path = 'mnt/ramdisk/max/90kDICT32px'
 
#lists for validation dataset
raw_img = []
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []
 
max_label_len = 0

i = 10
flag = 0
 
for root, dirnames, filenames in os.walk(path):
    for f_name in fnmatch.filter(filenames, '*.jpg'):            
        # split the 150000 data into validation and training dataset as 10% and 90% respectively
        if i%10 == 0:
            print(i)
            # read input image and convert into gray scale image
            img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)   
            
            # convert each image of shape (32, 128, 1)
            w, h = img.shape
            if h > 128 or w > 32:
                continue
            
            raw_img.append(img)
            
            if w < 32:
                add_zeros = np.ones((32-w, h))*255
                img = np.concatenate((img, add_zeros))
     
            if h < 128:
                add_zeros = np.ones((32, 128-h))*255
                img = np.concatenate((img, add_zeros), axis=1)
            img = np.expand_dims(img , axis = 2)
            
            # Normalize each image
            img = img/255.
            
            # get the text from the image
            txt = f_name.split('_')[1]
            
            # compute maximum length of the text
            if len(txt) > max_label_len:
                max_label_len = len(txt)
            
            valid_orig_txt.append(txt)   
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))  
        
        # break the loop if total data is 150000
        if i == 150000:
            flag = 1
            break
        i+=10
    if flag == 1:
        break

#%%
# pad each output label to maximum text length
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))

#%%
inputs = Input(shape=(32,128,1))
 
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
act_model.summary()

#%%
labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
 
 
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 
 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
 
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#%%
valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

#%%
def VisualizeHistory(file_history):
    ''' Vẽ đồ thị accuracy và loss '''

    # load history
    saved_history = np.load(file_history, allow_pickle='TRUE').item()
    # summarize history for loss
    plt.plot(saved_history['loss'])
    plt.plot(saved_history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

#%%
# load the saved best model weights
act_model.load_weights('best_model.hdf5')
file_history = "history.npy"
VisualizeHistory(file_history)

# predict outputs on validation images
prediction = act_model.predict(valid_img)
 
# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

#%%
def TestImage(index, text):
    image = raw_img[index]
    h,w = image.shape
    image = cv2.resize(image, (w*3, h*3))
    image = cv2.copyMakeBorder(image, 0, int(h*1.5), 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        image, #numpy array on which text is written
        text,
        (0, h*4), #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        1, #font size
        (255,255,255), #font color
        2,
        cv2.LINE_AA) #font stroke
    filename = "test/" + str(index) + "-" + valid_orig_txt[index] + ".png"
    cv2.imwrite(filename, image)
    
#%%
def convert_to_plaintext(output):
    pred = ""
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)] , end = '') 
            pred = pred + char_list[int(p)] 
    return pred

#%%
# see the results
i = 0
correct = []

for x in out:
    orig = valid_orig_txt[i]
    print("orig_text =  " + orig)
    pred = convert_to_plaintext(x)
    print("pred_text = " + pred)
    TestImage(i, pred)
    if pred == orig: 
        correct.append(raw_img[i])
    print('\n')
    i+=1
    
print("Test accuracy (%): ", len(correct)*100/len(out))

#%%
def RawImage(index):
    image = raw_img[index]
    h,w = image.shape
    image = cv2.resize(image, (w*3, h*3))
    image = cv2.copyMakeBorder(image, 0, int(h*1.5), 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        image, #numpy array on which text is written
        valid_orig_txt[index],
        (0, h*4), #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        1, #font size
        (255,255,255), #font color
        2,
        cv2.LINE_AA) #font stroke
    filename = "raw/" + str(index) + "-" + valid_orig_txt[index] + ".png"
    cv2.imwrite(filename, image)
    
for j in range(len(raw_img)): RawImage(j)
