import tensorflow as tf
import numpy as np
from cnnModel import cnnModel
import os
import pickle
import time
import getConfig
import sys
import random
gConfig = {}
gConfig = getConfig.get_config(config_file="config.ini")
def read_data(dataset_path,im_dim,num_channels,num_files,images_per_file):
    files_names = os.listdir(dataset_path)
    dataset_array = np.zeros(shape=(num_files*images_per_file,im_dim,im_dim,num_channels))
    dataset_labels = np.zeros(shape=(num_files*images_per_file),dtype=np.uint8)
    index = 0
    for file_name in files_names:
        if file_name[0:len(file_name)-1] == "data_batch_":
            print("正在处理数据:",file_name)
            data_dict = unpickle_patch(dataset_path + file_name)
            images_data = data_dict[b"data"]
            print(images_data.shape)
            images_data_reshaped = np.reshape(images_data,newshape= (len(images_data),im_dim,im_dim,num_channels))
            dataset_array[index*images_per_file:(index + 1)*images_per_file,:,:,:]=images_data_reshaped
            dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
            index = index + 1
        return dataset_array, dataset_labels

def unpickle_patch(file):
    patch_bin_file = open(file,'rp')
    patch_dict = pickle.load(patch_bin_file,encoding='bytes')
    return patch_dict

def create_model():
    if 'pretrained_model'in gConfig:
        model= tf.keras.models.load_model(gConfig['pretrained_model'])
        return model
    ckpt=tf.io.gfile.listdir(gConfig['Working_directory'])

    if ckpt:
        model_file=os.path.join(gConfig['pretrained_model'],ckpt[-1])
        print("Reading model parameters from %s" % model_file)
        model = tf.keras.models.load_model(model_file)
        return model
    else:
        model=cnnModel(gConfig['learnimg_rate'],gConfig['rate'])
        model=model.createModel()
        return model

dataset_array, dataset_labels= read_data(dataset_path=gConfig['dataset_path'],im_dim=gConfig['im_dim'],
                                             num_channels=gConfig['num_channels'],num_files=gConfig['num_files'],
                                             images_per_file=gConfig['images_per_file'])
dataset_array=dataset_array.astype('float32')/255
dataset_labels=tf.keras.utils.to_categorical(dataset_labels,10)

def train():
     model=create_model()
     history=model.fit(dataset_array,dataset_labels,verbose=1,epochs=10,validation_split=0.2)
     filename='cnn_model.h5'
     checkpoint_path = os.path.join(gConfig['working_directory'],filename)
     model.save(checkpoint_path)

def predict(data):
    ckpt=os.listdir(gConfig['working_directory'])
    checkpoint_path = os.path.join(gConfig['working_directory'],'cnn_model.h5')
    model=tf.keras.models.load_model(checkpoint_path)
    predicton=model.predict(data)
    index=tf.math.argmax(predicton[0]).numpy()
    return label_names_dict[index]
if __name__=='_main_':
    gConfig = getConfig.get_config()
    if getConfig['mode']=='train':
        train()
    elif getConfig['mode']=='server':
        print('请使用：python3 app.py')
