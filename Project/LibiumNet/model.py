
"""
This model generates generator of the datasets for the Network. 

@authors : Mustapha Tidoo Yussif, Samuel Atule, Jean Sabastien Dovonon
         and Nutifafa Amedior. 
"""

import os, glob
import imageio
import itertools
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, TimeDistributed, LSTM, Input, CuDNNLSTM, BatchNormalization, Conv2D, MaxPooling2D, Reshape, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

imageio.plugins.ffmpeg.download()
tf.enable_eager_execution()


class LibiumNet(object):
    """TA lipreading model, `LibiunNet`
    This is lip reading model which reads or predicts the words of a spoken mouth in a silent video. 
    This model implements the RCNN (Recurrent Convolutional Neural Network) architecture. 

    :param img_c: The number of channels of the input image. i.e. a frame in a video (default 3).
    :param img_w: The width of the input image i.e. a frame in a video (default 256)
    :param img_h: The height of the input image i.e. a frame in a video (default 256)
    :param frames_n: The total number of frames in an input video (default 29)
    :param output_size: The output size of the network. 
    
    """
    def __init__(self, img_c=3, img_w=256, img_h=256, frames_n=29, output_size=10):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.output_size = output_size
        self.build()
    
    def build(self):
        """
        Retrieves the features from the last pool layer in the densenet pretrained model 
        and pass obtained features to LSTM network. 
        """
        input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c) # input shape

        feature_extractor = Sequential()
        inputShape = (self.img_w, self.img_h, self.img_c)
        chanDim = -1
        
        feature_extractor.add(Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True), input_shape=inputShape))
        feature_extractor.add(MaxPooling2D(pool_size=(2, 2)))
        
        # first CONV => RELU => CONV => RELU => POOL layer set
        
        feature_extractor.add(Conv2D(32, (3, 3)))
        feature_extractor.add(Activation("relu"))
        feature_extractor.add(BatchNormalization(axis=chanDim))
        feature_extractor.add(Conv2D(32, (3, 3)))
        feature_extractor.add(Activation("relu"))
        feature_extractor.add(BatchNormalization(axis=chanDim))
        feature_extractor.add(MaxPooling2D(pool_size=(2, 2)))
        feature_extractor.add(Dropout(0.25))
        
        # second CONV => RELU => CONV => RELU => POOL layer set
        feature_extractor.add(Conv2D(64, (3, 3)))
        feature_extractor.add(Activation("relu"))
        feature_extractor.add(BatchNormalization(axis=chanDim))
        feature_extractor.add(Conv2D(64, (3, 3)))
        feature_extractor.add(Activation("relu"))
        feature_extractor.add(BatchNormalization(axis=chanDim))
        feature_extractor.add(MaxPooling2D(pool_size=(2, 2)))
        feature_extractor.add(Dropout(0.25))
        
        
        TIME_PERIODS = self.frames_n
        dims = 53824

        model_m = Sequential()
        model_m.add(Conv1D(10, 2, activation='relu', input_shape=(TIME_PERIODS, dims)))
        model_m.add(Conv1D(10, 2, activation='relu'))
        
        
        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        self.image_frame_features = TimeDistributed(feature_extractor)(self.input_data) ## extracting the features from the images
        
        self.flat = TimeDistributed(Flatten())(self.image_frame_features) ## flatten before passing on to the recurrent network

        self.sequence1 = model_m(self.flat)
        self.sequence = CuDNNLSTM(16)(self.sequence1) 
        
        self.dense3 = Dense(10, activation='relu')(self.sequence)
        self.dense2 = Dense(10, activation='relu')(self.dense3)
        self.dense = Dense(self.output_size, name='logits')(self.dense2)

        self.pred = Activation('softmax', name='softmax')(self.dense)


        self.model = Model(inputs = self.input_data, outputs=self.pred)


    def summary(self):
        """"Summarizes the architecture of the model.
        
        :return: returns the model architecture summary
        """
        return self.model.summary()
      
    
    def train(self, generator,steps_per_epoch=None, epochs=1,validation_data=None, validation_steps=None, filepath="/gdrive/My Drive/LibiumNet/checkpoint.h5"):
        # Callbacks
        early_stopping_monitor = EarlyStopping(patience=3)
        checkpoint = ModelCheckpoint(
                            filepath, monitor='val_acc', verbose=1, 
                            save_best_only=True, mode='max'
                    )
        
        reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', factor=0.2,
                              patience=7, min_lr=1e-9
                    )
        
        callbacks_list = [checkpoint, reduce_lr, early_stopping_monitor]

        
        print('Training...')
        
        self.model.compile(
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy']
        )
        
        history = self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data, validation_steps = validation_steps)
        
        #self.visualize_accuracy(history)
        self.visualize_loss(history)
      
      
    def predict(self, input_batch):
        """Predicts a video
        
        :param input_batch: A batch of a sequence of frames. 
        :return: returns the predicted probailities
        """
        return self.model(input_batch)
      
    def visualize_accuracy(self, history):
      """Visualize model accuracy
      """
      plt.plot(history.history['acc'], label='training accuracy')
      plt.plot(history.history['val_acc'], label='testing accuracy')
      plt.title('Accuracy')
      plt.xlabel('epochs')
      plt.ylabel('accuracy')
      plt.legend()
      
    def visualize_loss(self, history):
      """Visualizes model loss"""
      plt.plot(history.history['loss'], label='training loss')
      plt.plot(history.history['val_loss'], label='testing loss')
      plt.title('Loss')
      plt.xlabel('epochs')
      plt.ylabel('loss')
      plt.legend()