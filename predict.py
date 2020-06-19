#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import backend

class waste:
    def __init__(self,filename):
        self.filename =filename


    def predictionwaste(self):
        # it tells Keras that you will be using predict only and not teaching your CNN.
        backend.set_learning_phase(0)
        # load model
        model = load_model('my_keras_model.h5')
        #model = tf.keras.models.load_model('model_14.h5')
        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (200, 200))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        class_idx = np.argmax(result[0])
        if(class_idx==0):
            prediction = 'NON-RECYCLABLE'
            return [{ "image" : prediction}]
        elif(class_idx== 1):
            prediction = 'ORGANIC'
            return [{ " image " : prediction}]

        else:
            prediction='RECYCLABLE'
            return [{" image " : prediction}]



