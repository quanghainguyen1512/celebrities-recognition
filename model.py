from keras.models import Model
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.layers import Input
from sklearn.preprocessing import normalize
import numpy as np


# model
def VggFace(path = None, is_origin=False):
    img = Input(shape=(224, 224,3))

    #convolution layers
    conv1_1 = Conv2D(64, (3,3), activation='relu', name='conv1_1',padding='same')(img)
    conv1_2 = Conv2D(64, (3,3), activation='relu', name='conv1_2',padding='same')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3,3), activation='relu', name='conv2_1',padding='same')(pool1)
    conv2_2 = Conv2D(128, (3,3), activation='relu', name='conv2_2',padding='same')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3,3), activation='relu', name='conv3_1',padding='same')(pool2)
    conv3_2 = Conv2D(256, (3,3), activation='relu', name='conv3_2',padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3,3), activation='relu', name='conv3_3',padding='same')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3,3), activation='relu', name='conv4_1',padding='same')(pool3)
    conv4_2 = Conv2D(512, (3,3), activation='relu', name='conv4_2',padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3,3), activation='relu', name='conv4_3',padding='same')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3,3), activation='relu', name='conv5_1',padding='same')(pool4)
    conv5_2 = Conv2D(512, (3,3), activation='relu', name='conv5_2',padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3,3), activation='relu', name='conv5_3',padding='same')(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool5')(conv5_3)

    #classification layer of original mat file
    fc6 = Conv2D(4096, (7,7), activation='relu', name='fc6',padding='valid')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    
    fc7 = Conv2D(4096, (1,1), name='fc7',padding='valid')(fc6_drop)
    #norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(fc7)
    
    fc7_activation = Activation('relu')(fc7)
    fc7_drop = Dropout(0.5)(fc7_activation)
    fc8 = Conv2D(2622, (1,1), activation='relu', name='fc8',padding='valid')(fc7_drop)
    flat = Flatten(name='flat')(fc8)
    
    # 
    norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(flat)
    
    prob = Activation('softmax',name='prob')(flat)

    if is_origin:
      model = Model(inputs = img, outputs = prob)
      model._make_predict_function()
      model.load_weights(path)
      return Model(inputs = img, outputs = norm)
    else:
      model = Model(inputs = img, outputs = norm)
      model._make_predict_function()
      model.load_weights(path)
      return model
