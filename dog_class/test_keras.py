#!/usr/bin/env python
# -*- coding:utf8 -*-
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255-0.5
x_test = x_test.astype('float32')/255-0.5
x_train = x_train.reshape((x_train.shape[0],-1))
x_test = x_test.reshape((x_test.shape[0],-1))
print(x_train.shape)
print(x_test.shape)

#784压缩为2
encoding_dim = 2

#Input相当于placeholder
input_img = Input(shape=(784,))

#encoder layer   输出                    输入
encoded = Dense(128,activation='relu')(input_img)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(10,activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

#decoder layer
decoded = Dense(10,activation='relu')(encoded_output)
decoded = Dense(64,activation='relu')(encoded)
decoded = Dense(128,activation='relu')(encoded)
decoded = Dense(784,activation='tanh')(encoded)

"""两个模型，一个是编码和解码整体，另一个只包含类编码"""
#construct the autoencoder model  从line19开始到line31
autoencoder = Model(input=input_img,output=decoded) 
#construct the encoder model for plotting 包含从line19 到line25
encoder = Model(input=input_img, output=encoded_output)

#compile autocoder
autoencoder.compile(optimizer='adam',loss='mse')

#training     输入输出值都是x_train
autoencoder.fit(x_train,x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
               )

#plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c = y_test)
plt.colorbar()
plt.show()
