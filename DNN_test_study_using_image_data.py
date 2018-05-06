# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras import datasets
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    # 정수(1D) -> 이진 벡터(10class, 2D)
    Y_train = np_utils.to_categorical(y_train)
    Y_test  = np_utils.to_categorical(y_test)

    #벡터 이미지 형태로 변환(4D->2D)
    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W*H*C)
    X_test  = X_test.reshape(-1, W*H*C)

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0
    return (X_train, Y_train), (X_test, Y_test)

from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, Nin, Nh, Pd, Nout):#Pd_1 - 2 argument로 드롭아웃 확률을 지정
        super().__init__()
        #Hidden 1,2 활성함수 relu, relu, softmax, 
        self.add(layers.Dense(Nh[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd[0]))
        self.add(layers.Dense(Nh[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(Pd[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        
#from keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

def main():
    Nh = [100,50]
    Pd = [0.0,0.0] 
    number_of_class = 10
    Nout = number_of_class
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    model = DNN(X_train.shape[1], Nh, Pd, Nout)
    #학습(100회, 흐름은 history에 저장)
    history = model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)
    plt.show()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
        
        