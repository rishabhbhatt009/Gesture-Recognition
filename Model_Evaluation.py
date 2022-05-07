import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def model_create(labels):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(labels.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print('Model Created')
    print(model.summary())

    return model


if __name__ == '__main__':

    src_path = 'D:\\0. PSU\\!Projects\\1. Tensors'

    tensors = {}
    name = 'First - All'
    src_path = os.path.join(src_path, name)

    for tensor in os.listdir(src_path):
        load = np.load(os.path.join(src_path, tensor))
        print(f'Tensor Loaded = {load.shape}')
        tensors[tensor.split('.')[0]] = load

    X_train = tensors['X_train']
    X_test = tensors['X_test']
    y_train = tensors['y_train']
    y_test = tensors['y_test']

    labels = ['HELLO', 'THANK YOU', 'FATHER', 'MOTHER', 'MY', 'YOU']

    model = model_create(np.array(labels))
    src_path = 'D:\\0. PSU\\!Projects\\2. Models\\Exp3.h5'
    model.load_weights(src_path)

    # Evaluation
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    arr = multilabel_confusion_matrix(ytrue, yhat)
    acc = accuracy_score(ytrue, yhat)

    print(arr)
    print(acc)

