import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

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


def train(labels, X_train, X_test, y_train, y_test):

    # Model -------------------------------------------------
    model = model_create(np.array(labels))
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    early_callback = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
    log_csv = CSVLogger('my_logs_all.csv', separator='|', append=False)

    filepath = "saved_models/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(X_train, y_train, epochs=500, validation_split=0.1, callbacks=[tb_callback, early_callback, log_csv])

    # model.save('action.h5')
    print('Model Trained')
    print(model.summary())

    return model


if __name__ == '__main__':
    src_path = 'D:\\0. PSU\\!Projects\\1. Tensors'
    dst_path = 'D:\\0. PSU\\!Projects\\2. Models'

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

    labels = ['BYE', 'EAT FOOD', 'FATHER', 'FINE', 'FIVE', 'GOOD MORNING', 'HELLO', 'HELP', 'HOW', 'I', 'LIKE', 'MEET', 'MORE', 'MOTHER', 'MY', 'NAME', 'NICE', 'NO', 'PLEASE', 'SEE YOU LATER', 'THANK YOU', 'WANT', 'WHAT', 'YES', 'YOU']
    model = train(labels, X_train, X_test, y_train, y_test)

    # dst_path = os.path.join(dst_path, 'Models')
    # # Making Root Folder
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)
    #     print('Models/ created')
    # else:
    #     print('Models/ exists')

    model_name = 'Exp4'
    model.save(os.path.join(dst_path, model_name+'.h5'))
