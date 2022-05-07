import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def fetch_data(DATA_PATH, labels, sequence_length=30):

    label_map = {label: num for num, label in enumerate(labels)}

    videos, cls = [], []

    for label in labels:

        print(f'Generating Tensors for - {label}')
        filenames = os.listdir(os.path.join(DATA_PATH, label))

        if filenames is None:
            print(f'{os.path.join(DATA_PATH, label)} empty.')
        else:
            for vid in filenames:
                window = []

                for frame_num in range(1, sequence_length + 1):
                    try:
                        frame = np.load(os.path.join(DATA_PATH, label, vid, f'{frame_num}.npy'))
                    except:
                        print(os.path.join(DATA_PATH, label, vid, f'{frame_num}.npy'))
                        frame = np.zeros((1662,))
                    window.append(frame)

                # print(np.array(window).shape)
                videos.append(window)
                cls.append(label_map[label])

    print(f'(Number of Videos, frames/vid, keypoints/frame) = {np.array(videos).shape}')
    print(f'(Number of Videos) = {np.array(cls).shape}')

    X = np.array(videos)
    y = to_categorical(cls).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    src_path = 'D:\\0. PSU\\!Projects\\0. Dataset\\Prep_Data'
    dst_path = 'D:\\0. PSU\\!Projects'
    labels = ['HELLO', 'THANK YOU', 'FATHER', 'MOTHER', 'MY', 'YOU']

    # 1 ['BYE', 'EAT FOOD', 'FATHER', 'FINE', 'FIVE']
    # 2 ['GOOD MORNING', 'HELLO', 'HELP', 'HOW', 'I']
    # 3 ['LIKE', 'MEET', 'MORE', 'MOTHER', 'MY', 'NAME']
    # 4 ['NICE', 'NO', 'PLEASE', 'SEE YOU LATER']
    # 5 ['THANK YOU', 'WANT', 'WHAT', 'YES', 'YOU']

    X_train, X_test, y_train, y_test = fetch_data(src_path, labels=labels, sequence_length=30)

    save = X_train, X_test, y_train, y_test

    # Save Tensors
    dst_path = os.path.join(dst_path, '1. Tensors')
    exp_name = 'Demo-1'
    dst_path = os.path.join(dst_path, exp_name)

    # Making Root Folder
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        print(f'Tensors/{exp_name} created')
    else:
        print(f'Tensors/{exp_name} exists')

    names = ['X_train', 'X_test', 'y_train', 'y_test']

    for name, arr in zip(names, save):
        npy_path = os.path.join(dst_path, name)
        np.save(npy_path, arr)
        print(f'{npy_path} - {arr.shape} saved.')