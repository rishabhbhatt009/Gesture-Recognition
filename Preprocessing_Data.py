# Imports
import cv2
import numpy as np
import os
import json
import shutil

import mediapipe as mp
import KeyPointDetection as kpd

# Generating Landmark Data ------------------------------------------------------------------------

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def process_frame(frame, frame_number, dst_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Normalise
        dim = (1280, 720)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Make detections
        image, results = kpd.mediapipe_detection(frame, holistic)
        # Draw landmarks
        kpd.draw_landmarks(image, results)

        # Export key-points (1662,) => 468*3 + 33*4 + 21*3 + 21*3 = 1662
        key_points = kpd.extract_keypoints(results)

        # Save Keypoints
        npy_path = os.path.join(dst_path, str(frame_number))
        np.save(npy_path, key_points)

        # Display Frame
        # cv2.imshow('Video1', frame)
        # cv2.imshow('Video2', image)


def generate_data(src_path, dst_path):
    cap = cv2.VideoCapture(src_path)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Last frame is garbage EOF frame
    size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(f'{frames=}, {size=}, {fps=}')

    frame_num = 0
    sample_size = 30
    sample = 0
    length = frames / fps
    sample_rate = length / sample_size

    while cap.isOpened():
        # Read Frame
        check, frame = cap.read()

        # Inc Frame Count
        frame_num += 1

        # Sampling Algo --------------------------------------------------
        if check:
            if frames > sample_size:
                if frame_num * sample_size // frames > sample:
                    sample += 1
                    process_frame(frame, sample, dst_path)
                    # print(f'{sample=}, frame={frame_num}')
            else:
                while sample * sample_rate < frame_num * (1 / fps):
                    sample += 1
                    process_frame(frame, sample, dst_path)

            # Press 'q' to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            cap.release()
            break

    if sample != 30:
        print(f'Fault : {frames=} {fps=} {sample=} : {src_path}')


if __name__ == '__main__':
    # Generating Landmark Data ------------------------------------------------------------------------

    src_path = 'D:\\0. PSU\\!Projects\\0. Dataset\\Data_Videos'
    dst_path = 'D:\\0. PSU\\!Projects\\0. Dataset'

    dst_path = os.path.join(dst_path, 'Prep_Data')

    # Making Root Folder
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        print('Prep_Data/ created')
    else:
        print('Prep_Data/ exists')

    done = []
    to_do = []

    #1 ['BYE', 'EAT FOOD', 'FATHER', 'FINE', 'FIVE']
    #2 ['GOOD MORNING', 'HELLO', 'HELP', 'HOW', 'I']
    #3 ['LIKE', 'MEET', 'MORE', 'MOTHER', 'MY', 'NAME']
    #4 ['NICE', 'NO', 'PLEASE', 'SEE YOU LATER']
    #5 ['THANK YOU', 'WANT', 'WHAT', 'YES', 'YOU']

    for label in os.listdir(src_path):

        if label not in to_do:
            continue

        label_src_path = os.path.join(src_path, label)
        label_dst_path = os.path.join(dst_path, label)

        filenames = set(os.listdir(label_src_path))

        print(f'Generating\t : {label_dst_path}')

        for vid in filenames:

            # Generating Data for each label
            vid_src_path = os.path.join(label_src_path, vid)
            vid_dst_path = os.path.join(label_dst_path, vid)

            if not os.path.exists(vid_dst_path):
                os.makedirs(vid_dst_path)
            else:
                print(f'{vid_dst_path} exists')

            # print(f'Generating\t : {vid_src_path}')
            generate_data(vid_src_path, vid_dst_path)
            # print(f'Generated\t :{vid_dst_path}')

        print(f'Generated\t : {label_dst_path}')
        done.append(label)
        print(done)