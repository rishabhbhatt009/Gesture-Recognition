import sys
import cv2
import numpy as np
import os


# Folder Setup _------------------------------------------------------------------
def setup_folder(path, labels):
    path = os.path.join(path, 'Data_Collection')

    # Making Root Folder
    if not os.path.exists(path):
        os.mkdir(path)
        print('Data_Collection/ created')
    else:
        print('Data_Collection/ exists')

    for label in labels:
        try:
            os.makedirs(os.path.join(path, label))
            print(f'{label}/ created')
        except Exception as E:
            print(E)
            print(f'{label}/ exists')
    return path


# Collection -------------------------------------------------------------------
def collect_data(labels, signer, dst_path='', no_sequences=30, sequence_length=60):
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    win_name = 'WebCam Feed'

    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    print(f'WebCam Prop : FRAME SIZE={size}, FPS={fps}')

    def display_comment(window_name, cmt, action, vid_num=0):
        blank_image = np.zeros((480, 640, 3), np.uint8)
        if cmt == 'Start':
            cv2.putText(blank_image, f'STARTING COLLECTION FOR \'{action}\'', (100, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(blank_image, f'PRESS \'s\' TO START', (100, 230), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            print('Waiting ....')
            while cv2.waitKey(10) & 0xFF != ord('s'):
                cv2.imshow(window_name, blank_image)

        elif cmt == 'Next':
            cv2.putText(blank_image, f'Collecting frames for {label} - Video Number {str(vid_num)}/{no_sequences}', (70, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(blank_image, f'Press \'n\' to start next video', (70, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            print('Waiting ....')
            while cv2.waitKey(10) & 0xFF != ord('n'):
                cv2.imshow(window_name, blank_image)

        return

    if not capture.isOpened():
        print('ERROR - CAM DID NOT CONNECT')
        sys.exit(0)

    for label in labels:
        print(f'Collecting data for \'{label}\'')
        display_comment(win_name, 'Start', label)

        vid_cnt = 1
        while vid_cnt <= no_sequences:

            # Video writer Config
            vid_name = os.path.join(dst_path, label, signer+str(vid_cnt))
            out = cv2.VideoWriter(vid_name + '.mp4', fourcc, fps, size)

            display_comment(win_name, 'Next', label, vid_cnt)
            # Collect Video Frames
            frame_cnt = 1
            while frame_cnt <= sequence_length:

                # Capture Frame by Frame
                check, frame = capture.read()

                if check:
                    cv2.imshow(win_name, frame)
                    out.write(frame)
                    frame_cnt += 1
                else:
                    print('ERROR : BAD FRAME')
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    capture.release()
                    break

            vid_cnt += 1
            out.release()

        print(f'Label = {label}, data collected.')

    capture.release()
    cv2.destroyAllWindows()
    print('Released')
    return


if __name__ == '__main__':
    name = 'Rishabh'
    actions = ['hello', 'Mother', 'Father', 'bye', 'five', 'thank you']
    # ['eat food', 'fine', 'Good Morning', 'help', 'how', 'I', 
    #  'like', 'meet', 'more', 'my', 'name', 'nice', 'no', 'please', 
    #  'see you later', 'want', 'what', 'yes', 'you']
    destination_path = ''
    no_of_videos = 1
    no_of_frames = 60

    destination_path = setup_folder(destination_path, actions)
    collect_data(actions, name, destination_path, no_of_videos, no_of_frames)
