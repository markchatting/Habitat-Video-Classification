import cv2
import os

#This script extracts frames from the habitat videos. These frames will later be used for model training.

dir_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/Videos'
dir_dir2 = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/Vid_Images_test'
for filename in os.listdir(dir_dir):
    vidcap = cv2.VideoCapture(os.path.join(dir_dir, filename))
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(dir_dir2, filename[0:-5], filename[0:-5] + "frame%d.jpg") % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
