from tkinter import *
from PIL import ImageTk, Image
import cv2
from tkinter import filedialog
from tkinter import ttk
import threading

import pandas as pd
import numpy as np
import pytesseract
import re
from scipy import stats
import tensorflow as tf


CATEGORIES = ['coral_reef', 'dead_coral', 'macro_algae', 'offshore_sand_sheets', 'other', 'outer_reef_sand',
              'reef_rubble', 'seagrass']

den_cat = ['dense', 'sparse']

cols = ['habitat', 'density', 'depth', 'temp', 'lat', 'lon', 'lat_lon_str', 'frame']

model = tf.keras.models.load_model('habitat_base_model')
den_mod = tf.keras.models.load_model('density_base_model')

IMG_SIZE = 150

lat_regex = '2+[0-9]\.+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]'
lon_regex = '5+[0-9]\.+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]'
temp_regex = '[0-9]+[0-9]\.+[0-9]+[0-9]'
depth_regex = '[0-9]\.+[0-9]'

root = Tk()
root.title('Automated Habitat Video Classification')


def open():
    global my_image
    root.filename = filedialog.askopenfilename(initialdir='/Documents', title='Select file',
                                               filetypes=(('webm files', '*.webm'), ('mp4 files', '*.mp4'), ('all files', '*.*')))
    vidcap = cv2.VideoCapture(root.filename)
    success, image = vidcap.read()
    count = 0
    lst = []

    while success:
        success, image = vidcap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(image, lang='eng')
        new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        new_array = new_array / 255
        img = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(img)
        habitat = CATEGORIES[int(prediction.argmax())]

        if habitat == 'coral_reef' or habitat == 'macro_algae' or habitat == 'seagrass':
            den_pred = den_mod.predict(img)
            density = den_cat[int(den_pred.argmax())]
        else:
            density = ''

        if bool(re.search(lat_regex, text)) and bool(re.search(lon_regex, text)):
            lat = float(re.findall(lat_regex, text)[0])
            lon = float(re.findall(lon_regex, text)[0])
            temp = re.findall(temp_regex, text)
            water_temp = float(temp[-3])
            depth = float(re.findall(depth_regex, text)[1])
            lst.append([habitat, density, depth, water_temp, lat, lon, str(lat) + str(lon), count])

        else:
            lat = 0
            lon = 0
            temp = 0
            depth = 0
            lst.append([habitat, density, depth, temp, lat, lon, str(lat) + str(lon), count])

        #print('Image_' + str(count) + '_' + habitat + '_' + str(lat) + '_' + str(lon))

        count += 30
        vidcap.set(1, count)
        my_progress['value'] += 10
        root.update_idletasks()

    df1 = pd.DataFrame(lst, columns=cols)
    df2 = df1.groupby(['lat_lon_str']).agg(lambda x: stats.mode(x)[0][0])
    df2 = df2.iloc[1:]
    df2 = df2.sort_values('frame')
    df2.to_csv(r'Habitats_Transect_' + str(root.filename[-9:-5]) + '.csv', index=False)
    top = Toplevel()
    Label(top, text='Classification of first video has completetd\n and been saved as:\n' +
                    'Habitats_' + str(root.filename[-9:-5]) + '.csv').pack(pady=10)




def open2():
    global my_image
    root.filename2 = filedialog.askopenfilename(initialdir='/Documents', title='Select file',
                                               filetypes=(('webm files', '*.webm'), ('mp4 files', '*.mp4'), ('all files', '*.*')))
    vidcap = cv2.VideoCapture(root.filename2)
    success, image = vidcap.read()
    count = 0
    lst = []

    while success:
        success, image = vidcap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(image, lang='eng')
        new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        new_array = new_array / 255
        img = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(img)
        habitat = CATEGORIES[int(prediction.argmax())]

        if habitat == 'coral_reef' or habitat == 'macro_algae' or habitat == 'seagrass':
            den_pred = den_mod.predict(img)
            density = den_cat[int(den_pred.argmax())]
        else:
            density = ''

        if bool(re.search(lat_regex, text)) and bool(re.search(lon_regex, text)):
            lat = float(re.findall(lat_regex, text)[0])
            lon = float(re.findall(lon_regex, text)[0])
            temp = re.findall(temp_regex, text)
            water_temp = float(temp[-3])
            depth = float(re.findall(depth_regex, text)[1])
            lst.append([habitat, density, depth, water_temp, lat, lon, str(lat) + str(lon), count])

        else:
            lat = 0
            lon = 0
            temp = 0
            depth = 0
            lst.append([habitat, density, depth, temp, lat, lon, str(lat) + str(lon), count])

        #print('Image_' + str(count) + '_' + habitat + '_' + str(lat) + '_' + str(lon))

        count += 30
        vidcap.set(1, count)
        my_progress2['value'] += 10
        root.update_idletasks()

    df3 = pd.DataFrame(lst, columns=cols)
    df4 = df3.groupby(['lat_lon_str']).agg(lambda x: stats.mode(x)[0][0])
    df4 = df4.iloc[1:]
    df4 = df4.sort_values('frame')
    df4.to_csv(r'Habitats_Transect_' + str(root.filename2[-9:-5]) + '.csv', index=False)
    top2 = Toplevel()
    Label(top2, text='Classification of second video has completetd\n and been saved as:\n' +
                    'Habitats_' + str(root.filename2[-9:-5]) + '.csv').pack(pady=10)





title_lab = Label(root, bg='white',
                  text='Welcome to the automated video classification user interface.\nAll you need to do is select your video'
                             ' and wait for it to\nfinish. The results will be written to a CSV file in the same'
                             ' folder as this\nprogram. You can select up to 2 files to classify at a time. Enjoy!!!!!')
title_lab.pack()


logo = ImageTk.PhotoImage(Image.open('exxon_logo.jpg'))
panel = Label(root, image=logo).pack()

my_btn = Button(root, text='    Choose First Video    ', command=threading.Thread(target=open).start)
my_btn.pack()


my_progress = ttk.Progressbar(root, orient=HORIZONTAL, length=200, mode='indeterminate')
my_progress.pack(pady=10)


my_btn2 = Button(root, text='    Choose Second Video    ', command=threading.Thread(target=open2).start)
my_btn2.pack()

my_progress2 = ttk.Progressbar(root, orient=HORIZONTAL, length=200, mode='indeterminate')
my_progress2.pack(pady=10)


root['bg'] = 'white'

root.mainloop()


