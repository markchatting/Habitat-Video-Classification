import shutil
import os

train_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/labelled_copy/train'
test_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/labelled_copy/test'
for dirname in os.listdir(train_dir):
    from_dir = os.path.join(train_dir + '/' + dirname + '/')
    dest_dir = os.path.join(test_dir + '/' + dirname + '/')
    for filename in os.listdir(from_dir)[::5]:
        shutil.move(from_dir + filename, dest_dir)
