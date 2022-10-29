import cv2 as cv
import os

def read_directory(path):
    class_count = 1
    if not os.path.exists('./train_data'):
        os.mkdir('./train_data')
    if not os.path.exists('./test_data'):
        os.mkdir('./test_data')
    split = 1

    for img_name in os.listdir(path):
        label = img_name.split('_')[0]
        if not os.path.exists('./train_data/' + label):
            os.mkdir('./train_data/' + label)
        if not os.path.exists('./test_data/' + label):
            os.mkdir('./test_data/' + label)
        img = cv.imread(path + '/' + img_name)

        if split <= 8000:
            cv.imwrite('./train_data/' + label + '/' + img_name, img)
        else:
            cv.imwrite('./test_data/' + label + '/' + img_name, img)

        if split >= 10000:
            split = 1
            print("Progress:", class_count, "/9")
            class_count += 1
        split += 1

img_path = './geometry_dataset/output'
read_directory(img_path)
