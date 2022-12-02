# -----Train Model-----

import os
import cv2
import numpy as np
from PIL import Image


# use this function to obtain id and images

def get_images_and_labels(path):
    # load classifier
    detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # obtain images path
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    # place id and
    for image_path in image_paths:

        # transfer to gary
        img = Image.open(image_path).convert('L')

        # transfer to array
        img_np = np.array(img, 'uint8')

        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        # split id and obtain name\if\face
        name = os.path.split(image_path)[-1].split(".")[1]
        id = int(os.path.split(image_path)[-1].split(".")[-2])
        faces = detector.detectMultiScale(img_np)

        # add img and id in list
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(id)
    return face_samples, ids, name


# delete all the images in the data
def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + "\\" + i
        os.remove(file_data)


def training():
    # setting the path of image data
    path = './data'
    # initialization of recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print('Training...')
    faces, ids, name = get_images_and_labels(path)
    # training model
    recognizer.train(faces, np.array(ids))
    # save model
    recognizer.save('models/' + str(name) + '.yml')
    print("Training completed,model saved")

    del_file(path)
    print('Temporary images in data has been deleted')
