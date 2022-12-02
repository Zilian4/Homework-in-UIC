# -----obtain samples of face-----
import cv2


def sampling():
    # open camera
    cap = cv2.VideoCapture(0)
    # load the classifier
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # mark an id for new face
    face_id = input('\n User data input,Look at the camera and wait ...\n'
                    'please input username')
    with open('Userlist.txt','a') as file:
        file.writelines(face_id + '\n')

    # sampleNum to count the total number of samples
    count = 0

    while True:
        # get images from camera
        success, img = cap.read()
        # transfer to gary
        if success is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

        # detect face and input each frame in to classifier
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # using a for loop to start a
        for (x, y, w, h) in faces:
            # xy are the coordinates ,w :width，h:height, we use the rectangular
            cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
            # number of sample plus one
            count += 1
            # save image
            cv2.imwrite("data/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
            # show image
            cv2.imshow('image', img)
        # press q to exit
        k = cv2.waitKey(1)
        print(str(count/10)+"%completed")
        if k == '27':
            break
        # if we got 500 samples exit
        elif count >= 1000:
            break

    # close the camera
    cap.release()
    cv2.destroyAllWindows()