# -----detection and out put-----
import cv2


def recognize():
    name = input('\n Please input the name of the person you want to recognize:')
    with open('Userlist.txt', 'r') as file:
        names = file.read().split('\n')
        if name not in names:
            print("You didn't upload your facial information please record your face first!!!")
            return
    # initialization of recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # load model
    recognizer.read('models/' + str(name) + '.yml')

    # load classifier
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # setting font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # setting camera
    cam = cv2.VideoCapture(0)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )
        # make comparison
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _, difference = recognizer.predict(gray[y:y + h, x:x + w])

            # output result with this difference
            if difference < 70:
                result = name
            else:
                result = "unknown"

            # output the result of recognition
            cv2.putText(img, str(result), (x + 5, y - 5), font, 1, (0, 0, 255), 1)

            cv2.imshow('camera', img)
        # press esc to exit
        if cv2.waitKey(100) == 27:
            break

    # release camera
    cam.release()
    cv2.destroyAllWindows()
