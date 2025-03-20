import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

dog_filter = cv2.imread('C:\\Users\\Admin\\PycharmProjects\\pythonProject18\\filter_wws\\dog.png')


def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))

    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    if 0 <= y + i - int(0.375 * h) < fc.shape[0] and 0 <= x + j - int(0.35 * w) < fc.shape[1]:
                        fc[y + i - int(0.375 * h)][x + j - int(0.35 * w)][k] = dog[i][j][k]

    return fc


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.09, 7)

    for (x, y, w, h) in faces:
        frame = put_dog_filter(dog_filter, frame, x, y, w, h)

    cv2.imshow('Dog Filter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()