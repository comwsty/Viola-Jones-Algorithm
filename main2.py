import cv2
import screeninfo

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

filename = 'pic4.jpg'
dog_filter = cv2.imread('C:\\Users\\Admin\\PycharmProjects\\pythonProject18\\filter_wws\\dog.png')

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))

    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i, j, k] < 235:
                    y_offset = y + i - int(0.375 * h)
                    x_offset = x + j - int(0.35 * w)

                    if 0 <= y_offset < fc.shape[0] and 0 <= x_offset < fc.shape[1]:
                        fc[y_offset, x_offset, k] = dog[i, j, k]

    return fc

faces = face_cascade.detectMultiScale(gray, 1.09, 7)
for (x, y, w, h) in faces:
    img = put_dog_filter(dog_filter, img, x, y, w, h)

screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

img_height, img_width = img.shape[:2]

scale = min(screen_width / img_width, screen_height / img_height, 1.0)
new_width = int(img_width * scale)
new_height = int(img_height * scale)

if scale < 1.0:
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
else:
    img_resized = img

cv2.namedWindow('Dog Filter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Dog Filter', new_width, new_height)  # Размер окна по масштабу изображения

cv2.imshow('Dog Filter', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

