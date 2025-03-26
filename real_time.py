import cv2
import insightface
from insightface.app import FaceAnalysis

# изображение-донор
src_frame = cv2.imread('tokaev.jpg')

if src_frame is None:
    raise ValueError("Ошибка загрузки изображения-донора. Проверь название файла!")

# FaceAnalysis
providers = ["CPUExecutionProvider"]
FACE_ANALYSER = FaceAnalysis(name="buffalo_l", root=".", providers=providers, allowed_modules=["detection", "recognition"])
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

# распознание лица донора
src_faces = FACE_ANALYSER.get(src_frame)
if not src_faces:
    raise ValueError("Не удалось обнаружить лицо на изображении-источнике")

# модель
model_path = './models/inswapper_128.onnx'
model_swap = insightface.model_zoo.get_model(model_path, download=True, providers=providers)

# веб-камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # распознавание лица
    target_faces = FACE_ANALYSER.get(frame)

    if target_faces:
        img_fake = model_swap.get(img=frame, target_face=target_faces[0], source_face=src_faces[0], paste_back=True)
    else:
        img_fake = frame

    # результат
    cv2.imshow('Face Swap', img_fake)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
