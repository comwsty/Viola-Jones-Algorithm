import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

#  изображения
target_frame = cv2.imread('dean.jpg')
src_frame = cv2.imread('johnny.jpg')

# проверка
if target_frame is None or src_frame is None:
    raise ValueError("Ошибка загрузки изображений. Проверь названия файлов!")

# FaceAnalysis
providers = ["CPUExecutionProvider"]
FACE_ANALYSER = FaceAnalysis(name="buffalo_l", root=".", providers=providers,
                             allowed_modules=["detection", "recognition"])
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

# распознование
src_faces = FACE_ANALYSER.get(src_frame)
target_faces = FACE_ANALYSER.get(target_frame)

# проверка
if not src_faces or not target_faces:
    raise ValueError("Не удалось обнаружить лицо на одном из изображений")

# модель
model_path = './models/inswapper_128.onnx'
model_swap = insightface.model_zoo.get_model(model_path, download=True, providers=providers)

# замена лица
img_fake = model_swap.get(img=target_frame, target_face=target_faces[0], source_face=src_faces[0], paste_back=True)

# результат
plt.imshow(cv2.cvtColor(img_fake, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# сохранение
cv2.imwrite('result2.jpg', img_fake)
