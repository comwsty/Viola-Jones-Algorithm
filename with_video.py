import cv2
import insightface
from insightface.app import FaceAnalysis

input_video_path = 'video.mp4'  # Укажи путь к своему видео
output_video_path = 'output2.mp4'

src_frame = cv2.imread('scarlet.jpg')
if src_frame is None:
    raise ValueError("Ошибка загрузки изображения-донора. Проверь название файла!")

providers = ["CPUExecutionProvider"]
FACE_ANALYSER = FaceAnalysis(name="buffalo_l", root=".", providers=providers,
                             allowed_modules=["detection", "recognition"])
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

src_faces = FACE_ANALYSER.get(src_frame)
if not src_faces:
    raise ValueError("Не удалось обнаружить лицо на изображении-источнике")

model_path = './models/inswapper_128.onnx'
model_swap = insightface.model_zoo.get_model(model_path, download=True, providers=providers)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError("Ошибка загрузки видео. Проверь путь!")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    target_faces = FACE_ANALYSER.get(frame)

    if target_faces:
        img_fake = model_swap.get(img=frame, target_face=target_faces[0], source_face=src_faces[0], paste_back=True)
    else:
        img_fake = frame
    out.write(img_fake)

cap.release()
out.release()

print(f"✅ Видео сохранено: {output_video_path}")
