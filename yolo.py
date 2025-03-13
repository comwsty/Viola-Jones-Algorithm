from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "kkk.jpg"
image = cv2.imread(image_path)

orig_h, orig_w = image.shape[:2]

results = model(image)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        label = model.names[int(box.cls[0])]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)  # Сделал текст крупнее


screen_w = 1920
screen_h = 1080

if orig_w > screen_w or orig_h > screen_h:
    scale = min(screen_w / orig_w, screen_h / orig_h) * 0.95
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
else:
    resized_image = image

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Detection", resized_image)

cv2.moveWindow("Detection", (screen_w - resized_image.shape[1]) // 2, (screen_h - resized_image.shape[0]) // 2)

cv2.waitKey(0)
cv2.destroyAllWindows()
