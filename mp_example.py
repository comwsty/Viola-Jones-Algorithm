import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_result = hands.process(frame_rgb)
    face_result = face_detection.process(frame_rgb)
    pose_result = pose.process(frame_rgb)

    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Mediapipe example", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()