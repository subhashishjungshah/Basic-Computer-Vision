import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # id gives the landmark position and lm gives the corrdinate
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h) # This gives the pixel value
            cv2.circle(img,(cx,cy),5,(255,0,0), cv2.FILLED)

    cv2.imshow("Pose Detection", img)
    cv2.waitKey(1)
