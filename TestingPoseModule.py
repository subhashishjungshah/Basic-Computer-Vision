import cv2
import PoseModule as pm

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    posedetector = pm.PoseDetector()
    img = posedetector.findPose(img, True)
    cv2.imshow("Pose Detection", img)
    cv2.waitKey(1)