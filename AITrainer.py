import cv2
import numpy as np
import PoseModule as pm
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = pm.PoseDetector()
direction = 0
count = 0
while True:
    success, img = cap.read()
    # img = cv2.imread("./exercise.jpg")
    img = cv2.flip(img,1)
    img = detector.findPose(img, draw=True)
    lmList = detector.getPosition(img, False)
    if lmList !=0:
        #Right arm
        angle = detector.findAngle(img,12,14,16)
        # cv2.putText(img, str(int(angle)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        #left arm
        # detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle,(40,150),(0,100))
        # cv2.putText(img, str(int(per)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        if per == 100:
            if direction == 0:
                count+=0.5
                direction=1
        if per == 0:
            if direction == 1:
                count+=0.5
                direction=0
    print(int(count))
    cv2.putText(img, str(int(count)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("AI Trainer",img)
    cv2.waitKey(1)