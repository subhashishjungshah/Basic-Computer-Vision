import cv2
import mediapipe as mp
import time
import HandDetectionModule as hdm
pTime = 0
currentTime = 0
cap = cv2.VideoCapture(0)
handdetector = hdm.HandDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = handdetector.findHands(img)
    lmlist = handdetector.findPosition(img,0,True)
    if(len(lmlist)>0):
        print(lmlist[0])
    currentTime = time.time()
    fps = 1 / (currentTime - pTime)
    pTime = currentTime
    # Putting the fps value into our image window
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    # displays the frame in new window
    cv2.imshow("Image", img)
    # it returns the ascii value of keyboard pressed by the user
    cv2.waitKey(1)

