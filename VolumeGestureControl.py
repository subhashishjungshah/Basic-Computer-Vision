import cv2
import numpy as np
import math
import HandDetectionModule as hdm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVolume = volRange[0]
maxVolume = volRange[1]


cap = cv2.VideoCapture(0)
wCam,hCam = 640,480

cap.set(3,wCam)
cap.set(4,hCam)

handDetector = hdm.HandDetector(detConf=0.7)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = handDetector.findHands(img)
    lmlist = handDetector.findPosition(img,draw=False)
    if len(lmlist) > 0:
        # print(lmlist[4],lmlist[8])
        # capturing the target fingure which is landmark 4 and 8
        x1,y1 = lmlist[4][1] , lmlist[4][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]

        # getting the center of the line
        cx,cy = (x1+x2) // 2 , (y1+y2) // 2

        # calculates the lenegth between two figures using pythagorus theorem
        length = math.hypot(x2-x1,y2-y1)
        print(length)

        # our hand range was from 15 - 250
        vol = np.interp(length, [15,250],[minVolume,maxVolume])
        print(int(vol))
        volume.SetMasterVolumeLevel(int(vol), None)
        # drawing circle on the captured landmark
        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        # drawing line between them
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        if(length<15):
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Volume Control", img)
    cv2.waitKey(1)
