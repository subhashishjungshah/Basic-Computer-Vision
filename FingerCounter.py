import cv2
import os
import HandDetectionModule as hdm
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

folderPath = "Fingers"
myList = os.listdir(folderPath)
print(myList)

listImage = []
for imagepath in myList:
    image = cv2.imread(f'{folderPath}/{imagepath}')
    listImage.append(image)
# print(listImage)
new_width = 200
new_height = 200
detector = hdm.HandDetector(detConf=0.75)
tipIds = [4,8,12,16,20]
while True:
    success, img = cap.read()
    # img = cv2.flip(img,1)
    #this resizes the image size

    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:


        fingers = []
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        img[0:200, 0:200] = cv2.resize(listImage[totalFingers-1], (new_width, new_height))

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)

    cv2.imshow("Finger Counter",img)
    cv2.waitKey(1)