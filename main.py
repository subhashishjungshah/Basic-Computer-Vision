import cv2
import mediapipe as mp
import time

# it uses the first camera of the pc
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
currentTime = 0


# Continuously captures the frame
while True:
    # Reads the frame and stores in variable image. Whether the frame was read or not is stored as boolean in success
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # send imgRGB into hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detects the hand from our frames
    results = hands.process(imgRGB)

    # for each hand it draws the landmarks and connection
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h,w,c = img.shape
                # converting decimal values into pixel
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS calculation
    currentTime = time.time()
    fps = 1/(currentTime-pTime)
    pTime = currentTime

    # Putting the fps value into our image window
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
    # displays the frame in new window
    cv2.imshow("Image", img)
    # it returns the ascii value of keyboard pressed by the user
    cv2.waitKey(1)
