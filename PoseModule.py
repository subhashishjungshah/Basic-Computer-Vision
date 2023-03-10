import cv2
import mediapipe as mp
import math

class PoseDetector():
    def __init__(self,static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity,self.smooth_landmarks, self.enable_segmentation,self.smooth_segmentation,self.min_detection_confidence,self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                pass
                # self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img,flag=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # This gives the pixel value
                self.lmList.append([id, cx, cy])
                if flag:
                    pass
                    # cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw=True):
        # Getting the landmarks of three points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        #calculate the angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        # print(angle)
        if angle < 0:
            angle += 360
        #draw
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x2,y2),(x3,y3),(255,255,255),3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255) , cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

        return angle
