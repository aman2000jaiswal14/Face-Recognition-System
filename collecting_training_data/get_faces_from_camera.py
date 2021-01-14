import sys
import os

from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
from datetime import datetime

MAX_FACES = 50
imagePath = os.path.join("..","dataset","train","Aman")

class TrainingDataCollector:
    def __init__(self):
        self.detector = MTCNN()
    def preprocess(self,img,box):
        res_img = cv2.resize(img, (640, 480))
        bboxes = self.detector.detect_faces(res_img)
        max_area = 0
        max_bbox = 0
        flg = False
        if (len(bboxes) != 0):
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                keypoints = bboxe["keypoints"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if area > max_area:
                    max_bbox = bbox
                    landmarks = keypoints
                    max_area = area

            max_bbox = max_bbox[:4]
            print(box,max_bbox)
            flg = True
            return res_img[max_bbox[1]:max_bbox[3],max_bbox[0]:max_bbox[2],:],flg
        return None,flg

    def collectImageFromCamera(self):
        cap = cv2.VideoCapture(0)

        faces = 0
        frames = 0

        max_faces = MAX_FACES
        max_bbox = np.zeros(4)

        if( not os.path.exists(imagePath)):
            os.makedirs(imagePath)

        while(faces<max_faces):
            ret,frame = cap.read()
            frames+=1

            dtString = str(datetime.now().microsecond)
            bboxes = self.detector.detect_faces(frame)



            if(len(bboxes)!=0):
                max_area = 0

                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[:4]

                if frames%3==0:
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])


                    landmarks = landmarks.reshape((2, 5)).T
                    nimg,flg = self.preprocess(frame, max_bbox)
                    # nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
                    if(flg):
                        cv2.imwrite(os.path.join(imagePath, "{}.jpg".format(dtString)), nimg)
                        cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)

                    faces += 1

            cv2.imshow("Face detection",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("SUCCESS")
        cap.release()
        cv2.destroyAllWindows()

# '''
collector = TrainingDataCollector()
collector.collectImageFromCamera()
# '''