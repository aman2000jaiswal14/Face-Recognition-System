import os
import cv2
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from mtcnn.mtcnn import MTCNN

class FacePredictor:
    def __init__(self):
        self.class_name = os.listdir(os.path.join("..","dataset","train"))
        self.class_to_n_dict = {"NONE":0}
        self.n_to_class_dict = {0:"NONE"}

        for n,c in enumerate(self.class_name):
            self.class_to_n_dict[c] = n+1
            self.n_to_class_dict[n+1] = c

        self.detector = MTCNN()

        self.model = Sequential()
        self.facenet_model = load_model(os.path.join("..","embedding","facenet_keras.h5"))
        self.facenet_model.trainable = False
        self.softmax_model = load_model(os.path.join("model.h5"))
        self.softmax_model.trainable = False
        self.model.add(self.facenet_model)
        self.model.add(self.softmax_model)


    def preprocess(self,img,box):
        flg = False
        if (len(box) != 0):
            max_bbox = box[:4]
            flg = True
            return img[max_bbox[1]:max_bbox[3],max_bbox[0]:max_bbox[2],:],flg
        return None,flg


    def detectFace(self):
        cap = cv2.VideoCapture(0)

        while(1):
            ret,frame = cap.read()



            max_bbox = np.zeros(4)
            bboxes = self.detector.detect_faces(frame)

            if (len(bboxes) != 0):
                max_area = 0

                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                nimg, flg = self.preprocess(frame, max_bbox)
                if (flg):
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)

            pixel = cv2.resize(frame, (160, 160))
            mean, std = pixel.mean(), pixel.std()
            pixel = (pixel - mean) / std
            pixel = np.expand_dims(pixel, axis=0)
            pred = self.model.predict(pixel)[0]
            pid = np.argmax(pred)
            val = 0
            print(pred)

            if(pred[pid]>0.4):
                val = pid+1
            print(self.n_to_class_dict, "                  ", val)
            # for num,p in enumerate(pred[0]):
            class_pred = self.n_to_class_dict[val]
            text = class_pred
            cv2.putText(frame, text, (250, 50 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 1)


            cv2.imshow("Face detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()


faceDetector = FacePredictor()
faceDetector.detectFace()