from keras.models import load_model
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

class GenerateFaceEmbedding:
    def __init__(self):
        self.filename = os.path.join("..","dataset","train")
        self.class_name = os.listdir(self.filename)

        self.model_file = 'facenet_keras.h5'
        self.model  = load_model(self.model_file)

        self.X = []
        self.y = []

    def read_images(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 160))

        return img


    def load_data(self):
        X_file = []
        y_file = []

        for num,cls in enumerate(self.class_name):
            class_path = os.path.join(self.filename,cls)
            class_images = os.listdir(class_path)
            for img_file in class_images:
                img_path = os.path.join(class_path,img_file)
                X_file.append(self.read_images(img_path))

                y_file.append(num)
        return X_file,y_file

    def get_embedding(self,pixel):
        mean,std = pixel.mean(),pixel.std()
        pixel = (pixel-mean)/std
        pixel = np.expand_dims(pixel,axis=0)
        pred = self.model.predict(pixel)

        return pred[0]



    def genFaceEmbedding(self):
        X,y = self.load_data()
        self.y = y
        for x in X:
            self.X.append(self.get_embedding(x))

        return np.array(self.X),np.array(self.y),len(self.class_name)




# '''
embd = GenerateFaceEmbedding()
x,y,_ = embd.genFaceEmbedding()
print(x.shape,y.shape)
# '''
