import tensorflow
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

def predict_images():
    if(not os.path.exists("../training_on_images/model1.h5")):
        print("model not found.------------------------------")
        return
    model = load_model("../training_on_images/model2.h5")
    print("model_loaded--------------------------------------")
    cam = cv2.VideoCapture(0)
    width = 600
    height = 600
    name = ""

    class_list = os.listdir('../Image_Data/training_data')
    class_dict={}
    for i,name in enumerate(class_list):
        class_dict[i] = name
    print(class_dict)

    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('my webcam', width, height)
        img_height = 224
        img_width = 224
        img = cv2.resize(img, (img_height, img_width))
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        #with open('file1.txt','w') as f:
        #    f.write(str(img))
        pred = model.predict(img)
        n_name = class_dict[np.argmax(pred[0])]
        # print(n_name,pred)
        if(n_name!=name):
            print(n_name)
            name=n_name
        if cv2.waitKey(1)  & 0XFF == ord('q'):
            break
    print("ended")
    cv2.destroyAllWindows()
predict_images()