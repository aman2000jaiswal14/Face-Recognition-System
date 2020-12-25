import cv2
import numpy as np
import os

img_height = 224
img_width = 224
def preprocessing(img):
    nid = np.random.randint(0,8)
    if(nid==0):
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    if(nid==2):
        img = cv2.rotate(img, cv2.ROTATE_180)
    if(nid==4):
        img = img[50:]
    if(nid==5):
        img = img[:50]

    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.*255.

    return img


def show_webcam(obj_name = 'Aman',width=600, height=600):
    num_of_train_images = 50
    img_counter = 0
    cam = cv2.VideoCapture(0)

    folderName = "../Image_Data/training_data"+"/"+obj_name
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('my webcam', width, height)
        img_counter += 1
        img = preprocessing(img)
        # -----------------------------------
        if(not os.path.exists(folderName)):
            os.mkdir(folderName)
        img_name = "img"+"{}".format(img_counter)+".jpg"
        filename = os.path.join(folderName,img_name)
        cv2.imwrite(filename,img)
        # -------------------------------------

        if(img_counter==num_of_train_images):
            break
        if cv2.waitKey(1)  & 0XFF == ord('q'):
            break

    cv2.destroyAllWindows()

show_webcam('Glass')