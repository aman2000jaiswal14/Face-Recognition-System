import cv2
import numpy as np
import os


def show_webcam(obj_name = 'Aman',width=600, height=600):
    num_of_train_images = 5
    img_counter = 0
    cam = cv2.VideoCapture(0)

    folderName = "../Image_Data/training_data"+"/"+obj_name
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('my webcam', width, height)
        img_counter += 1

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
            break  # esc to quit

    cv2.destroyAllWindows()

# show_webcam()