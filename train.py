import pdb
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import matplotlib.pyplot as plt
batch_size = 32
img_height = 224
img_width = 224

def preprocessing(full_path):
    img = plt.imread(full_path)
    img = img/255.
    '''
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
    # img = img / 255.
    '''
    img = np.expand_dims(img,axis=0)
    return img

def training_model():

    data_dir = "../Image_Data/training_data"
    class_names = os.listdir(data_dir)
    img_lists = []
    classes = []
    class_count = 0
    print(class_names)
    for i in class_names:
        path = os.path.join(data_dir,i)
        for j in os.listdir(path):
            full_path = os.path.join(path,j)
            classes.append(class_count)
            img = preprocessing(full_path)

            if(img_lists==[]):
                img_lists=img
            else:
                img_lists = np.append(img_lists,img,axis=0)
        class_count+=1
    num_classes = len(class_names)
    targets = np.array([np.eye(num_classes)[i] for i in classes])
    id = np.random.randint(low=0, high=len(img_lists), size=len(img_lists))
    img_lists = img_lists[id]
    targets = targets[id]

    model = Sequential([
        # layers.Conv2D(16,7,2,input_shape=(img_width,img_height,3),activation='relu'),
        # layers.Conv2D(32,5,2,activation='relu'),
        # layers.MaxPool2D(),
        #
        # layers.Conv2D(128,5,2,activation='relu'),
        # layers.MaxPool2D(),
        #
        # layers.Conv2D(256, 3,1, activation='relu'),
        # layers.MaxPool2D(),
        tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_width,img_height,3)),

        layers.Flatten(),
       layers.Dense(512, activation='relu'),
        layers.Dense(num_classes,activation='softmax')
        ])
    model.layers[0].trainable=False


    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print("training_images--------------------------------------------------")
    epochs=10
    history = model.fit(
        img_lists,
        targets,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
        )
    print("model_created")
    model.save('model2.h5')

training_model()