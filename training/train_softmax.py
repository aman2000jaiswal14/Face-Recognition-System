from sklearn.model_selection import KFold

import os
import numpy as np
from embedding.softmax import SoftMax
from embedding.faces_embedding import GenerateFaceEmbedding

class TrainFaceRecogModel:
    def __init__(self,X,y,num_classes):
        self.X = X
        self.y = y
        self.input_shape = self.X[0].shape
        self.num_classes = num_classes
        self.Epochs = 10
        self.BatchSize = 8


        genFaceEmbdng = GenerateFaceEmbedding()
        self.X,self.y,self.num_classes = genFaceEmbdng.genFaceEmbedding()
        self.input_shape = self.X[0].shape




    def trainKerasModelForFaceRecognition(self):
        softmax_model = SoftMax(self.input_shape, self.num_classes)
        model = softmax_model.build()

        cv = KFold(n_splits=2, random_state=42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        for train_idx, valid_idx in cv.split(self.X,self.y):

            X_train, X_val, y_train, y_val = self.X[train_idx], self.X[valid_idx], self.y[train_idx],self.y[valid_idx]
            his = model.fit(X_train, y_train, batch_size=self.BatchSize,epochs=self.Epochs, verbose=1,
                                validation_data=(X_val, y_val))
            print(his.history['acc'])

            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']
        model.save("../predicting/model.h5")


# '''
X = np.random.randn(10,10)
y = [1,2,3,4,5,6,7,8,9,10]
tr = TrainFaceRecogModel(X,y,10)
tr.trainKerasModelForFaceRecognition()
# '''