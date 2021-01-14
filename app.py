import tkinter as tk
from tkinter import *
import tkinter.font as font
import webbrowser
import logging

from collecting_training_data.get_faces_from_camera import TrainingDataCollector
from training.train_softmax import TrainFaceRecogModel
from predicting.facePredictor import FacePredictor
from embedding.faces_embedding import GenerateFaceEmbedding


class RegistrationModule:
    def __init__(self, logFileName):

        self.logFileName = logFileName
        self.window = tk.Tk()
        # helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
        self.window.title("Face Recognition and Tracking")

        # this removes the maximize button
        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        # window.geometry('880x600')
        self.window.configure(background='#ffffff')

        # window.attributes('-fullscreen', True)

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        header = tk.Label(self.window, text="Employee Monitoring Registration", width=80, height=2, fg="white", bg="#363e75",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)
        clientID = tk.Label(self.window, text="Client ID", width=10, height=2, fg="white", bg="#363e75", font=('times', 15))
        clientID.place(x=80, y=80)

        displayVariable = StringVar()
        self.clientIDTxt = tk.Entry(self.window, width=20, text=displayVariable, bg="white", fg="black",
                               font=('times', 15, 'bold'))
        self.clientIDTxt.place(x=205, y=80)

        empID = tk.Label(self.window, text="EmpID", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empID.place(x=450, y=80)

        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=80)

        empName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empName.place(x=80, y=140)

        self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empNameTxt.place(x=205, y=140)

        emailId = tk.Label(self.window, text="Email ID :", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        emailId.place(x=450, y=140)

        self.emailIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.emailIDTxt.place(x=575, y=140)

        mobileNo = tk.Label(self.window, text="Mobile No :", width=10, fg="white", bg="#363e75", height=2,
                            font=('times', 15))
        mobileNo.place(x=450, y=140)

        self.mobileNoTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.mobileNoTxt.place(x=575, y=140)

        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911", font=('times', 15))
        self.message.place(x=220, y=220)
        lbl3.place(x=80, y=260)

        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=205, y=260)

        takeImg = tk.Button(self.window, text="Take Images", command=self.collectUserImageForRegistration, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '))
        takeImg.place(x=80, y=350)

        trainImg = tk.Button(self.window, text="Train Images", command=self.trainModel, fg="white", bg="#363e75", width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        trainImg.place(x=350, y=350)

        predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        predictImg.place(x=600, y=350)

        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'))
        quitWindow.place(x=650, y=510)

        link2 = tk.Label(self.window, text="aman2000jaiswal14", fg="blue", )
        link2.place(x=690, y=580)
        # link2.pack()
        link2.bind("<Button-1>", lambda e: self.callback("http://google.com"))
        label = tk.Label(self.window)

        self.window.mainloop()

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')



    def collectUserImageForRegistration(self):
        clientIDVal = (self.clientIDTxt.get())
        empIDVal = self.empIDTxt.get()
        name = (self.empNameTxt.get())
        training_data_obj = TrainingDataCollector()
        training_data_obj.collectImageFromCamera()

    def getFaceEmbedding(self):
        genFaceEmbdng = GenerateFaceEmbedding()
        X,y,num_classes = genFaceEmbdng.genFaceEmbedding()
        return X,y,num_classes

    def trainModel(self):
        X,y,num_classes = self.getFaceEmbedding()
        faceRecogModel = TrainFaceRecogModel(X,y,num_classes)
        faceRecogModel.trainKerasModelForFaceRecognition()
    def makePrediction(self):
        faceDetector = FacePredictor()
        faceDetector.detectFace()

    def close_window(self):
        self.window.destroy()

    def callback(self,url):
        webbrowser.open_new(url)

logFileName = "ProceduralLog.txt"
regStrtnModule = RegistrationModule(logFileName)