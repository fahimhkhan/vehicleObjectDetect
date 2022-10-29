# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:29:08 2022

@author: Fahim
"""

import numpy as np
import pathlib
# use "pip install tensorflow==1.15" for installation
import tensorflow as tf
# use "pip install opencv-python" for installation
import cv2

import os
import sys
import pathlib
import shutil
import time

# use "pip install PyQt5" for installation
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
#from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QInputDialog, QLineEdit, QComboBox, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QFileDialog

vehicleObj = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter"]

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Main Menu")
        
        uic.loadUi("vehicleObjectDetect.ui", self)
        
        self.webcamButton.clicked.connect(self.button1_clicked)
        self.videofileButton.clicked.connect(self.button2_clicked)
        
    def button1_clicked(self):
        self.message.setPlainText("Processing video feed from webcam and showing preview. Saving output video to the directory outputVideos. Click on preview window or press any key to stop.")
        runML(0)
        
    def button2_clicked(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "","Video files (*.mp4 *.avi *.mov)")
        print(fname[0])
        self.message.setPlainText("Processing video file " + fname[0] + " and showing preview. Saving output video to the directory outputVideos. Click on preview window or press any key to stop.")
        runML(fname[0])
 
def load_labels(filename):
    my_labels = []
    input_file = open(filename, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels

def draw_rect(image, box, label_string, conf_score):
    h, w, _ = image.shape
    y_min = int(max(1, (box[0] * h)))
    x_min = int(max(1, (box[1] * w)))
    y_max = int(min(h, (box[2] * h)))
    x_max = int(min(w, (box[3] * w)))

    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    label = label_string + ": {:.2f}%".format(conf_score * 100)
    cv2.putText(image, label, (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def runML(source):
    interpreter = tf.lite.Interpreter(model_path="models/mobilenetV2.tflite")
    labels = load_labels("models/labels.txt")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    interpreter.allocate_tensors()

    ts = time.time()

    # cap = cv2.VideoCapture("inputs/waymo.avi")
    cap = cv2.VideoCapture(source)
    cv2.namedWindow('PreviewWindow')
    cv2.setMouseCallback('PreviewWindow', onMouse)

    try:
        if not os.path.exists('outputVideos'):
            os.makedirs('outputVideos')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of outputVideos')

    dest = "outputVideos/" + str(ts) + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dest, fourcc, 20.0, (1280, 720))

    frame = 0
    print('Showing preview. Click on preview window or press any key to stop.')
    ret, img = cap.read()
    while cv2.waitKey(1) is -1 and not clicked:
        ret, img = cap.read()
        if ret == True:
            new_img = cv2.resize(img, (300, 300))
            interpreter.set_tensor(input_details[0]['index'], [new_img])

            interpreter.invoke()
            rects = interpreter.get_tensor(output_details[0]['index'])
            classes = interpreter.get_tensor(output_details[1]['index'])
            scores = interpreter.get_tensor(output_details[2]['index'])

            for index, score in enumerate(scores[0]):
                idx = int(classes[0][index]) + 1
                lbl = labels[idx]
                if score > 0.5 and lbl in vehicleObj:
                    # print(ctr, file.name, score, lbl)
                    draw_rect(img, rects[0][index], lbl, score)

            output_img = cv2.resize(img, (1280, 720))
            out.write(output_img)
            cv2.imshow('PreviewWindow', output_img)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
