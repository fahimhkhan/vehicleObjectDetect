# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:29:08 2022

@author: Fahim
"""

import numpy as np
import pathlib
# use "pip install tensorflow" for installation
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

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def runML(source):
    detector = tf.saved_model.load("saved_model/0")
    labels = load_labels("saved_model/labelmap.txt")
    # detector = tf.saved_model.load("2")
    # labels = load_labels("2/labels.txt")
    # detector = tf.saved_model.load("1")
    # labels = load_labels("1/labels.txt")

    width = 480
    height = 480

    ts = time.time()

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
    ret, frame = cap.read()
    while cv2.waitKey(1) == -1 and not clicked:
        ret, frame = cap.read()
        if ret == True:
            inp = cv2.resize(frame, (width , height))

            #Convert img to RGB
            #rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            rgb = inp
            #img_boxes = inp

            #Is optional but i recommend (float convertion and convert img to tensor image)
            rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

            #Add dims to rgb_tensor
            rgb_tensor = tf.expand_dims(rgb_tensor , 0)
            
            boxes, scores, classes, num_detections = detector(rgb_tensor)
            
            pred_labels = classes.numpy().astype('int')[0]
            
            pred_labels = [labels[i] for i in pred_labels]
            pred_boxes = boxes.numpy()[0].astype('int')
            pred_scores = scores.numpy()[0]
        
        #loop throughout the detections and place a box around it  
            for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
                if score < 0.5:
                    continue

                h, w, _ = frame.shape
                
                y_min = int(max(1, (ymin * (h/height))))
                x_min = int(max(1, (xmin * (w/width))))
                y_max = int(min(h, (ymax * (h/height))))
                x_max = int(min(w, (xmax * (w/width))))
                    
                rgb = cv2.rectangle(frame,(x_min, y_max),(x_max, y_min),(0,0,255),2)      
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt = label + ": " + ": {:.2f}%".format(score * 100)
                cv2.putText(rgb,txt,(x_min, y_max-10), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                #cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
            
            outp = cv2.resize(rgb, (1280, 720))
            out.write(outp)
            cv2.imshow('PreviewWindow', outp)
            print('Showing preview. Click on preview window or press any key to stop.')
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
