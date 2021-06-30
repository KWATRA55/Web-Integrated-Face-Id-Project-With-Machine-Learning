#importing libs
from os import name
from numpy.lib.shape_base import tile
from numpy.lib.type_check import imag
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from opencv import capture_image
from face_trainer import train_data



# creating the classifier (recognizer)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

rec = cv2.face.LBPHFaceRecognizer_create() # this is the image recognizer
rec.read("C:/python programmes/streamlit/face_detection_test/trainingData.yml") # we are using the recodnizer to identify the training faces from the face_trainer.py





# predicting real time video

def video():
    run = st.button('Video Capture')
    closewin = st.button("Close")
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while run:
        _, img = camera.read()  
        
        
            # Convert to grayscale  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        
            # Detect the faces  
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
        
            # Draw the rectangle around each face  
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            id, uncertainty = rec.predict(gray[y: y+h, x: x+w])

            if(uncertainty<53):

                path = 'C:/python programmes/streamlit/face_detection_test/dataset'
                
                imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

                for imagepath in imagePaths:
                    ID = int(os.path.split(imagepath)[-1].split(".")[1])
                    NAME = str(os.path.split(imagepath)[-1].split(".")[0])



                if(id == ID ):
                    name = NAME
                    cv2.putText(img, name, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255))

                
            else:
                cv2.putText(img, "Unknown", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255))


        if closewin:
            break

        FRAME_WINDOW.image(img)
        



# streamlit app

st.title("Face detection App")
options = ("Add a face", "Recognise", "Emotion Detection")
option = st.selectbox("Choose a task", options)


# options ->

# option 1
if option == "Add a face":
    capture_image()
    train_data()

#option 2
elif option == "Recognise":
    video()






















