# in this file, we would train a cv2 inbuild model on the faces (training images) we get from opencv.py file

import cv2
import os
import numpy as np
from PIL import Image

def train_data():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # this recognizer can train on the input images
    path = 'C:/python programmes/streamlit/face_detection_test/dataset'

    def getImagesWithId(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] # merging the path and file name
        faces = []
        IDs = []

        for imagepath in imagePaths: # for every image in the dataset folder
            faceImg = Image.open(imagepath).convert("L") # opening the images in dataset
            faceNp = np.array(faceImg, "uint8") # converting the image into a numpy array
            print(imagepath)
            ID = int(os.path.split(imagepath)[-1].split(".")[1]) # splitting the image path to get the user id from the filename(imagepath)
            faces.append(faceNp)
            IDs.append(ID)
            cv2.waitKey(10)

        return np.array(IDs), faces

    Ids, faces = getImagesWithId(path) # getting the faces array and id from above function
    recognizer.train(faces, Ids) # training the recognizer on the above faces and ids
    recognizer.save("C:/python programmes/streamlit/face_detection_test/trainingData.yml")
    cv2.destroyAllWindows()

