# in this file, we would use cv2 to get the training faces

user_data = {}

def capture_image():
    import cv2
    import streamlit as st
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # capture pictures and store it in cam
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # this classiier has been trained to classify faces in an image

    FRAME_WINDOW = st.image([])

    name = st.text_input("Enter Your Name")
    Id = st.number_input("Enter Your Id", min_value=1, max_value=10, value=5, step=1)
    

    # creating a button for capturing training data
    capture = st.button("Capture")
    if capture:
        sampleNum = 0
        while(True):
            ret, img = cam.read()  #cam.read returns 2 values, img that is our image, ret - that is boolean value and we dont need it, we only need the img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # our image was in rgb and we are convering it rn into gray as its easier to detect face in gray format
            faces = detector.detectMultiScale(gray, 1.3, 5) # now we use the detector made earliar from pretrained classifier and it detects faces in our grayscale image
            for (x,y,w,h) in faces:  # it would create a rectangle on the detected face and (x,y) are cordinates of one egde of the rectangle and (w,h) is the width and height of rectangle
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) # these are the cordinates of rectangle on the face in each image [(255,0,0) is the color of the box, 2 is the width of the box]


                # saving the captured face in the dataset folder
                cv2.imwrite("C:/python programmes/streamlit/face_detection_test/dataset/" + name + "." +  str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                sampleNum = sampleNum + 1

            FRAME_WINDOW.image(img)
            
            # break if the sample number is more than 20
            if sampleNum > 20:
                break

        cam.release()
        cv2.destroyAllWindows()

        st.write("Congratulations, {} is added with User ID - {}".format(name, Id))




