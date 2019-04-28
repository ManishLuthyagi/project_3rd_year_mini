# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:20:05 2019
@author: manishluthyagi
"""

import cv2 as cv
import pickle as pckl
import numpy as nmpy
from keras.preprocessing import img
import timeit 



print("\n\tCHOOSE YOUR OPTION \n\n 1.)\tPOSE DETECTION \n 2.)\tFACE RECOGNITION " )
category = input().upper()

if category == "Face RECOGNITION":
    while True:
    
        file_model = open('classifier1.pickle','rb')
        cnn_model = pckl.loads(file_model.read())
        file_model.close()
       
        Casc_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        captured_frame = cv.VideoCapture(0)
    
        bool_val, img_input = captured_frame.read()
        
        img_gray = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
         
        face_set = casc_classifier.detectMultiScale( img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE )
    
        for (x1, y1, x2, y2) in face_set:
            cv.rectangle(img_input, (x1, y1), (x1+x2, y1+y2), (0, 255, 255), 1)
    
            img_resized = cv.resize(frame,(64,64))
            img_array = img.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis = 0)
            predict_name = cnn_model.predict(img_array)
            if predict[0][0] >= 0.5:
                person_name = 'udit'
            else predict[0][0] < 0.5:
                person_name = 'manish'
    
            cv.putText(img_input, , (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
    
        # Display the resulting frame
        cv.imshow('FACE RECOGNITION', img_input)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    
    
else category == "POSE DETECTION:
    model_TP = cv.dnn.readNetFromTensorflow("tensor_pose.pb")
    Body_joints = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    
    joints_pairs = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


    Width = 368
    Height = 368

    frame_capture = cv.VideoCapture(0)

    while True:

        bool_val, frame1 = frame_capture.read()
        
        frameWidth = frame1.shape[1]
        frameHeight = frame1.shape[0]
        
        model_TP.setInput(cv.dnn.blobFromImage(frame1, 1.0, (Width, Height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        
        out_data = model.forward()
        out_data = out_data[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        
        assert(len(Body_joints) == out_data.shape[1])
        points = []
        
        for i in range(len(Body_joints)):
         
            heatMap = out[0, i, :, :]
            
         _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > 0.25 else None)

        for pair in joints_pairs:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in Body_joints) 
            assert(partTo in Body_joints)

        idFrom = Body_joints[partFrom]
        idTo = Body_joints[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame1, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame1, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame1, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


        t, _ = model.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.imshow('POSE DETECTION', frame1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture

frame_capture.release()
cv.destroyAllWindows()
