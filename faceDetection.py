# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:03:13 2018

@author: Kishore1
"""

## OpenCV program to detect face in real time 
## import libraries of python OpenCV 
## where its functionality resides 
#import cv2 
#
## load the required trained XML classifiers 
## https://github.com/Itseez/opencv/blob/master/ 
## data/haarcascades/haarcascade_frontalface_default.xml 
## Trained XML classifiers describes some features of some 
## object we want to detect a cascade function is trained 
## from a lot of positive(faces) and negative(non-faces) 
## images. 
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
#
## https://github.com/Itseez/opencv/blob/master 
## /data/haarcascades/haarcascade_eye.xml 
## Trained XML file for detecting eyes 
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
#
## capture frames from a camera 
#cap = cv2.VideoCapture(0) 
#
## loop runs if capturing has been initialized. 
#while 1: 
#
#	# reads frames from a camera 
#	ret, img = cap.read() 
#
#	# convert to gray scale of each frames 
#	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#
#	# Detects faces of different sizes in the input image 
#	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
#
#	for (x,y,w,h) in faces: 
#		# To draw a rectangle in a face 
#		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
#		roi_gray = gray[y:y+h, x:x+w] 
#		roi_color = img[y:y+h, x:x+w] 
#
#		# Detects eyes of different sizes in the input image 
#		eyes = eye_cascade.detectMultiScale(roi_gray) 
#
#		#To draw a rectangle in eyes 
#		for (ex,ey,ew,eh) in eyes: 
#			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
#
#	# Display an image in a window 
#	cv2.imshow('img',img) 
#
#	# Wait for Esc key to stop 
#	k = cv2.waitKey(30) & 0xff
#	if k == 27: 
#		break
#
## Close the window 
#cap.release() 
#
## De-allocate any associated memory usage 
#cv2.destroyAllWindows() 



import cv2
import os
from matplotlib import pyplot as plt

def find_faces(image_path):
    imgtest = cv2.imread(image_path,cv2.IMREAD_COLOR)
    # Make a copy to prevent us from modifying the original
    color_img = imgtest.copy()
    #filename = os.path.basename(image_path)
    # OpenCV works best with gray images
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    # Use OpenCV's built-in Haar classifier
    haar_classifier = cv2.CascadeClassifier('/Users/nageshsinghchauhan/Documents/projects/bitbucket/face_recognition/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/nageshsinghchauhan/Documents/projects/bitbucket/face_recognition/haarcascade_eye.xml')
    faces = haar_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    print('Number of faces found: {faces}'.format(faces=len(faces)))
    for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = gray_img[y:y+h, x:x+w]        
        plt.imshow(face_detect)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
            plt.imshow(eye_detect)
    
if __name__ == '__main__':
    find_faces('/Users/nageshsinghchauhan/Documents/projects/bitbucket/face_recognition/Nagesh.jpg')
