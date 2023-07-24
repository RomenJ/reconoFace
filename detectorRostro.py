import cv2
import numpy as np
#github de Gabisol: https://github.com/GabySol/OmesTutorials/blob/master/Detecci%C3%B3n%20de%20Rostros/detectorRostro.py#
#haarcascade front face https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml#

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('oficina12.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30,30),
	maxSize=(200,200))

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()