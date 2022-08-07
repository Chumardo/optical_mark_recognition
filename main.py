import cv2
import numpy as np


path = 'Resources/1.png'
width_img = 700
height_img = 700


img = cv2.imread(path)
img = cv2.resize(img, (width_img, height_img))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5),1)
img_canny = cv2.Canny(img_blur,10, 50)




cv2.imshow('Original', img)
cv2.waitKey(0)