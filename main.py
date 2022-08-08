import cv2
import numpy as np
import utils

path = 'Resources/1.png'
width_img = 700
height_img = 700


img = cv2.imread(path)
img = cv2.resize(img, (width_img, height_img))
img_countours = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5),1)
img_canny = cv2.Canny(img_blur,10, 50)

countours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_countours,countours,-1,(0,255,0),10)

img_blank = np.zeros_like(img)
img_array = ([img, img_gray, img_blur,img_canny],
             [img_countours,img_blank,img_blank,img_blank])
img_stacked = utils.stack_images(img_array, 0.5)


cv2.imshow('Stacked images', img_stacked)
cv2.waitKey(0)