import cv2
import numpy as np
import utils

path = 'Resources/1.png'
width_img = 700
height_img = 700
questions = 5
choices = 5


img = cv2.imread(path)
img = cv2.resize(img, (width_img, height_img))
img_contours = img.copy()
img_biggest_cons = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5),1)
img_canny = cv2.Canny(img_blur,10, 50)


contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_contours,contours,-1,(0,255,0),10)

rect_con = utils.rect_contour(contours)
biggest_contour = utils.get_corner_points(rect_con[0])
grade_points = utils.get_corner_points(rect_con[1])


if biggest_contour.size != 0 and grade_points.size != 0:
    cv2.drawContours(img_biggest_cons,biggest_contour,-1,(0,255,0),20)
    cv2.drawContours(img_biggest_cons, grade_points,-1, (255,0,0),20)
    
    biggest_contour = utils.reorder(biggest_contour)
    grade_points = utils.reorder(grade_points)
    pt1 = np.float32(biggest_contour)
    pt2 = np.float32([[0, 0],[width_img,0],[0,height_img],[width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    img_warp_colored = cv2.warpPerspective(img, matrix, (width_img, height_img))
    
    
    pt1_grade = np.float32(grade_points)
    pt2_grade = np.float32([[0, 0],[325,0],[0,150],[325, 150]])
    matrix_grade = cv2.getPerspectiveTransform(pt1_grade, pt2_grade)
    img_grade_display = cv2.warpPerspective(img, matrix_grade, (325, 150))
    # cv2.imshow("Grade", img_grade_display)
    
    img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(img_warp_gray,170,255,cv2.THRESH_BINARY_INV)[1]
    
    boxes = utils.split_boxes(img_thresh)    
    # cv2.imshow("test", boxes[2])
    
    my_pixel_val = np.zeros((questions, choices))
    count_cols = 0
    count_rows = 0
    for image in boxes:
        total_pixels = cv2.countNonZero(image)
        my_pixel_val[count_rows][count_cols] = total_pixels
        count_cols += 1
        if (count_cols == choices):
            count_rows += 1
            count_cols = 0
    print(my_pixel_val)
    
    
    
img_blank = np.zeros_like(img)
img_array = ([img, img_gray, img_blur,img_canny],
             [img_contours,img_biggest_cons,img_warp_colored,img_thresh])
img_stacked = utils.stack_images(img_array, 0.5)


cv2.imshow('Stacked images', img_stacked)
cv2.waitKey(0)