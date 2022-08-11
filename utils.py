import cv2
import numpy as np 

def stack_images(img_array, scale, labels = []):
    rows = len(img_array)
    columns = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range (0, rows):
            for y in range (0, columns):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range (0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        hor_con = np.concatenate(img_array)
        ver = hor
    if len(labels) != 0:
        each_img_width = int(ver.shape[1] / columns)
        each_img_height = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range (c, columns):
                cv2.rectangle(ver,(c*each_img_width,each_img_height*d),(c*each_img_width+len(labels[d][c])*13+27,30+each_img_height*d),(255, 255, 255),cv2.FILLED)
                cv2.putText(ver, labels[d][c],(each_img_width*c+10,each_img_height*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rect_contour(contours):
    
    rect_con = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if len(approx) == 4:
                rect_con.append(i)
                
    rect_con = sorted(rect_con, key = cv2.contourArea,reverse=True)

    return rect_con


def get_corner_points(cont):
    peri = cv2.arcLength(cont,True)
    approx = cv2.approxPolyDP(cont,0.02*peri,True)
    return approx


def reorder(my_points):
    my_points = my_points.reshape((4,2))
    my_points_new = np.zeros((4,1,2),np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)] #[0, 0]
    my_points_new[3] = my_points[np.argmax(add)] #[w, height]
    diff = np.diff(my_points,axis=1)
    my_points_new[1] = my_points[np.argmin(diff)] #[width, 0]
    my_points_new[2] = my_points[np.argmax(diff)] #[0, height]
    
    return my_points_new


def split_boxes(img):
    rows = np.vsplit(img,5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
            
    return boxes

def show_answers(img,my_index,grading,ans,questions,choices):
    sec_width = img.shape[1]/questions
    sec_height = img.shape[0]/choices
    
    for x in range(0,questions):
        my_answer = my_index[x]
        #Finding Center value of given box
        c_x = int((my_answer*sec_width)+sec_width/2)
        c_y = int((x*sec_height)+sec_height/2)
        if grading[x] == 1:
            my_color = (0,255,0)
        else:
            my_color = (0, 0, 255)
            correct_ans = ans[x]
            x_for_circle = int((correct_ans*sec_width)+sec_width//2)
            y_for_circle = int((x*sec_height)+sec_height//2)
            cv2.circle(img,(x_for_circle,y_for_circle),30,(0,255,0),cv2.FILLED)
        
        cv2.circle(img,(c_x,c_y),50,my_color,cv2.FILLED)
    return img