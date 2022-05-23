import cv2
import numpy as np

import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

class Error(Exception):
    pass

def hough_line(img, edges, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # print(lines.shape)
    # h, i, n = lines.shape
    # points = lines
    # points = np.reshape(h, i*2, n/2)
    # print(points.shape)
    for line in lines:    
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
            cv2.circle(img, (x2,y2), 5, (255,0,0), -1)

    return img


def hough_line_range(img, edges, rho, theta, threshold, min_line_len, max_line_gap, range):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lefts = list()
    rights = list()
    for line in lines:
        for x1, y1, x2, y2 in line:
            xdiff = x2 - x1
            ydiff = y2 - y1

            xavg = (x1 + x2) / 2
            yavg = (y1 + y2) / 2

            if xdiff == 0:
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
                # cv2.circle(img, (x2,y2), 5, (255,0,0), -1)
                if xavg <= (img.shape[1] / 2):
                    lefts.append([xavg, yavg])
                else:
                    rights.append([xavg, yavg])
            else:
                if range[0] < ydiff / xdiff < range[1]:
                    pass
                else:
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    # cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
                    # cv2.circle(img, (x2,y2), 5, (255,0,0), -1)
                    if xavg <= (img.shape[1] / 2):
                        lefts.append([xavg, yavg])
                    else:
                        rights.append([xavg, yavg])
    left_lines = [[],[],[],[],[],[],[],[]]
    right_line = [[],[],[],[],[],[],[],[]]

    for left in lefts:
        if 0 < left[1] < 60:
            left_lines[0].append(left)
        elif 60 < left[1] < 120:
            left_lines[1].append(left)
        elif 120 < left[1] < 180:
            left_lines[2].append(left)
        elif 180 < left[1] < 240:
            left_lines[3].append(left)
        elif 240 < left[1] < 300:
            left_lines[4].append(left)
        elif 300 < left[1] < 360:
            left_lines[5].append(left)
        elif 360 < left[1] < 420:
            left_lines[6].append(left)
        elif 420 < left[1] < 480:
            left_lines[7].append(left)
        else:
            pass

    for right in rights:
        if 0 < right[1] < 60:
            right_line[0].append(right)
        elif 60 < right[1] < 120:
            right_line[1].append(right)
        elif 120 < right[1] < 180:
            right_line[2].append(right)
        elif 180 < right[1] < 240:
            right_line[3].append(right)
        elif 240 < right[1] < 300:
            right_line[4].append(right)
        elif 300 < right[1] < 360:
            right_line[5].append(right)
        elif 360 < right[1] < 420:
            right_line[6].append(right)
        elif 420 < right[1] < 480:
            right_line[7].append(right)
        else:
            pass

    print('left')
    print(left_lines)
    print('right')
    print(right_line)

    return img




def hough_line_range_roi(img, edges, rho, theta, threshold, min_line_len, max_line_gap, out_range):
    # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if len(img.shape) > 2:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    row_size = width / 2
    column_size = height / 6
    roi_img = [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
    lines = list()
    for x in range(2):
        for y in range(6):
            x1 = row_size * (x)
            x2 = row_size * (x+1)
            y1 = column_size * (y)
            y2 = column_size * (y+1)
            vertices = np.array([[(x1, y2),(x1, y1), (x2, y1), (x2, y2)]], dtype=np.int32)
            roi_img[x][y] = region_of_interest(edges, vertices)
            
            # xy = (x, y)
            # xd = str(xy)
            # cv2.imshow(xd, roi_img[x][y])
    roi_img = np.array(roi_img)

    def hough(edges):
        return cv2.HoughLinesP(roi_img[x][y], rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)        
    
    for x in range(2):
        for y in range(6):
            lines.append(hough(roi_img[x][y])) 
    lines = np.array(lines)
    lefts = list()
    rights = list()
    # print(lines.shape)
    for line0 in lines:
        try:
            line1 = line0[0]
            for line in line0:
                for x1, y1, x2, y2 in line:
                    xdiff = x2 - x1
                    ydiff = y2 - y1

                    xavg = (x1 + x2) / 2
                    yavg = (y1 + y2) / 2

                    if xdiff == 0:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
                        cv2.circle(img, (x2,y2), 5, (255,0,0), -1)
                        if xavg <= (img.shape[1] / 2):
                            lefts.append([xavg, yavg])
                        else:
                            rights.append(np.array([xavg,yavg]))
                    else:
                        if out_range[0] <= ydiff / xdiff <= out_range[1]:
                            pass
                        else:
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
                            cv2.circle(img, (x2,y2), 5, (255,0,0), -1)
                            if xavg <= (img.shape[1] / 2):
                                lefts.append([xavg, yavg])
                            else:
                                rights.append(np.array([xavg, yavg]))
        except:
            # print('pass')
            pass
    lefts = np.array(lefts)
    rights = np.array(rights)
    # print(lefts)
    left_lines = [[],[],[],[],[],[]]
    right_lines = [[],[],[],[],[],[]]
    left_line = list()
    right_line = list()

    for left in lefts:
        if 0 <= left[1] <= column_size:
            left_lines[0].append(left)
        elif column_size < left[1] <= column_size*2:
            left_lines[1].append(left)
        elif column_size*2 < left[1] <= column_size*3:
            left_lines[2].append(left)
        elif column_size*3 < left[1] <= column_size*4:
            left_lines[3].append(left)
        elif column_size*4 < left[1] <= column_size*5:
            left_lines[4].append(left)
        elif column_size*5 < left[1] <= column_size*6:
            left_lines[5].append(left)
        else:
            pass

    for right in rights:
        if 0 <= right[1] <= column_size:
            right_lines[0].append(right)
        elif column_size < right[1] <= column_size*2:
            right_lines[1].append(right)
        elif column_size*2 < right[1] <= column_size*3:
            right_lines[2].append(right)
        elif column_size*3 < right[1] <= column_size*4:
            right_lines[3].append(right)
        elif column_size*4 < right[1] <= column_size*5:
            right_lines[4].append(right)
        elif column_size*5 < right[1] <= column_size*6:
            right_lines[5].append(right)
        else:
            pass

    for y in range(6):
        left_lines[y] = np.array(left_lines[y])
    left_lines = np.array(left_lines)
    for y in range(6):
        left_lines[y] = left_lines[y].T
        if not left_lines[y].any():
            left_line.append(False)
        else:    
            left_line.append(np.array([left_lines[y][0].mean(), left_lines[y][1].mean()]))

    for y in range(6):
            right_lines[y] = np.array(right_lines[y])
    right_lines = np.array(right_lines)
    for y in range(6):
        right_lines[y] = right_lines[y].T
        if not right_lines[y].any():
            right_line.append(False)
        else:    
            right_line.append(np.array([right_lines[y][0].mean(), right_lines[y][1].mean()]))
        
    print(left_line) 
    print(right_line)
    print('next') 


    return img








def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅
    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def mix_layer(frame):
    white_frame = np.copy(frame)
    blue_threshold = 187
    green_threshold = 193
    red_threshold = 187
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (frame[:,:,0] < bgr_threshold[0]) \
                | (frame[:,:,1] < bgr_threshold[1]) \
                | (frame[:,:,2] < bgr_threshold[2])
    white_frame[thresholds] = [0,0,0]

    yellow_frame = np.copy(frame)
    hsv = cv2.cvtColor(yellow_frame, cv2.COLOR_BGR2HSV)
    h_threshold = [15, 40]     #opencv_h range 0~180, hsv_full rane = 0~360
    s_threshold = [70, 255]
    v_threshold = [80, 255]
    hsv_mask = cv2.inRange(hsv, np.array([h_threshold[0], s_threshold[0], v_threshold[0]])\
                             , np.array([h_threshold[1], s_threshold[1], v_threshold[1]]))
    yellow = cv2.bitwise_and(yellow_frame, yellow_frame, mask=hsv_mask)
    yellow_frame = cv2.cvtColor(yellow, cv2.COLOR_HSV2BGR)


    yellow_frame1 = np.copy(frame)
    hsv = cv2.cvtColor(yellow_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.inRange(h, h_threshold[0], h_threshold[1])
    h_ch = cv2.bitwise_and(hsv, hsv, mask = h)
    s = cv2.inRange(s, s_threshold[0], s_threshold[1])
    s_ch = cv2.bitwise_and(hsv, hsv, mask = s)
    v = cv2.inRange(v, v_threshold[0], v_threshold[1])
    v_ch = cv2.bitwise_and(hsv, hsv, mask = v)
    m_ch = cv2.add(h_ch, s_ch, v_ch)

    src = cv2.bitwise_or(white_frame, m_ch)
    return m_ch, src


def transform(img, pts1, pts2, xsize, ysize):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    p_img = cv2.warpPerspective(img, M, (xsize, ysize))
    return p_img






mov = cv2.VideoCapture('road_mov/2.mp4')
wid, hei = 854, 480
x1, y1, x2, y2 = 369.7354320721189, 305.0, 476.28795444480807, 305.0
pts1 = np.float32([[(0,hei),(x1, y1), (x2, y2), (wid+30,hei)]])
pts2 = np.float32([[((wid-hei)/2,hei*2), ((wid-hei)/2, 0),(wid, 0), (wid, hei*2)]])
pts2 = pts2 / 2
vertices = np.array([[(0,hei),(x1, y1), (x2, y2), (wid+30,hei)]], dtype=np.int32)


while True:
    ret, frame = mov.read()
    if ret:
        # print(frame.shape)
        src = mix_layer(frame)[1]
        # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src_blur = cv2.GaussianBlur(src, (3, 3), 1, 1)
        edges = cv2.Canny(src_blur, 50, 150, apertureSize=3)
        roi_img = region_of_interest(edges, vertices)
        hough_img_src = hough_line(frame, roi_img, 1, 1 * np.pi/180, 30, 5, 30)
        f_src = cv2.polylines(hough_img_src, vertices, True, (0,0,255), 2)

        p_src = transform(src, pts1, pts2, int((wid+hei/2)/2), int(hei))
        p_src = cv2.cvtColor(p_src, cv2.COLOR_BGR2GRAY)
        # perspective_src = cv2.GaussianBlur(p_src, (3,3), 1, 1)
        c_p_src = cv2.Canny(p_src, 50, 150, apertureSize=3)
        l_range = -1.8, 1.8
        # hough_img__perspecive = hough_line_range_roi(p_src, c_p_src, 1, 1 * np.pi/180, 10, 5, 30,l_range)


        # f_src = cv2.polylines(hough_img_src, vertices, True, (0,0,255), 2)
        
      

        cv2.imshow('f', f_src)
        cv2.imshow('p', p_src)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
mov.release()
cv2.destroyAllWindows()

