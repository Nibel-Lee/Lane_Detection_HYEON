import cv2
import numpy as np

class Error(Exception):
    pass


def roi(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅
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


def transform(img, pts1, pts2, xsize, ysize):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    p_img = cv2.warpPerspective(img, M, (xsize, ysize))
    return p_img

def yw_Mask(frame):
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
    return src

def hough_line(img, edges, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    for line in lines:    
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (x1,y1), 5, (0,255,0), -1)
            cv2.circle(img, (x2,y2), 5, (255,0,0), -1)

    return img, lines

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
    ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3]) #Typo was here
    # if xdiff[0] == 0 | xdiff[1] == 0 | ydiff[0] == 0 | ydiff[1] == 0:
    #     raise Exception('vertical and pychopical delete')
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def deta(a):
        return a[0] * a[3] - a[1] * a[2]
    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (deta(*line1), deta(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def outlier(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3-q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return np.where((data > upper_bound)|(data < lower_bound))

def intersection(lines):
    xlist = list()
    ylist = list()
    try:
        for line1 in lines:
            for line2 in lines:
                try: 
                    x, y = line_intersection(line1, line2)
                    xlist.append(x)
                    ylist.append(y)
                except :
                    pass
    
        if not xlist:
                raise Exception('no point')
        tx = (np.array(xlist))
        ty = (np.array(ylist))
        # tx = outlier(np.array(xlist))
        # ty = outlier(np.array(ylist))
        px, py = np.mean(tx), np.mean(ty)
    except:
        print(Exception.__traceback__)
        pass

    return px,py

def mintersection(lines):
    xlist = list()
    ylist = list()
    mline = [[wid/2, hei, wid/2, 0]]
    try:
        for line in lines:
            try: 
                x, y = line_intersection(line, mline)
                xlist.append(x)
                ylist.append(y)
            except :
                pass
    
        if not xlist:
                raise Exception('no point')
        tx = (np.array(xlist))
        ty = (np.array(ylist))
        # tx = outlier(np.array(xlist))
        # ty = outlier(np.array(ylist))
        px, py = np.mean(tx), np.mean(ty)
    except:
        print(Exception.__traceback__)
        pass

    return px,py



mov = cv2.VideoCapture('road_mov/2.mp4')
# 영상 크기
wid, hei = 854, 480

while True:
    ret, frame = mov.read()
    if ret:
        try:
            # 1. 영상을 반으로 쪼개서 아래만 추출한다. h_frame
            hvertices = np.array([[(0,hei),(0, hei*2/3), (wid, hei*2/3), (wid,hei)]], dtype=np.int32)
            h_frame = roi(frame, hvertices)
            hm_frame = cv2.GaussianBlur(yw_Mask(h_frame), (3, 3), 1, 1)
            # 2. 추출한 이미지로 Canny, Hough Transform
            hmc_frame = cv2.Canny(hm_frame, 50, 150, apertureSize=3)
            hmch_frame, lines = hough_line(hm_frame, hmc_frame, 1, 1 * np.pi/180, 30, 10, 50)
            # 3. 소실점을 구하고(삼각형) --> ppt
            px,py = mintersection(lines)
            # 4. 관심 영역을 설정한다.(삼각형->사다리꼴(삼각형 높이의 70%)) rvertice
            lp1 = [[0, hei, px, py]]
            lp2 = [[wid, hei, px, py]]
            lpm = [[0, int(hei*7/10), wid, int(hei*7/10)]]
            px1, py1 = line_intersection(lp1, lpm)
            px2, py2 = line_intersection(lp2, lpm)
            # 5. 관심 영역을 설정한 것을 핀다 --> t_frame
            pts1 = np.float32([[(0,hei),(px1, py1), (px2, py2), (wid,hei)]])
            pts2 = np.float32([[(0,hei*2), (0, 0),(wid, 0), (wid, hei*2)]])
            t_frame = transform(frame, pts1, pts2, int(wid), int(hei*2))
            # 6. 관심 영역 frame에 표시
            vertices = np.array([[(0,hei),(px1, py1), (px2, py2), (wid,hei)]], dtype=np.int32)
            cv2.circle(frame, (int(px),int(py)), 10, (0,255,0), -1)
            cv2.polylines(frame, vertices, True, (0,0,255), 2)
            # 영상 표시
            cv2.imshow('src', frame)
            cv2.imshow('hmch', hmch_frame)
            cv2.imshow('roi', t_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except:
            pass
    else:
        break
mov.release()
cv2.destroyAllWindows()
