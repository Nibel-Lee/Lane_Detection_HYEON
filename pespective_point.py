import cv2
import numpy as np

class Error(Exception):
    pass

hei, wid = None, None

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
    return src


def hough_line_test(img, edges, rho, theta, threshold, min_line_len, max_line_gap, p_p):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hei, wid, __ = img.shape
    out_of_range = wid, hei
    xylist = [[p_p[0], p_p[1]]]
    try:
        for line1 in lines:
            for line2 in lines:
                # print(line1)
                # print(line2)
                try: 
                    x, y = line_intersection(line1, line2, out_of_range)
                    xylist.append([x, y])
                    # print('make x,y')
                    # print(x, y)
                except :
                    # print('error')
                    pass
    
        if not xylist:
                raise Exception('no point')
        for x,y in xylist:
            cv2.circle(img, (int(x),int(y)), 1, (0,255,0), -1)
        txylist = np.array(xylist)
        txylist = txylist.T
        px, py = sum(txylist[0], 0.0) / len(txylist[0]), sum(txylist[1], 0.0) / len(txylist[1])
        cv2.circle(img, (int(px),int(py)), 5, (0,0,255), -1)
    except:
        print(Exception.__traceback__)
        pass

    return px,py


def line_intersection(line1, line2, out_of_range):
    xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
    ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3]) #Typo was here
    if xdiff[0] == 0 | xdiff[1] == 0 | ydiff[0] == 0 | ydiff[1] == 0:
        raise Exception('vertical and pychopical delete')


    # print(xdiff, ydiff)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def deta(a):
        return a[0] * a[3] - a[1] * a[2]

    div = det(xdiff, ydiff)
    if div == 0:
    #    print('interset')
       raise Exception('lines do not intersect')
    
    d = (deta(*line1), deta(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    if not out_of_range: 
        pass   
    else:
        if (x < 0) | (x > out_of_range[0]) | (y < 0) | (y > out_of_range[1]):
            # print('out_of_range')
            raise Exception('out_of_range')
    # print('sucess')
    return x, y


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


def transform(img, pts1, pts2, xsize, ysize):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    p_img = cv2.warpPerspective(img, M, (xsize, ysize))
    return p_img


mov = cv2.VideoCapture('road_mov/4.mp4')
p_p=[0,0]
while True:
    ret, frame = mov.read()
    if ret:
        height, width,_ = frame.shape
        hei = height
        wid = width
        vertices = np.array([[(30,height),(30, height*5/6), (width-30, height*5/6), (width-30,height)]], dtype=np.int32)
        cv2.polylines(frame, vertices, True, (0,0,255), 2)
        pimg = cv2.cvtColor(mix_layer(frame), cv2.COLOR_BGR2GRAY)
        # roi_img = region_of_interest(pimg, vertices)
        # c_img = cv2.Canny(roi_img, 100,200, apertureSize=3)
        c_img = cv2.Canny(pimg, 100, 200, apertureSize=3)
        p_p = hough_line_test(frame, c_img, 1, 1 * np.pi/180, 40, 20, 30, p_p)
        # cv2.imshow('roi', roi_img)
        cv2.imshow("canny", c_img)
        cv2.imshow('re', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break


px,py = p_p.split()
print(px, py)
print(width, height)
line_1 = [[30, hei, px, py]]
line_2 = [[wid, hei, px, py]]
line_m = [[0, int(hei*8/13), wid, int(hei*8/13)]]
x1, y1 = line_intersection(line_1, line_m, None)
x2, y2 = line_intersection(line_2, line_m, None)

mov.release()
cv2.destroyAllWindows()

plus_y = 10
y1 += plus_y
y2 += plus_y
print (wid, hei)
print('vertices')
print(x1, y1)
print(x2, y2)

pts1 = np.float32([[(0,hei),(x1, y1), (x2, y2), (wid+30,hei)]])
pts2 = np.float32([[((wid-hei)/2,hei*2), ((wid-hei)/2, 0),(wid, 0), (wid, hei*2)]])
vertices = np.array([[(30,hei),(x1, y1), (x2, y2), (wid,hei)]], dtype=np.int32)
mov = cv2.VideoCapture('road_mov/3.mp4')

while True:
    ret, frame = mov.read()
    if ret:
        copy_frame = np.copy(frame)
        cv2.polylines(frame, vertices, True, (0, 0, 255), 1)
        cv2.imshow('', frame)
        cv2.imshow("h",transform(copy_frame, pts1, pts2, int(wid+hei), int(hei*2)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

mov.release()
cv2.destroyAllWindows()



