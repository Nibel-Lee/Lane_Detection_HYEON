import cv2
import numpy as np
# from matplotlib import pyplot as plt

def img_list_print(img):
    cv2.imshow("origin", img[0])
    # cv2.imshow("gray", img[1])
    # cv2.imshow("Gaussian", img[2])
    # cv2.imshow("bilateral", img[3])
    cv2.imshow("re_mark", img[4])
    cv2.imshow("res_mark", img[5])
    # cv2.imshow("Line Detect_G", img[6])
    # cv2.imshow("Line Detect_b", img[7])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

def line_detect(ori_img, img, thr, minLineLength, maxLineGap, blur_kind):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_mark = np.zeros_like(ori_img)
    img_blur = img_gray
    if blur_kind == "Gaussian":
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1, 1)
        print(blur_kind)
    elif blur_kind == "bilteral":
        img_blur = cv2.bilateralFilter(img_gray, 9, 75, 75)
        print(blur_kind)
    elif blur_kind == "median":
        img_blur = cv2.medianBlur(img_gray, 5)
        print(blur_kind)
    
    img_canny = cv2.Canny(img_blur,50, 150, apertureSize=3)

    lines = cv2.HoughLines(img_canny, 1, np.pi/180, thr, minLineLength, maxLineGap)
    hei, wid = img.shape[:2]
    # print(img.shape)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1200*(-b))
        y1 = int(y0 + 1200*a)
        x2 = int(x0 - 1200*(-b))
        y2 = int(y0 - 1200*a)
        
        # print(x0, y0)
        # cv2.line(ori_img, (int(wid/2), 0), (int(wid/2), hei), (255, 255, 255), 1)
        # cv2.line(ori_img, (int(wid/4), 0), (int(wid/4), hei), (255, 255, 255), 1)
        # cv2.line(ori_img, (int(wid*3/4), 0), (int(wid*3/4), hei), (255, 255, 255), 1)
        # cv2.line(ori_img, (0, int(hei/2)), (wid, int(hei/2)), (255, 255, 255), 1)
        # cv2.line(ori_img, (0, int(hei/4)), (wid, int(hei/4)), (255, 255, 255), 1)
        # cv2.line(ori_img, (0, int(hei*3/4)), (wid, int(hei*3/4)), (255, 255, 255), 1)
        text = "("+str(wid)+", "+str(hei)+")"
        cv2.putText(ori_img, text, (10, hei-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if (y0 > hei*3/4) | (y0 < hei/4):
            cv2.line(base_mark, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_8)    
        # cv2.circle(ori_img, (x0,y0), 5, (255,0,0), -1)
        # cv2.circle(ori_img, (x1,y1), 5, (0,255,0), -1)
        # cv2.circle(ori_img, (x2,y2), 5, (0,255,0), -1)

    vertices = np.array([[(50,height),(100, height*2/3), (width-100, height*2/3), (width-50,height)]], dtype=np.int32)
    re_mark = region_of_interest(base_mark, vertices, (255,255,255))
    print(re_mark.shape)
    (my, mx, _) = re_mark.shape
    xl1 = []
    xr1 = []
    xl2 = []
    xr2 = []
    
    mid = mx / 2

    y1 = my
    for x in range(mx):
        if re_mark[y1-1][x-1][2] == 255:
            if x-1 < mid:
                xl1 += [x-1]
            else:
                xr1 += [x-1]

    y2 = int(my*2/3)
    for x in range(mx):
        if re_mark[y2+1][x-1][2] == 255:
            if x-1 < mid:
                xl2 += [x-1]
            else:
                xr2 += [x-1]
    
        
        # re_mark[mx][my]
        # re_mark[mx][my*2/3]
    print(xl1)
    print(xr1)
    print(xl2)
    print(xr2)
    res_img = cv2.bitwise_or(ori_img, re_mark)
    cv2.line(res_img, (int((xl1[-1] + xr1[0]) / 2), y1), (int((xl2[-1] + xr2[0]) / 2), y2), (0, 255, 0), 5, cv2.LINE_8)
 
    return ori_img, img_gray, img_canny, img_blur, re_mark, res_img



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





img = cv2.imread("road_img/4.jpeg", cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h0, s0, v0 = cv2.split(hsv)
h0 = cv2.inRange(h0, 20, 50)
yellow = cv2.bitwise_and(hsv, hsv, mask = h0)
yellow = cv2.cvtColor(yellow, cv2.COLOR_HSV2BGR)

white = np.copy(img)
blue_threshold = 200
green_threshold = 200
red_threshold = 200
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로
thresholds = (img[:,:,0] < bgr_threshold[0]) \
            | (img[:,:,1] < bgr_threshold[1]) \
            | (img[:,:,2] < bgr_threshold[2])
white[thresholds] = [0,0,0]

img_mix = cv2.add(yellow, white)
# cv2.imshow("0", yellow)
# cv2.imshow("1", white)
# cv2.imshow("2", img_mix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img_list_print(line_detect(img, 140))


height, width = img.shape[:2] # 이미지 높이, 너비

# 사다리꼴 모형의 Points
vertices = np.array([[(50,height),(100, height/2), (width-100, height/2), (width-50,height)]], dtype=np.int32)
roi_img = region_of_interest(img, vertices)

blur_kind = "Gaussian"
minLineLength = 50
maxLineGap = 10
img_list_print(line_detect(img, img_mix, 160, minLineLength, maxLineGap, blur_kind))