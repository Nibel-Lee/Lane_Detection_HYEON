import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob2
from moviepy.editor import VideoFileClip
import math
# For cordinate isolation
#%matplotlib qt  

## 에러 발생시 무시하고 계속 실행도록 하는 Class
class Error(Exception):
    pass

## 전역변수 생성        
global polygon, area

## Compute camera calibration matrix and distortion coefficients.
## camera_ Calibration 함수는 내가 안짜서 내용물이 뭔지 모르겠음
## 하는 일은 광각카메라로 사진을 찍을 경우, 카메라 렌즈에 의해서 이미지 왜곡 현상이 발생하는데 이를 보정하여 이미지 왜곡을 없애는 함수임
## 실제로 이 함수의 결과를 보면 내부 파라미터 mtx와 왜곡계수 dist를 반환함, 이는 카메라의 왜곡 정도와 내부 파라미터를 판단하여 리턴해주는 함수
def camera_Calibraton(directory, filename, nx, ny, img_size):
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = [] 
    imgpoints = []

    # Image List
    images = glob2.glob('./'+directory+'/'+filename+'*'+'.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    if (len(objpoints) == 0 or len(imgpoints) == 0):
        raise Error("Calibration Failed")
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
    return mtx, dist




## undistort 함수는 위의 camera_calibration 함수를 통해서 얻은 파라미터를 통해서 실질적으로 이미지를 보정하는 함수이다.
## camera_calibration 함수와 떨어트려서 보면 안되는 함수임
def undistort(image, mtx, dist):
    image = cv2.undistort(image, mtx, dist, None, mtx)
    return image




## 카메라 캘리브레이션을 실제로 실행하여 받는 부분
## 카메라에 대한 기본 정보이기 때문에, 아래와 같이 따로 파일 위치를 선언하여 사용한다.
## 기본 카메라 이미지는 주로 체스판을 활용하여 이를 통해서 왜곡 정보를 획득한다.
## 아래의 내용에 대해서 상세 파일이 존재하지 않으므로 Error 발생으로 Pass 된다.
mtx, dist = camera_Calibraton('camera_cal', 'calibration', 9, 6, (720, 1280))
checker_dist = mpimg.imread("./camera_cal/calibration2.jpg")
checker_undist = undistort(checker_dist, mtx, dist)


## 카메라 왜곡 이미지 보정이 제대로 되었는지 확인하는 부분
f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 18))
ax1.imshow(checker_dist)
ax1.set_title('Original', fontsize=15)
ax2.imshow(checker_undist)
ax2.set_title('Undistorted', fontsize=15)




## 절대값 소벨 변환을 기준으로 엣지 추출
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    assert(orient == 'x' or orient == 'y'), "Orientation must be x or y"
    
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize = sobel_kernel)
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
 
    return grad_binary




## 미분 소벨 변환을 기준으로 엣지 추출
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobelxy = np.power((np.power(sobelx,2)+np.power(sobely,2)),0.5)
    
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return mag_binary



## 이것도 소벨 변환으로 엣지 추출하는건데 아마 적분이지 않을까 생각중
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary


## 이 부분은 이미지의 Feature를 성분별로 추출해내는거임
## RGB, HSV, HLS, YCrCb, Lab로 이미지를 각가 추출해내는 거임
## 이 함수를 이미지와 추출해낼 채널을 아래와 같이 입력하면 이미지에서 그에 대한 성분을 리턴해줌
def channel_Isolate(image,channel):
    ## Takes in only RBG images
    if (channel == 'R'):
        return image[:,:,0]
    
    elif (channel == 'G'):
        return image[:,:,1]
    
    elif (channel == 'B'):
        return image[:,:,2]
    
    elif (channel == 'H'):
        HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return HSV[:,:,0]
    
    elif (channel == 'S'):
        HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return HSV[:,:,1]
        
    elif (channel == 'V'):
        HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return HSV[:,:,2]
        
    elif (channel == 'L'):
        HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        return HLS[:,:,1]
    
    elif (channel == 'Cb'):
        YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return YCrCb[:,:,2]
    
    elif (channel == 'U'):
        LUV = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        return LUV[:,:,2]
    ## 채널 입력 안되면 에러 반환함
    else:
        raise Error("Channel must be either R, G, B, H, S, V, L, Cb, U")





## thresh 값에 맞는 영역을 channel에서 찾아서 반환하는 매커니즘, thresh 값에 해당하지않으면 0을 대신 넣어서 이미지를 만들어서 넘겨주는걸로 앎
def threshold_Channel(channel,thresh):
    retval, binary = cv2.threshold(channel.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    return binary


def transform(undist,src,dst,img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped

def find_lines(img):
    right_side_x = []
    right_side_y = []
    left_side_x = []
    left_side_y = []  
    
    past_cord = 0
    
    # right side
    for i in reversed(range(10,100)):
        histogram = np.sum(img[i*img.shape[0]/100:(i+1)*img.shape[0]/100,img.shape[1]/2:], axis=0) 
        xcord = int(np.argmax(histogram)) + 640
        ycord = int(i*img.shape[0]/100)
        if (i == 50):
            right_lane_dp = (xcord)
        if (ycord == 0 or xcord == 0):
            pass
        elif (abs(xcord-past_cord) > 100 and not(i == 99) and not( past_cord == 0)):
            pass
        elif (xcord == 640):
            pass
        else:
            right_side_x.append(xcord)
            right_side_y.append(ycord)
            past_cord = xcord

    past_cord = 0
    # left side
    for i in reversed(range(10,100)):
        histogram = np.sum(img[i*img.shape[0]/100:(i+1)*img.shape[0]/100,:img.shape[1]/2], axis=0)
        xcord = int(np.argmax(histogram))
        ycord = int(i*img.shape[0]/100)
        if (i == 50):
            left_lane_dp = (xcord)
        if (ycord == 0 or xcord == 0):
            pass
        elif (abs(xcord-past_cord) > 100 and not(i == 99) and not(past_cord == 0)):
            pass
        else:
            left_side_x.append(xcord)
            left_side_y.append(ycord)
            past_cord = xcord
    
    left_line =  (left_side_x,left_side_y)
    right_line = (right_side_x,right_side_y)
    
    left_line =  (left_line[0][1:(len(left_line[0])-1)],left_line[1][1:(len(left_line[1])-1)])
    right_line = (right_line[0][1:(len(right_line[0])-1)],right_line[1][1:(len(right_line[1])-1)])
    
    lane_middle = int((right_lane_dp - left_lane_dp)/2.)+left_lane_dp
    
    if (lane_middle-640 > 0):
        leng = 3.66/2
        mag = ((lane_middle-640)/640.*leng)
        head = ("Right",mag)
    else:
        leng = 3.66/2.
        mag = ((lane_middle-640)/640.*leng)*-1
        head = ("Left",mag)

    
    return left_line, right_line, head


def lane_curve(left_line,right_line):
    
    degree_fit = 2
    
    fit_left = np.polyfit(left_line[1], left_line[0], degree_fit)
    
    fit_right = np.polyfit(right_line[1], right_line[0], degree_fit)
    
    x = [x * (3.7/700.) for x in left_line[0]]
    y = [x * (30/720.) for x in left_line[1]]
    
    curve = np.polyfit(y, x, degree_fit)
           
    return fit_left, fit_right, curve


def impose_Lane_Area(undist, fit_left, fit_right, trans, src, dst, img_size, curve):
    
    global left_points,right_points
    
    left_points  = []
    right_points = []

    left = np.poly1d(fit_left)
    right = np.poly1d(fit_right)

    rad_curve = ((1 + (2*curve[0]*710/2 + curve[1])**2)**1.5)/np.absolute(2*curve[0])  
 
    for i in range(100,710,2):
        if (int(left(i)<0)):
            pass
        else:
            left_points.append([int(left(i)),i])

    for i in range(100,710,2):
        if (int(right(i)<0)):
            pass
        else:
            right_points.append([int(right(i)),i])

    
    polygon_points = right_points + list(reversed(left_points))
    polygon_points = np.array(polygon_points)

    
    overlay = np.zeros_like(trans)
    trans_image = cv2.fillPoly(overlay, [polygon_points], (0,255,0) )
    
    M = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(trans_image, M, img_size)
    
    area = cv2.contourArea(polygon_points)
    
    #area = 10000000000

    return unwarped, area, rad_curve, polygon_points

def pipeline(image,mtx, dist, dash): 
    global polygon, area, left, right, last, frame_count,polygon_points_old, last, Head_b, rad_curve_b, vehicle_count, IMAGE
    
    frame_count+=1
    
    
    try:
        # Set Parameters
        img_size = (image.shape[1], image.shape[0])
        src =  np.float32([[250,700],[1200,700],[550,450],[750,450]])
        dst = np.float32([[250,700],[1200,700],[300,50],[1000,50]])

        # Run 2
        undist = undistort(image, mtx, dist)

        # Run 4
        trans = transform(undist,src,dst,img_size)

        # Run 3
        red_threshed = threshold_Channel(channel_Isolate(trans,'R'),(220,255))
        V_threshed = threshold_Channel(channel_Isolate(trans,'V'),(220,255))        
        # Cb_tresh = threshold_Channel(channel_Isolate(trans,'Cb'),(200,255))
        HSV = cv2.cvtColor(trans, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
        
        sensitivity_1 = 68
        white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

        sensitivity_2 = 60
        HSL = cv2.cvtColor(trans, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
        
        white_3 = cv2.inRange(trans, (200,200,200), (255,255,255))
 
        bit_layer = red_threshed | V_threshed | yellow | white | white_2 | white_3
        
        
        # Run 5
        left_line, right_line, head = find_lines(bit_layer)


        # Run 6 
        fit_left, fit_right, curve = lane_curve(left_line,right_line)

        # Run 7
        unwarped, area, rad_curve, polygon_points = impose_Lane_Area(undist,fit_left, fit_right, trans, src, dst, img_size, curve)
    
        # Detect
        vehicle_count = 0
    
        ########## Transformation visualization #########
#         return trans
        
    
        ########## Bit layer visualization #########
#         return bit_layer
    
        
        
        if area < 250000:
            
            print ("Error in area")
            
            result = cv2.addWeighted(undist, 1, polygon, 0.3, 0)

            original = undist.copy()

            original[475:720] = result[475:720]

            original[50:130,50:350,:] = dash
        
            font = cv2.FONT_HERSHEY_DUPLEX
            text = last[0]
            cv2.putText(original,text , (256,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

            dist = last[1]
            cv2.putText(original,str(dist) , (181,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

            rad_curve = last[2]
            cv2.putText(original,str(rad_curve) , (244,91), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

            cv2.putText(original,str(last[3]) , (189,111), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
            
            #cv2.putText(original,"AREA ERROR" , (500,500), font, 1, (255,0,0), 1, cv2.LINE_AA)

            return original
        
        

        else:
            ######### Lane points found visualization #########

#             left =  []
#             right = []
              
#             for i in range(len(left_line[0])):
#                 left.append([left_line[0][i],left_line[1][i]])
                
#             for i in range(len(right_line[0])):
#                 right.append([right_line[0][i],right_line[1][i]])
            
#             for point in left:
#                 cv2.circle(trans, (point[0],point[1]), 10, 255, -1)
                
#             for point in right:
#                 cv2.circle(trans, (point[0],point[1]), 10, 255, -1)
   
#             return trans
         
    
            ########## Polyfit visualization #########
            
#             for i in left_points:
#                 cv2.circle(trans, (i[0],i[1]), 10, 255, -1)
#             for i in right_points:
#                 cv2.circle(trans, (i[0],i[1]), 10, 255, -1)
                
#             return trans    

    
    
            ######### Augmented visualization #########
        
            if (polygon_points_old == None):
                polygon_points_old = polygon_points
            
            
            a = polygon_points_old
            b = polygon_points
            ret = cv2.matchShapes(a,b,1,0.0)     
            
            original = undist.copy()
            
            font = cv2.FONT_HERSHEY_DUPLEX
                    
            #cv2.putText(original,(str(ret)), (10,300), font, 1, (255,0,0), 1, cv2.LINE_AA)
            
            
            if (ret < 0.045 or IMAGE):
            
                polygon = unwarped

                result = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

                polygon_points_old = polygon_points
                
                Head_b = head
                
                rad_curve_b = rad_curve
                
                vehicle_count_b = vehicle_count

            else:
                
                result = cv2.addWeighted(undist, 1, polygon, 0.3, 0)        
                
                polygon = polygon
                
            original[475:720] = result[475:720]
            
            original[50:130,50:350,:] = dash
                        
            if ((frame_count % 10)== 0):
        
                font = cv2.FONT_HERSHEY_DUPLEX
                text = Head_b[0]
                cv2.putText(original,text , (256,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                dist = math.ceil(Head_b[1]*100)/100
                cv2.putText(original,str(dist) , (181,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                rad_curve = math.ceil(rad_curve_b*100)/100
                cv2.putText(original,str(rad_curve) , (244,91), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                cv2.putText(original,str(vehicle_count) , (189,111), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                last = [text,dist,rad_curve,vehicle_count]
            
            else:
                
                font = cv2.FONT_HERSHEY_DUPLEX
                text = last[0]
                cv2.putText(original,text , (256,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                dist = last[1]
                cv2.putText(original,str(dist) , (181,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                rad_curve = last[2]
                cv2.putText(original,str(rad_curve) , (244,91), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

                cv2.putText(original,str(last[3]) , (189,111), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
            
            
            return original 
    
    except Exception as e:
        
        print (e)
        print ("Error in something?")
        
        result = cv2.addWeighted(undist, 1, polygon, 0.3, 0)

        original = undist.copy()

        original[475:720] = result[475:720]
        
        original[50:130,50:350,:] = dash
        
        font = cv2.FONT_HERSHEY_DUPLEX
        text = last[0]
        cv2.putText(original,text , (256,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

        dist = last[1]
        cv2.putText(original,str(dist) , (181,71), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

        rad_curve = last[2]
        cv2.putText(original,str(rad_curve) , (244,91), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(original,str(last[3]) , (189,111), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
        
        #cv2.putText(original,"DROP ERROR" , (500,500), font, 1, (255,0,0), 1, cv2.LINE_AA)
        
        return original

directory = 'camera_cal'
filename = 'calibration'
nx = 9 
ny = 6
cal_img_size = (720, 1280)

# Run 1  (Run once to save time)
mtx, dist = camera_Calibraton(directory, filename, nx, ny, cal_img_size)

# Establish Inital Frame Count
frame_count = 10
polygon_points_old = None
IMAGE = True
last = ["i","i","i","i"]

# Read In Data Project Video
image_1 = mpimg.imread("./test_images/test1.jpg")
image_2 = mpimg.imread("./test_images/test2.jpg")
image_3 = mpimg.imread("./test_images/test3.jpg")
image_4 = mpimg.imread("./test_images/test4.jpg")
image_5 = mpimg.imread("./test_images/test5.jpg")
image_6 = mpimg.imread("./test_images/test6.jpg")

# # # Read In Data Harder Video
# image_1 = mpimg.imread("./test_images_1/test1.jpg")
# image_2 = mpimg.imread("./test_images_1/test2.jpg")
# image_3 = mpimg.imread("./test_images_1/test3.jpg")
# image_4 = mpimg.imread("./test_images_1/test4.jpg")
# image_5 = mpimg.imread("./test_images_1/test5.jpg")
# image_6 = mpimg.imread("./test_images_1/test6.jpg")

# # Read In Data Hardest Video
# image_1 = mpimg.imread("./test_images_2/test1.jpg")
# image_2 = mpimg.imread("./test_images_2/test2.jpg")
# image_3 = mpimg.imread("./test_images_2/test3.jpg")
# image_4 = mpimg.imread("./test_images_2/test4.jpg")
# image_5 = mpimg.imread("./test_images_2/test5.jpg")
# image_6 = mpimg.imread("./test_images_2/test6.jpg")

# Import Dash Interface
dash = mpimg.imread("./Dash.jpg")


# Run pipline on images

Image_1 = pipeline(image_1, mtx, dist, dash)
Image_2 = pipeline(image_2, mtx, dist, dash)
Image_3 = pipeline(image_3, mtx, dist, dash)
Image_4 = pipeline(image_4, mtx, dist, dash)
Image_5 = pipeline(image_5, mtx, dist, dash)
Image_6 = pipeline(image_6, mtx, dist, dash)


# Single Image View
# plt.figure(figsize=(460,440))
# plt.imshow(Image_3, cmap="gray")


f, ((ax1, ax2), (ax3, ax4),(ax5, ax6), (ax7, ax8),(ax9, ax10), (ax11, ax12)) = plt.subplots(6, 2, figsize=(12, 18))
ax1.imshow(image_1)
ax1.set_title('Original Image 1', fontsize=20)
ax2.imshow(Image_1)
ax2.set_title('Augmented 1', fontsize=20)
ax3.imshow(image_2)
ax3.set_title('Original Image 2', fontsize=20)
ax4.imshow(Image_2)
ax4.set_title('Augmented 2', fontsize=20)
ax5.imshow(image_3)
ax5.set_title('Original Image 3', fontsize=20)
ax6.imshow(Image_3)
ax6.set_title('Augmented 3', fontsize=20)
ax7.imshow(image_4)
ax7.set_title('Original Image 4', fontsize=20)
ax8.imshow(Image_4)
ax8.set_title('Augmented 4', fontsize=20)
ax9.imshow(image_5)
ax9.set_title('Original Image 5', fontsize=20)
ax10.imshow(Image_5)
ax10.set_title('Augmented 5', fontsize=20)
ax11.imshow(image_6)
ax11.set_title('Original Image 6', fontsize=20)
ax12.imshow(Image_6)
ax12.set_title('Augmented 6', fontsize=20)

plt.subplots_adjust(hspace=0.3)