# -*- coding: utf-8 -*-

# interpolation test

import numpy as np
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

# 0~10까지 15개로 나누어 점을 찍음.
x0 = np.linspace(0, 10, 15)
print('x0=', x0)
# cosine 값을 계산
y0 = np.cos(x0)

# x, y (샘플)값을 주고 추정하는 스플라인 곡선을 만든다.
spl = splrep(x0, y0)
# 0~10까지 50구간에 대한 모든 점들을 위 스플라인 곡선으로 추정한 y값을 구한다.
x1 = np.linspace(0, 10, 50)
y1 = splev(x1, spl)

# 그린다.
plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(x0, y0, 'o')
plt.plot(x1, y1, 'r')
plt.grid()

# 이번에는 sine 곡선으로 추정해 본다.
plt.subplot(122)
y2=np.sin(x0)
spl2=splrep(x0, y2)
y3=splev(x1, spl2)
plt.plot(x0, y2, 'o')
plt.plot(x1, y3, 'b')
plt.grid()
plt.show()



# def solver(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
#     a = Symbol('a')
#     b = Symbol('b')
#     c = Symbol('c')
#     d = Symbol('d')
#     e = Symbol('e')
#     f = Symbol('f')

#     eq1 = a*x1**5+b*x1**4+c*x1**3+d*x1**2+e*x1**1+f-y1
#     eq2 = a*x2**5+b*x2**4+c*x2**3+d*x2**2+e*x2**1+f-y2
#     eq3 = a*x3**5+b*x3**4+c*x3**3+d*x3**2+e*x3**1+f-y3
#     eq4 = a*x4**5+b*x4**4+c*x4**3+d*x4**2+e*x4**1+f-y4
#     eq5 = a*x5**5+b*x5**4+c*x5**3+d*x5**2+e*x5**1+f-y5
#     eq6 = a*x6**5+b*x6**4+c*x6**3+d*x6**2+e*x6**1+f-y6
    
#     return solve([eq1, eq2, eq3, eq4, eq5, eq6 ], a, b, c, d, e, f)

    

# def solver_T(a, b, c, x_data, y):
#     result = - (b / 2 / a)
#     if x_data < ((-b) / 2 / a) :
#         result += tf.sqrt((y - c + b *b / 4 / a) / a)
#     else:    
#         result -= tf.sqrt((y - c + b *b / 4 / a) / a)
#     return result




def point_make(img, a1, b1, c1, a2, b2, c2, gap):
    roop, _, _ = roi_img.shape
    roop = int(roop / gap)
    point = []
    y,_,_ = img.AttributeErrorshape
    for x in range(roop):
        y -= x
    return point

