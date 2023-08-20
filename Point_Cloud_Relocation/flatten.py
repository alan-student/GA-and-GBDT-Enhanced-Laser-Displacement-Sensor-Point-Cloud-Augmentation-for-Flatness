from math import cos, pi, sqrt,sin
from re import X
from cv2 import circle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.stats import norm
import functools
from numpy import Inf
import itertools
from itertools import combinations
import matplotlib as mpl

#函数名:FlattenGet
#输入变量: xMat,yMat,zMat,均为形状为(14,)的np.array对象
#输出变量:一个（5,）的List对象,前三位对应a*x+b*y+c=0,第四位对应方差，第五位对应平面度误差
def FlattenGet(xMat,yMat,zMat):
    # 创建系数矩阵M
    M = np.zeros((3, 3))  # 定义一个3X3的矩阵，并初始化都为0
    for i in range(0, 14):  # i=14  一共14个坐标
        M[0, 0] = M[0, 0] + xMat[i] ** 2
        M[0, 1] = M[0, 1] + xMat[i] * yMat[i]
        M[0, 2] = M[0, 2] + xMat[i]
        M[1, 0] = M[0, 1]
        M[1, 1] = M[1, 1] + yMat[i] ** 2
        M[1, 2] = M[1, 2] + yMat[i]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = 14  # i=14  一共14个坐标
    # print('系数矩阵M=\n\n', M)

    # 创建列向量p
    p = np.zeros((3, 1))  # 定义一个3X1的矩阵，并初始化都为0
    for i in range(0, 14):  # i=14  一共14个坐标
        p[0, 0] = p[0, 0] + xMat[i] * zMat[i]
        p[1, 0] = p[1, 0] + yMat[i] * zMat[i]
        p[2, 0] = p[2, 0] + zMat[i]
    # print('\n列向量p=\n\n', p)

    # 求解Q，Q表示拟合平面的系数a=Q[0,0] , 系数b=Q[1,0] ,系数c=Q[2,0]
    M_inv = np.linalg.inv(M)  # np.linalg.inv()：矩阵求逆  ,求矩阵A的逆矩阵
    Q = np.dot(M_inv, p)  # np.dot(）则是向量内积
   # print('\n平面方程的求解出的系数为：[系数a=Q[0,0] , 系数b=Q[1,0] ,系数c=Q[2,0]]=', (Q[0, 0], Q[1, 0], Q[2, 0]))
   # print('\n最小二乘拟合平面结果为：  zMat =  (%.5f)* xMat + ( %.5f)* yMat + (%.5f)' % (Q[0, 0], Q[1, 0], Q[2, 0]))  # 保留5位小数
    # 计算方差R
    R = 0
    for i in range(0, 14):  # i=14  一共14个坐标
        R = R + (Q[0, 0] * xMat[i] + Q[1, 0] * yMat[i] + Q[2, 0] - zMat[i]) ** 2
    #print('\n方差为：R=%.*f\n' % (5, R))  # 保留5位小数

    result = list()
    for j in range(3):
        result.append(Q[j,0])
    result.append(R)
    # ************计算（实际坐标）点到平面的距离d******************#
    T = Q[0, 0] * xMat + Q[1, 0] * yMat - zMat + Q[2, 0]  # 此处是：公式的分子=a*x+b*y-z+c

    F = np.sqrt(np.sum(np.square([Q[0, 0], Q[1, 0], 1])))  # 此处是：公式的分母=根号（a^2+b^2+1）

    d = T / F  # 公式：点到平面的距离d=分子/分母
    #print('\n*********平面度数值：若数值为正，表示点在平面上方；若数值为负，表示点在平面下方；数值为0，表示点在平面上**********')
    #print('\n平面度数值为P=\n\n', d)  # 此处是：若输出为正，表示点在平面上方； 若输出为负，表示点在平面下方；输出为0，表示点在平面上；

    # **********平面度误差的最小二乘法评定结果**************#
    # ************数组中取出最大值、最小值****************#
    P1 = min(d)
    P2 = max(d)
    # print('\n平面度最小值为P1=\n\n', P1)
    # print('\n平面度最大值为P2=\n\n', P2)
    # ************最大值减去最小值，并取绝对值***************#
    f = abs(P2 - P1)
    # print('\n平面度误差为f=\n\n', f)
    result.append(f)
    return result


#函数名:FlattenGet7Only
#输入变量: xMat,yMat,zMat,均为形状为(7,)的np.array对象
#输出变量:一个（5,）的List对象,前三位对应a*x+b*y+c=0,第四位对应方差，第五位对应平面度误差
def FlattenGet7Only(xMat,yMat,zMat):
    # 创建系数矩阵M
    M = np.zeros((3, 3))  # 定义一个3X3的矩阵，并初始化都为0
    for i in range(0, 7):  # i=14  一共14个坐标
        M[0, 0] = M[0, 0] + xMat[i] ** 2
        M[0, 1] = M[0, 1] + xMat[i] * yMat[i]
        M[0, 2] = M[0, 2] + xMat[i]
        M[1, 0] = M[0, 1]
        M[1, 1] = M[1, 1] + yMat[i] ** 2
        M[1, 2] = M[1, 2] + yMat[i]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = 14  # i=14  一共14个坐标
    # print('系数矩阵M=\n\n', M)

    # 创建列向量p
    p = np.zeros((3, 1))  # 定义一个3X1的矩阵，并初始化都为0
    for i in range(0, 7):  # i=14  一共14个坐标
        p[0, 0] = p[0, 0] + xMat[i] * zMat[i]
        p[1, 0] = p[1, 0] + yMat[i] * zMat[i]
        p[2, 0] = p[2, 0] + zMat[i]
    # print('\n列向量p=\n\n', p)

    # 求解Q，Q表示拟合平面的系数a=Q[0,0] , 系数b=Q[1,0] ,系数c=Q[2,0]
    M_inv = np.linalg.inv(M)  # np.linalg.inv()：矩阵求逆  ,求矩阵A的逆矩阵
    Q = np.dot(M_inv, p)  # np.dot(）则是向量内积
   # print('\n平面方程的求解出的系数为：[系数a=Q[0,0] , 系数b=Q[1,0] ,系数c=Q[2,0]]=', (Q[0, 0], Q[1, 0], Q[2, 0]))
   # print('\n最小二乘拟合平面结果为：  zMat =  (%.5f)* xMat + ( %.5f)* yMat + (%.5f)' % (Q[0, 0], Q[1, 0], Q[2, 0]))  # 保留5位小数
    # 计算方差R
    R = 0
    for i in range(0, 7):  # i=14  一共14个坐标
        R = R + (Q[0, 0] * xMat[i] + Q[1, 0] * yMat[i] + Q[2, 0] - zMat[i]) ** 2
    #print('\n方差为：R=%.*f\n' % (5, R))  # 保留5位小数

    result = list()
    for j in range(3):
        result.append(Q[j,0])
    result.append(R)
    # ************计算（实际坐标）点到平面的距离d******************#
    T = Q[0, 0] * xMat + Q[1, 0] * yMat - zMat + Q[2, 0]  # 此处是：公式的分子=a*x+b*y-z+c

    F = np.sqrt(np.sum(np.square([Q[0, 0], Q[1, 0], 1])))  # 此处是：公式的分母=根号（a^2+b^2+1）

    d = T / F  # 公式：点到平面的距离d=分子/分母
    #print('\n*********平面度数值：若数值为正，表示点在平面上方；若数值为负，表示点在平面下方；数值为0，表示点在平面上**********')
    #print('\n平面度数值为P=\n\n', d)  # 此处是：若输出为正，表示点在平面上方； 若输出为负，表示点在平面下方；输出为0，表示点在平面上；

    # **********平面度误差的最小二乘法评定结果**************#
    # ************数组中取出最大值、最小值****************#
    P1 = min(d)
    P2 = max(d)
    # print('\n平面度最小值为P1=\n\n', P1)
    # print('\n平面度最大值为P2=\n\n', P2)
    # ************最大值减去最小值，并取绝对值***************#
    f = abs(P2 - P1)
    # print('\n平面度误差为f=\n\n', f)
    result.append(f)
    return result


#函数名:pointRotation
''' 输入变量:pointWantToRotate:待旋转的点坐标,(3,)形状的np.array对象
            point0:旋转轴穿过的点坐标,(3,)形状的np.array对象
            axelVector:旋转轴的方向向量,(3,)形状的np.array对象
            angel:期望旋转的角度,输入一个实数,默认为1
    输出变量:是一个形状为(5,)的List对象
            前四个不需要太管
            第五个是一个形状为(3,)的np.array对象,表示旋转后的点坐标'''
def pointRotation(pointWantToRotate,point0,axelVector,angel=1):
    t = (sum((pointWantToRotate[i]*axelVector[i] - point0[i]*axelVector[i]) for i in range(3)))/(sum((axelVector[j]**2) for j in range(3)))
    standPoint = point0 + (t * axelVector)
    standVector1 = pointWantToRotate - standPoint #算出待旋转点和轴之间的垂直向量，就是这个向量绕轴旋转
    #在原本的空间里作旋转会比较困难，所以需要作坐标变换，合计要求三个基准向量，还差一个
    standVector2 = np.empty(shape=(3))
    standVector2 = np.cross(standVector1,axelVector)
    # standVector2[2] = (n[1]*standVector1[0]-n[0]*standVector1[1])/(standVector1[2]*n[1]-standVector1[1]*n[2])
    # standVector2[1] = -(standVector1[2]/standVector1[0]/standVector1[1])*standVector1[2]
    # standVector2[0] = 1
    # standVector2[0] = n[1]*standVector1[2]-n[2]*standVector1[1]
    # standVector2[1] = n[2]*standVector1[0]-n[0]*standVector1[2]
    # standVector2[2] = n[0]*standVector1[1]-n[1]*standVector1[0]
    standVector1Length = np.linalg.norm(standVector1)
    standVector2Length = np.linalg.norm(standVector2)
    standVector2 = (standVector1Length/standVector2Length) * standVector2
    # standVector2 = -standVector2
    # print("===================================")
    # print(np.dot(standVector1,standVector2))
    # print(np.dot(standVector2,axelVector))
    # print(np.dot(axelVector,standVector2))
    # print(standVector1Length)
    # print(standVector2Length)
    # if standVector2[2] < 0:
    #     standVector2 = -standVector2
    beforeStandAxis = np.array([standVector1,standVector2,axelVector],dtype='float64')
    #分别对应映射后坐标的的x轴，y轴和z轴
    #两个基准向量的长度必须保持相等，z轴(n)的长度可以不用管
    #因为旋转只在xOy平面中旋转，z上始终为0
    afterStandAxis = np.array([[1,0,0],[0,1,0],[0,0,1]])
    k = np.dot(np.linalg.inv(beforeStandAxis),afterStandAxis)
    k_inv = np.linalg.inv(k) #变换矩阵的逆矩阵才是关键
    #把新的坐标轴中的旋转向量重新映射回原坐标轴中，要借助这个逆矩阵
    #下面把旋转向量求出来
    RotationVector = np.empty(shape=(1,3))
    RotationVector[0,0] = cos((angel/180)*pi)
    RotationVector[0,1] = sin((angel/180)*pi) # 不可忘记角度和弧度的转换
    RotationVector[0,2] = 0
    # print(beforeStandAxis[0])
    # RotationVectorBefore = np.dot(RotationVector,k_inv)
    # RotationVectorBefore = RotationVectorBefore + standPoint
    # print(RotationVectorBefore)
    # RotationVectorBefore[0,0] = 0
    # RotationVectorBefore[0,1] = 0
    pointAfterRotation1 = np.dot(RotationVector,k_inv)
    pointAfterRotation = list()
    for k in range(3):
        pointAfterRotation.append(pointAfterRotation1[0,k])
    pointAfterRotation = np.array(pointAfterRotation)
    pointAfterRotation = pointAfterRotation + standPoint
    return [standPoint,standVector1,standVector2,axelVector,pointAfterRotation]

def pointRotationYCZ(pointWantToRotate,point0,axelVector,angel=1):
    k = 1-cos((angel/180)*pi)
    cosAngel = cos((angel/180)*pi)
    sinAngel = sin((angel/180)*pi)
    point0 = np.array(point0)
    n = np.array(axelVector)
    M = sum((n[i]*point0[i]) for i in range(3))
    n = n/(np.linalg.norm(n))
    t = np.empty(shape=(3,4))
    for i in range(3):
        t[i,i] = n[i]**2*k+cosAngel
    t[0,1] = n[0]*n[1]*k - n[2]*sinAngel
    t[0,2] = n[0]*n[2]*k + n[2]*sinAngel
    t[0,3] = (point0[0]-n[0]*M)*k + (n[2]*point0[1]-n[1]*point0[2])*sinAngel
    t[1,0] = n[0]*n[1]*k + n[2]*sinAngel
    t[1,2] = n[1]*n[2]*k - n[1]*sinAngel
    t[1,3] = (point0[1]-n[1]*M)*k + (n[0]*point0[2]-n[2]*point0[0])*sinAngel
    t[2,0] = n[0]*n[2]*k - n[2]*sinAngel
    t[2,1] = n[1]*n[2]*k + n[1]*sinAngel
    t[2,3] = (point0[2]-n[2]*M)*k + (n[1]*point0[0]-n[0]*point0[1])*sinAngel

    resultList = [np.zeros(shape=(3))]*5
    for i in range(3):
        resultList[4][i] = t[i,0]*pointWantToRotate[0]+t[i,1]*pointWantToRotate[1]+t[i,2]*pointWantToRotate[2]+t[i,3]
    return resultList

def xyzMat2coordinate(xMat,yMat,zMat,index):
    return np.array([xMat[index],yMat[index],zMat[index]])