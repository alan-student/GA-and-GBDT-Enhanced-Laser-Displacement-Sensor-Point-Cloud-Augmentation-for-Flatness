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
import os
import argparse

mpl.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")
parser.add_argument("-p", "--precision", help="precision")
args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)



x = np.loadtxt(open("x.csv","rb"),delimiter=",",skiprows=0,usecols=range(14))
y = np.loadtxt(open("y.csv","rb"),delimiter=",",skiprows=0,usecols=range(14))

x1 = x[0:7]
y1 = y[0:7]
z1 = npzFile['z1']
x2 = x[7:14]
y2 = y[7:14]
z2 = npzFile['z2']
z = np.empty(shape=(14))
for j in range(7):
    z[j] = z1[j]
for j in range(7):
    z[j+7] = z2[j]

x_m = np.mean(x)
y_m = np.mean(y)
z_m = np.mean(z)

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

center_estimate = x_m, y_m
center_2, ier = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2       = calc_R(*center_2)
R_2        = Ri_2.mean()
residu_2   = sum((Ri_2 - R_2)**2)
#最小二乘法拟合圆心，圆心的坐标为(xc_2,yc_2,z_m)
circleCenterPoint = np.array([xc_2,yc_2+R_2,z_m])
def flattenGet(xMat,yMat,zMat):
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
        R = R + (Q[0, 0] * x[i] + Q[1, 0] * y[i] + Q[2, 0] - z[i]) ** 2
    #print('\n方差为：R=%.*f\n' % (5, R))  # 保留5位小数

    result = list()
    for j in range(3):
        result.append(Q[j,0])
    result.append(R)
    # ************计算（实际坐标）点到平面的距离d******************#
    T = Q[0, 0] * x + Q[1, 0] * y - z + Q[2, 0]  # 此处是：公式的分子=a*x+b*y-z+c

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
para1 = flattenGet(x,y,z)
#para2 = flattenGet(x2,y2,z2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

#ax1.set_title('Plane Equation  z = (%.5f)* x + ( %.5f)* y + (%.5f)' % (para1[0],para1[1],para1[2]))
ax1.set_xlabel("X Axis",labelpad=15)
ax1.set_ylabel("Y Axis",labelpad=15)
ax1.set_zlabel("Z Axis",labelpad=15)
ax1.scatter(x1, y1, z1, c='r', marker='o')  # 一次坐标用红色、圆圈表示
ax1.scatter(x2, y2, z2, c='g', marker='o')  # 二次坐标用绿色、圆圈表示
ax1.scatter(xc_2,yc_2,z_m,c='y',marker='o') # 圆心-黄色
x_1 = np.linspace(-10, 350, 300)  # （起始点，终止点，100生成start和stop之间100个等差间隔的元素)
y_1 = np.linspace(-10, 350, 300)  # （起始点，终止点，100生成start和stop之间100个等差间隔的元素)
x_1, y_1 = np.meshgrid(x_1, y_1)  # 从坐标向量中返回坐标矩阵,# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值。
z_1 = para1[0] * x_1 + para1[1] * y_1 + para1[2]


#ax1.plot_wireframe(x_1, y_1, z_2, rstride=10, cstride=10)
# print('最小二乘拟合平面三维图，如下所示：')

k = (x2-x1)/(y2-y1)
kdiv1 = list()
for i in k :
    if i == Inf:
        kdiv1.append(Inf)
    else:
        kdiv1.append(i)
kdiv1 = np.array(kdiv1) # 取得对称轴在xOy平面上投影的斜率，这是一个数组，取得七个可能值
kdiv1 = np.array([1]*7)
#下用最大似然估计处理斜率的可能值(考虑误差呈正态分布)
for i in kdiv1:
    if kdiv1[i] == Inf:
        kdiv1[i] == 1e10
kdiv1Best = norm.fit(kdiv1)[0] #如果斜率不是无穷大，直接用最大似然估计进行处理


# 下面考虑如何吧xOy的对称轴升到三维空间的平面里
# 还是使用之前所得到的那个平面
# 考虑的方法就是求得特征向量，然后再过圆心，即可求得一条直线
# 要求特征向量就需要两个特征点的坐标
# print(kdiv1Best)
MyFunc2DGetY = lambda x:kdiv1Best*(xc_2-x)+yc_2 #用于取得特征点的y坐标

MyFunc2DGetZ = lambda x,y:para1[0]*x + para1[1]*y +para1[2] #用于取得特征点的z坐标

CharaPoint2D1 = np.array([0,MyFunc2DGetY(0),MyFunc2DGetZ(0,MyFunc2DGetY(0))])
CharaPoint2D2 = np.array([350,MyFunc2DGetY(350),MyFunc2DGetZ(350,MyFunc2DGetY(350))])
#求得两个特征点，相减即为特征向量

x_line = np.array([CharaPoint2D1[0],CharaPoint2D2[0]])
y_line = np.array([CharaPoint2D1[1],CharaPoint2D2[1]])
z_line = np.array([CharaPoint2D1[2],CharaPoint2D2[2]])
#print(x_line,y_line,z_line)
ax1.plot3D(x_line,y_line,z_line,c='r') # 圆心-黄色
CharaVector = CharaPoint2D2 - CharaPoint2D1
CharaVector[1] = CharaVector[1] +75
# print(CharaVector)
# print(xc_2,yc_2)
ax1.plot_wireframe(x_1, y_1, z_1, rstride=10, cstride=10)

def pointRotation(pointWantToRotate,circlePoint,axelVector,angle=1):
    t = (sum((pointWantToRotate[i]*axelVector[i] - circlePoint[i]*axelVector[i]) for i in range(3)))/(sum((axelVector[j]**2) for j in range(3)))
    standPoint = circlePoint + (t * axelVector)
    standVector1 = pointWantToRotate - standPoint #算出待旋转点和轴之间的垂直向量，就是这个向量绕轴旋转
    #在原本的空间里作旋转会比较困难，所以需要作坐标变换，合计要求三个基准向量，还差一个
    standVector2 = np.empty(shape=(3))
    standVector2 = np.cross(standVector1,axelVector)

    standVector1Length = np.linalg.norm(standVector1)
    standVector2Length = np.linalg.norm(standVector2)
    standVector2 = (standVector1Length/standVector2Length) * standVector2

    beforeStandAxis = np.array([standVector1,standVector2,axelVector],dtype='float64')
    #分别对应映射后坐标的的x轴，y轴和z轴
    #两个基准向量的长度必须保持相等，z轴(axelVector)的长度可以不用管
    #因为旋转只在xOy平面中旋转，z上始终为0
    afterStandAxis = np.array([[1,0,0],[0,1,0],[0,0,1]])
    k = np.dot(np.linalg.inv(beforeStandAxis),afterStandAxis)
    k_inv = np.linalg.inv(k) #变换矩阵的逆矩阵才是关键
    #把新的坐标轴中的旋转向量重新映射回原坐标轴中，要借助这个逆矩阵
    #下面把旋转向量求出来
    RotationVector = np.empty(shape=(1,3))
    RotationVector[0,0] = cos((angle/360)*2*pi)
    RotationVector[0,1] = sin((angle/360)*2*pi) # 不可忘记角度和弧度的转换
    RotationVector[0,2] = 0

    pointAfterRotation1 = np.dot(RotationVector,k_inv)
    pointAfterRotation = list()
    for k in range(3):
        pointAfterRotation.append(pointAfterRotation1[0,k])
    pointAfterRotation = np.array(pointAfterRotation)
    pointAfterRotation = pointAfterRotation + standPoint
    return [standPoint,standVector1,standVector2,axelVector,pointAfterRotation]

    


result = flattenGet(x,y,z)
#print("当前角度%f,当前方差%f"%(0,result[3]))

angleList = np.arange(-0.1,0.1,float(args.precision))
bestR = 100
bestResult = 0
bestangle = 0
bestF = 100
bestFResult = 0
bestFangle = 0
angleList1 = []
RList = []
flist = []
for anglek in angleList:
    pointListX = np.tile(x,1)
    pointListY = np.tile(y,1)
    pointListZ = np.tile(z,1)
    result = flattenGet(pointListX,pointListY,pointListZ)
    for k in range(7,14):
        x_get = x[k]
        y_get = y[k]
        z_get = z[k]
        pointBeforeRo = np.array([x_get,y_get,z_get])
        pointAfterRo = pointRotation(pointBeforeRo,circleCenterPoint,CharaVector,float(anglek))
        pointListX[k] = pointAfterRo[4][0]
        pointListY[k] = pointAfterRo[4][1]
        pointListZ[k] = pointAfterRo[4][2]
    
    result = flattenGet(pointListX,pointListY,pointListZ)
    
    if result[3] < bestR:
        bestR = result[3]
        bestResult = result
        bestangle = anglek
    if result[4] < bestF:
        bestF = result[4]
        bestFResult = result
        bestFangle = anglek
    #print("当前角度%f,当前方差%f,平面度误差%f"%(anglek,result[3],result[4]))
    #print(anglek)
    RList.append(result[3])
    angleList1.append(anglek)
    flist.append(result[4])


print('\n平面度误差最优结果为角度=%f时,曲面方程z =  (%.10f)* x + ( %.10f)* y + (%.10f)' % (bestFangle,result[0],result[1],result[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(bestFResult[4]))

with open('./result/result3.txt','a') as f:
    f.writelines('%s %.10f,%.10f\n'%(args.file_name,bestFangle,bestFResult[4]))
with open('./result/result3.csv', 'a') as f:
    f.writelines('%s ,%.10f,%.10f\n' % (args.file_name, bestFangle, bestFResult[4]))