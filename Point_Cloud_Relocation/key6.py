from math import cos, pi, sqrt, sin
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
import flatten
import geatpy as ea
import argparse
from sko.PSO import PSO

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")
parser.add_argument("-g", "--generation", help="generation")
args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)

# print('方法六 粒子群算法搜索最优轴拟合')

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
flat0 = flatten.FlattenGet(x, y, z)  # 首先生成14个初始点的平面


def lineGet(flatA, flatB, flatC, lineK, lineB):
    MyFunc2DGetY = lambda x: lineK * x + lineB  # 用于取得特征点的y坐标
    MyFunc2DGetZ = lambda x, y: flatA * x + flatB * y + flatC  # 用于取得特征点的z坐标
    point1 = np.array([1, MyFunc2DGetY(1), MyFunc2DGetZ(1, MyFunc2DGetY(1))])
    point2 = np.array([100, MyFunc2DGetY(100), MyFunc2DGetZ(100, MyFunc2DGetY(100))])
    # 取得三维空间内两个特征点的坐标
    return (point2 - point1, point1)  # 返回旋转轴方向向量和旋转轴所过点


def target(para):
    k = para[0]
    b = para[1]
    angel = para[2]
    axel, pointImp = lineGet(flat0[0], flat0[1], flat0[2], k, b)
    xList, yList, zList = x, y, z
    for i in range(7):
        result = flatten.pointRotation(flatten.xyzMat2coordinate(x, y, z, 7 + i), np.array(pointImp), np.array(axel),
                                       angel)
        xList[i + 7], yList[i + 7], zList[i + 7] = result[4][0], result[4][1], result[4][2]
    flatResult = flatten.FlattenGet(xList, yList, zList)
    return (flatResult[4])

pso = PSO(func=target,dim=3 , pop=40, max_iter=150, lb=[-1e15, -1e3, -1], ub=[1e15, 1e3, 1], w=0.8, c1=0.5, c2=0.5)

for i in range(int(args.generation)):
    # vars,flatness = pso.run(1)
    pso.run(1)
    vars,flatness = pso.gbest_x,pso.gbest_y
    # print(vars,flatness)


print('\n平面度误差最优结果为出现在以y=%.10fx+%.10f为轴时,旋转角度=%.10f' % (vars[0],vars[1],vars[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(flatness))

with open('./result/result6.txt','a') as f:
    f.writelines('%s %.10f,%.10f %.10f\n'%(args.file_name,vars[0],vars[1],flatness))
with open('./result/result6.csv', 'a') as f:
    f.writelines('%s ,%.10f,%.10f,%.10f\n' % (args.file_name, vars[0], vars[1], flatness))



