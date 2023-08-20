
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
import argparse
import time

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")
parser.add_argument("-g", "--generation", help="generation")
args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)

# print('方法七 网格搜索最优轴拟合')

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
        time1 = time.time()
        result = flatten.pointRotation(flatten.xyzMat2coordinate(x, y, z, 7 + i), np.array(pointImp), np.array(axel),
                                       angel)
        time2 = time.time()
        using_time = time2 - time1
        # print("usingtime:%d")
        xList[i + 7], yList[i + 7], zList[i + 7] = result[4][0], result[4][1], result[4][2]
    flatResult = flatten.FlattenGet(xList, yList, zList)
    return (flatResult[4])


def grid_search():
    k_list = np.linspace(-1e15, 1e15, 100)
    b_list = np.linspace(-1e3, 1e3, 100)
    angel_list = np.linspace(-5, 5, 100)
    min_flatness = float('inf')
    best_k, best_b, best_angel = 0, 0, 0
    for k in k_list:
        for b in b_list:
            for angel in angel_list:
                flatness = target([k, b, angel])
                if flatness < min_flatness:
                    min_flatness = flatness
                    best_k, best_b, best_angel = k, b, angel
    return best_k, best_b, best_angel, min_flatness


best_k, best_b, best_angel, min_flatness = grid_search()
print('\n平面度误差最优结果为出现在以y=%.10fx+%.10f为轴时,旋转角度=%.10f' % (best_k, best_b, best_angel)) # 保留5位小数
print('此时的平面度误差为%.10f'%(min_flatness))

with open('./result/result7.txt','a') as f:
    f.writelines('%s %.10f,%.10f,%.10f %.10f\n'%(args.file_name,best_k,best_b,best_angel,min_flatness))
with open('./result/result7.csv', 'a') as f:
    f.writelines('%s ,%.10f,%.10f,%.10f,%.10f\n' % (args.file_name,best_k,best_b,best_angel,min_flatness))