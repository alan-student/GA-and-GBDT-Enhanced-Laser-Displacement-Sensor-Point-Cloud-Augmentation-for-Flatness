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

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")
parser.add_argument("-g", "--generation", help="generation")
args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)

# print('方法五 遗传算法搜索最优轴拟合')

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
print(z)
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


@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    f = target(Vars)
    CV = [0] * 3
    return f, CV


problem = ea.Problem(name='soea quick start demo',
                     M=1,  # 目标维数
                     maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                     Dim=3,  # 决策变量维数
                     varTypes=[0] * 3,  # 决策变量的类型列表，0：实数；1：整数
                     lb=[-1e15, -1e3, -1],  # 决策变量下界
                     ub=[1e15, 1e3, 1],  # 决策变量上界
                     evalVars=evalVars)
# 构建算法（soea_SEGA_templet，Ctrl+鼠标箭头，跳转）
algorithm = ea. soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=100),
                                 MAXGEN=int(args.generation),  # 最大进化代数。
                                 logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                 trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                 maxTrappedCount=int(args.generation))  # 进化停滞计数器最大上限值。
# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=0, outputMsg=True, drawLog=False, saveFlag=True,dirName='./result')#drawing=1,表示画图，0表示不画图
vars = res['Vars'][0]
flatness = res['ObjV'][0]
print(vars,flatness)

print('\n平面度误差最优结果为出现在以y=%.10fx+%.10f为轴时,旋转角度=%.10f' % (vars[0],vars[1],vars[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(flatness))

with open('./result/result5.txt','a') as f:
    f.writelines('%s %.10f,%.10f,%.10f %.10f\n'%(args.file_name,vars[0],vars[1],vars[2],flatness))
with open('./result/result5.csv', 'a') as f:
    f.writelines('%s ,%.10f,%.10f,%.10f,%.10f\n' % (args.file_name, vars[0], vars[1],vars[2], flatness))


