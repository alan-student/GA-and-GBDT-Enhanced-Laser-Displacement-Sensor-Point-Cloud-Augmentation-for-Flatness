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
import flatten

mpl.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")

args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)

# print("方法二 7点拟合直接得平面")
# print('%s 文件已读取，开始解析'%(args.file_name))

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


result = flatten.FlattenGet7Only(x1,y1,z1)

print('\n前七点曲面方程z =  (%.10f)* x + ( %.10f)* y + (%.10f)' % (result[0],result[1],result[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(result[4]))

result2 = flatten.FlattenGet7Only(x2,y2,z2)

print('\n后七点曲面方程z =  (%.10f)* x + ( %.10f)* y + (%.10f)' % (result2[0],result2[1],result2[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(result2[4]))

with open('./result/result2.txt','a') as f:
    f.writelines('%s %.10f %.10f\n'%(args.file_name,result[4],result2[4]))

with open('./result/result2.csv','a') as f:
    f.writelines('%s,%.10f,%.10f\n'%(args.file_name,result[4],result2[4]))