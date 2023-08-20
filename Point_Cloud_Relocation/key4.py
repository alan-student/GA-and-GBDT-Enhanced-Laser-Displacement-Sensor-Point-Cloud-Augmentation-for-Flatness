import flatten
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

mpl.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", help="timer name")
parser.add_argument("-p", "--precision", help="precision")
args = parser.parse_args()

npzFile = np.load('./flatNP/'+args.file_name)

# print('方法四 十四点寻求最优轴')
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

angelList = np.arange(-0.1,0.1,float(args.precision))


def point2pointIsAxes(point1,point2):
    axelVector = point2 - point1
    xList = x
    yList = y
    zList = z
    bestR = 100
    bestResult = 0
    bestangle = 0
    bestF = 100
    bestFResult = 0
    bestFangle = 0
    
    RList = []
    flist = []
    for angel in angelList:
        for i in range(7):
            result = flatten.pointRotation(np.array([x2[i],y2[i],z2[i]]),np.array(point1),np.array(axelVector),angel)
            xList[7+j],yList[7+j],zList[7+j] = result[4][0],result[4][1],result[4][2]
        result = flatten.FlattenGet(xList,yList,zList)
        if result[3] < bestR:
            bestR = result[3]
            bestResult = result
            bestangle = angel
        if result[4] < bestF:
            bestF = result[4]
            bestFResult = result
            bestFangle = angel
        #print("当前角度%f,当前方差%f,平面度误差%f"%(angel,result[3],result[4]))
        #print(angel)
        RList.append(result[3])
        # angleList1.append(angel)
        flist.append(result[4])
    return(RList,flist,bestR,bestF,bestangle,bestFangle,bestResult,bestFResult)
epoch = 0
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(30)
ax=fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax.set_title("方差")

ax2.set_title("平面度误差")
R = 100
F = 100
for i in range(6):
   for j in range(i+1,7):
        result = point2pointIsAxes(np.array([x[i],y[i],z[i]]),np.array([x[j],y[j],z[j]]))
        ax.plot(angelList,result[0])
        ax.scatter(angelList,result[0],s=0.1,label='Point%d And Point%d'%(i,j))
        ax2.plot(angelList,result[1])
        ax2.scatter(angelList,result[1],s=0.1,label='Point%d And Point%d'%(i,j))
        if result[2] < R:
            R = result[2]
            bestAngel = result[4]
            bestResult = result[6]
            bestPoint1 = i
            bestPoint2 = j
        if result[3] < F:
            F = result[3]
            bestFAngel = result[5]
            bestFResult = result[7]
            bestFPoint1 = i
            bestFPoint2 = j


print('\n平面度误差最优结果为出现在以%d点和%d为轴时,旋转角度=%.10f,曲面方程z =  (%.10f)* x + ( %.10f)* y + (%.10f)' % (bestFPoint1,bestFPoint2,bestFAngel,bestFResult[0],bestFResult[1],bestFResult[2]) ) # 保留5位小数
print('此时的平面度误差为%.10f'%(bestFResult[4]))

with open('./result/result4.txt','a') as f:
    f.writelines('%s %d,%d %.10f %.10f\n'%(args.file_name,bestFPoint1,bestFPoint2,bestFAngel,bestFResult[4]))

with open('./result/result4.csv', 'a') as f:
    f.writelines('%s,%d,%d ,%.10f ,%.10f\n'%(args.file_name,bestFPoint1,bestFPoint2,bestFAngel,bestFResult[4]))


