import os
import time
import numpy as np
Key1,Key2,Key3,Key4,Key5,Key6 = False,False,False,False,False,False

'''用户参数设置区
（注：Key1、Key2不需要搜索，直接拟合并计算）
（注：Key3不需要搜索，指定旋转轴，拟合并计算）
（注：Key4、Key5、Key6需要搜索最优旋转轴，通过最优旋转轴来确定平面度误差最小）
    Key1 十四点拟合
    Key2 七点拟合
    Key3 轴移动半径的一半拟合
    Key4 任意两点作轴搜索最优轴拟合
    Key5 遗传算法搜索最优轴拟合
    Key6 粒子群算法搜索最优轴拟合
    如果有不想使用的方法，请在下面对应处注释
    precision 角度搜索精度（步长）设定
    generationGA 遗传算法遗传次数
    generationPSO 粒子群算法遗传次数
'''

# Key1 = True
# Key2 = True
# Key3 = True
# Key4 = True
# Key5 = True
# Key6 = True
Key7 = True

Key1 = False
Key2 = False
Key3 = False
Key4 = False
Key5 = False
Key6 = False
# Key7 = False

precision = 1e-4    #移动步长
generationGA = 50  #遗传算法迭代次数
generationPSO = 200  #粒子群算法迭代次数


print("###########  铝板  ############")

fileList = os.listdir('./result')
if ((not('fig3' in fileList)) and Key3):
    os.mkdir('./result/fig3')
if ((not('fig4' in fileList)) and Key4):
    os.mkdir('./result/fig4')

print("###########一、Fourteen-LSM############")
#Key1 一、十四点直接拟合
time1 = time.time()
if Key1:
    with open('./result/result1.txt','a') as f:
        f.writelines('fileName flatNess\n')

    for i in range(1,51):
        os.system('python ./key1.py -f %d.npz'%(i))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))

print("###########二、Seven-LSM算法############")
#Key2 二、七点直接拟合
time1 = time.time()
if Key2:
    with open('./result/result2.txt','a') as f:
        f.writelines('fileName flatNess1 flatNess2\n')

    for i in range(1,51):
        os.system('python ./key2.py -f %d.npz'%(i))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))

print("###########三、Angel-Search-LSM算法############")
#Key3 三、中心轴移动半径的一半R/2拟合
time1 = time.time()
if Key3:
    with open('./result/result3.txt','a') as f:
        f.writelines('fileName angel flatNess\n')

    for i in range(1,51):
        os.system('python ./key3.py -f %d.npz -p %f'%(i,precision))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))

print("###########四、Random-Two-Axle-LSM############")
#Key4 四、任意两点作轴搜索最优轴拟合
time1 = time.time()
if Key4:
    with open('./result/result4.txt','a') as f:
        f.writelines('fileName twoPoint angel flatNess\n')

    for i in range(1,51):
        os.system('python ./key4.py -f %d.npz -p %f'%(i,precision))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))

print("###########五、GA-LSM算法（本课题方法）############")
#Key5 五、遗传算法搜索最优轴拟合
time1 = time.time()
if Key5:
    with open('./result/result5.txt','a') as f:
        f.writelines('fileName k,b angle flatNess\n')
    for i in range(1,51):
        os.system('python ./key5.py -f %d.npz -g %d'%(i,generationGA))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))

print("###########六、PSO-LSM算法############")
#Key6 六、粒子群算法搜索最优轴拟合
time1 = time.time()
if Key6:
    with open('./result/result6.txt','a') as f:
        f.writelines('fileName k,b flatNess\n')
    for i in range(1,51):
        os.system('python ./key6.py -f %d.npz -g %d'%(i,generationPSO))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))


print("###########七、网格-LSM算法############")

time1 = time.time()
if Key7:
    with open('./result/result7.txt','a') as f:
        f.writelines('fileName k,b angle flatNess\n')
    for i in range(1,51):
        os.system('python ./key7.py -f %d.npz -g %d'%(i,generationGA))
time2 = time.time()
print("测试用时%.7f"%((time2-time1)/50))
