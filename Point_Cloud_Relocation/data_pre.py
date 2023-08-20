import numpy as np
import os
import scipy.stats as norm
from scipy.stats import rv_continuous
import matplotlib as mpl
import matplotlib.pyplot as plt

dataFileFolder = './test_data'    #采集的原始坐标z值的文件

# Used to convert voltage values to distance values
def Voltage2Displacement(inputData):
    return inputData*0.8


# Processing Method 1: Take the average directly
def getAverage(fileName):
    standData = np.zeros(shape=(7))
    with open(dataFileFolder+'/'+fileName) as f:
        line_count = len(f.readlines())
    with open(dataFileFolder+'/'+fileName) as f:
        standDataRaw = f.readline()  # emove the label from the first line
        for i in range(line_count-1):            # Each txt contains 1,000 rows of data
            standDataRaw = f.readline()
            standDataRaw = standDataRaw.split(' ')
            standDataLine = np.empty(shape=(7))
            for j in range(3,9):
                standDataLine[j-3] = float(standDataRaw[j])  # The first two pieces of data in each row are the index and the time
            standDataLine[6] = float(standDataRaw[9].split('\\')[0]) # The last data in each row contains a newline character to be removed
            standData = standData + standDataLine
        standData = standData/1000
    return standData

# Processing Method 2: Use the maximum likelihood estimate
def getMLE(fileName):
    standDataList = list()
    standData = np.zeros(shape=(7))
    with open(dataFileFolder+'/'+fileName) as f:
        standDataRaw = f.readline()
        for i in range(1000):            # Each txt contains 1,000 rows of data
            standDataRaw = f.readline()
            standDataRaw = standDataRaw.split(' ')
            standDataLine = np.empty(shape=(7))
            for j in range(3,9):
                standDataLine[j-3] = float(standDataRaw[j])
            standDataLine[6] = float(standDataRaw[9].split('\\')[0])
            standDataList.append(standDataLine)
    # Calculate the maximum likelihood estimate
    for i in range(7):
        normList = list()
        for eachStandData in standDataList:
            normList.append(eachStandData[i])
        # plt.hist(normList, bins=100)  # 直方图显示
        # plt.show()
        plt.hist(normList, bins=100)  # 直方图显示
        # plt.show()

        # Fit the histogram of normList and plot the Gaussian distribution curve
        mu, sigma = norm.norm.fit(normList)
        # x = np.linspace(min(normList), max(normList), 100)
        # y = norm.norm.pdf(x, mu, sigma)
        # plt.plot(x, y, 'r-', label='Gaussian Fit')
        # plt.legend()
        # plt.show()

        standData[i] = mu
    return standData
        



fileList = os.listdir(dataFileFolder)
if not ("flatNP" in os.listdir("./")):
    os.mkdir("flatNP")
if not ("result" in os.listdir("./")):
    os.mkdir("result")

for i in range(0, int(len(fileList)/2-1)):
    z1 = getAverage('a%d-0.txt' % (i))
    z2 = getAverage('b%d-0.txt' % (i))

    # z1 = getMLE('a%d-0.txt' % (i))
    # z2 = getMLE('b%d-0.txt' % (i))

    z1 = Voltage2Displacement(z1)
    z2 = Voltage2Displacement(z2)

    np.savez('./flatNP/%d.npz' % (i), z1=z1, z2=z2)  # Each group of processed data is put into the same npz file


    with open('./flatCSV.csv', 'a') as f:
        for j in range(7):
            f.writelines('%.10f,'%(z1[j]))
        for j in range(7):
            f.writelines('%.10f,' %(z2[j]))
        f.writelines('\n')


            

