This folder contains algorithms that use point cloud relocation, and in the experiments we tested a variety of different search methods.


Before you start, you need to prepare the data file, which has two main parts

The first is the X and Y coordinate files of the point cloud data. The X and Y coordinates are fixed. Write the X coordinates of 14 points in x.csv file, and write the Y coordinates of 14 points in y.csv file

The second is to use the voltage values collected by the sensor, which are saved in the test_data folder

The files should be in one-to-one correspondence since each plane needs to be sampled twice

The files starting with a represent the data collected before mechanical rotation. The file beginning with b indicates the data collected after mechanical rotation.

For example, let's start by collecting half of the plane to get a0-0.txt. Then we rotate the plane and collect the data of b0-0.txt.

Each txt file contains 1000 lines of data, which are collected continuously, similar to the following format

Index Time(ms)  AI0(V)  AI1(V)  AI2(V)  AI3(V)  AI4(V)  AI5(V)  AI6(V)

0 0.00000 3.82507 4.89059 4.92981 5.08621 5.68283 5.46021 5.12909

You are free to choose how many lines of code you want to perform the maximum likelihood distance.Once you have these txt files, you are free to design your own preprocessing method in data_pre.py



The collected data is stored into these files in easy-to-read data. But these data are raw voltage data.

For further filtering processing, and coding into the form of easy programming. We need to preprocess the data.

Run data_pre.py to finish the preprocessing. This script will automatically complete the filtering of the data and encode it into an npz file.

npz files are supported by the numpy package in python. Applying the characteristics of numpy can facilitate the construction of point cloud relocation algorithm in the later stage.



The point cloud relocation algorithm is contained in the files key1-key7. You don't need to call these files directly.

Use the Process.py script to call the point cloud relocation algorithm. key1-key10 These scripts provide different search strategies for relocation parameters.

For the construction of point cloud relocation algorithm and the search of relocation algorithm parameters, please refer to our paper.

These scripts rely on a library file called flattn.py that we built.

The different strategies for key1-key10 are described below.

key1.py: Directly complete the fitting of 14 point cloud data without point cloud relocation.

Key2.py: only 7 point cloud data of the first acquisition (before mechanical rotation) are fitted, and point cloud relocation is not performed

key3.py: The rotation axis is determined as the position of the theoretical symmetry axis before and after the mechanical rotation, the rotation Angle is relocated using the strategy search of grid search, and the repositioning is fitted.

key4.py: Arbitrarily select two points in the point cloud as the rotation axis, and search the rotation Angle using the grid search strategy to relocate, and fit after relocation.

key5.py: Use genetic algorithm to complete the search of rotation axis and rotation Angle, and fit after relocation.

key6.py: Use the PSO algorithm to search the rotation axis and rotation Angle, and fit after relocation.

key7.py: Use the grid algorithm to search the rotation axis and rotation Angle, and fit after relocation.







这个文件夹中包含了使用点云重定位的算法，在实验中我们测试了各种不同的搜索方法。

使用前，需要准备好数据文件，主要包含两个部分
首先是点云数据的X、Y坐标文件，X、Y坐标是固定的，在x.csv文件里依次写好14个点的X坐标，y.csv文件里依次写好14个点的Y坐标
其次是使用传感器采集到的电压值，保存在test_data文件夹中
文件应该是一一对应的，因为每个平面需要被采样两次
其中以a开头的文件表示是机械旋转前采集的数据。b开头的文件表示是机械旋转后采集的数据。
比如，我们先采集平面半边的数据得到a0-0.txt。然后我们旋转平面后，再采集b0-0.txt的数据。
每个txt文件中包含1000行数据，这些数据是连续采集的，类似于下面的格式
Index Time(ms)  AI0(V)  AI1(V)  AI2(V)  AI3(V)  AI4(V)  AI5(V)  AI6(V)  
0 0.00000  3.82507 4.89059 4.92981 5.08621 5.68283 5.46021 5.12909
你可以自由选择需要多少行代码进行最大似然估计距离，这些txt文件准备好后，可以在data_pre.py里自由设计预处理方法

采集到的数据是以方便阅读的数据存储到这些文件中的。但是这些数据是原始的电压数据。
为了进一步作滤波处理，并编码成易于编程的形式。我们需要对数据进行预处理。
请运行data_pre.py以完成预处理。这个脚本会自动完整数据的滤波，并且会将其编码成npz文件。
npz文件是由python的numpy包提供的支持。应用numpy的特性可以在后期方便完成点云重定位算法的构建。

点云重定位算法被包含在key1-key7这些文件中。使用时，不需要直接调用这些文件。
请使用Process.py脚本完成点云重定位算法的调用。key1-key10这些脚本提供了不同的重定位参数搜索策略。
关于点云重定位算法的构建以及重定位算法参数的搜索，请参考我们的论文。
这些脚本依赖我们构建的库文件flatten.py。
下面介绍key1-key10的不同策略。
key1.py: 直接完成14个点云数据的拟合，不进行点云重定位。
Key2.py: 只对第一次采集(机械旋转前)的7个点云数据拟合，不进行点云重定位
key3.py: 旋转轴被确定为机械旋转前后的理论对称轴的位置，旋转角度使用网格搜索的策略搜索进行重定位，重定位后拟合。
key4.py: 任意选取点云中的两点连线作为旋转轴，旋转角度使用网格搜索的策略搜索进行重定位，重定位后拟合。
key5.py：使用遗传算法完成旋转轴和旋转角度的搜索，重定位后拟合。
key6.py：使用PSO算法完成旋转轴和旋转角度的搜索，重定位后拟合。
key7.py：使用网格算法完成旋转轴和旋转角度的搜索，重定位后拟合。


