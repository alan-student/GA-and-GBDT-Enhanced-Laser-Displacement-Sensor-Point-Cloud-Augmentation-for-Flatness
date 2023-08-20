import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from sklearn import metrics
import pickle


plt.style.use('ggplot')
# %load_ext klab-autotime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树


f = open('./total.csv')#数据集路径
data = pd.read_csv(f)
data.info()
type(data)
data.head()
data.describe()
X, y = data[data.columns.delete(-1)], data['flatNess']#最后一列是平面度flatNess





print("###########开始训练模型############")

print("###########一、多元线性回归模型MLR############")


# 使用多元线性回归模型
for i in range(0, 1, 1):
    X, y = data[data.columns.delete(-1)], data['flatNess']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)# 测试集的比例为0.2，训练集的比例为0.8
    linear_model = LinearRegression()

    with open("MLR_machine_learing_result1.txt","w") as f:
        f.write("true,pre\n")
    '''
    时间
    '''
    time1 = time.time()
    linear_model.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f"%(time2-time1))
    coef = linear_model.coef_  # 回归系数
    time1 = time.time()
    line_pre = linear_model.predict(X_test)  #预测值
    time2 = time.time()
    print("推理用时%.6f"%(time2-time1))

    '''
    评价指标
    '''
    y_test = np.array(y_test)
    with open("MLR_machine_learing_result1.txt","a") as f:
        for i in range(len(line_pre)):
            f.write("%.4f,%.4f\n"%(y_test[i],line_pre[i]))

    y_train_pre = linear_model.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, line_pre)
    RMSE = metrics.mean_squared_error(y_test, line_pre) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, line_pre)
    MAPE = metrics.mean_absolute_percentage_error(y_test, line_pre)
    print('测试集__MSE',MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, line_pre) ** 2
    print("测试集__R2:%f" % (R2))

    plt.title("MLR")
    plt.xlabel("y_test")
    plt.ylabel("line_pre")
    plt.scatter(y_test, line_pre)
    # plt.show()


print("###########二、GradientBoosting（梯度提升）GBDT############")
# -----------------------# GradientBoosting（梯度提升）----------------------
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn import ensemble, __all__

    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}
    # clf = ensemble.GradientBoostingRegressor(**params)
    clf = ensemble.GradientBoostingRegressor()

    with open("GBDT_machine_learing_result2.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    clf.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    clf_pre = clf.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))
    # print('真实值y_test\n', y_test)#真实值
    # print('预测值clf_pre\n', clf_pre)#预测值

    y_test = np.array(y_test)
    with open("GBDT_machine_learing_result2.txt","a") as f:
        for i in range(len(clf_pre)):
            f.write("%.4f,%.4f\n"%(y_test[i],clf_pre[i]))


    '''
    评价指标
    '''

    y_train_pre = clf.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, clf_pre)
    RMSE = metrics.mean_squared_error(y_test, clf_pre) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, clf_pre)
    MAPE = metrics.mean_absolute_percentage_error(y_test, clf_pre)
    print('测试集__MSE',MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, clf_pre) ** 2
    print("测试集__R2:%f" % (R2))
    with open("GBDT.pkl", "wb") as f:
        pickle.dump(clf, f)

    plt.title("GBDT")
    plt.xlabel("y_test")
    plt.ylabel("clf_pre")
    plt.scatter(y_test, clf_pre)
    # plt.show()


print("###########三、XgBoost回归############")
# -----------------------# XgBoost回归----------------------
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    import xgboost as xgb


    clf = xgb.XGBRegressor()

    with open("xgboost_machine_learing_result2.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    clf.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    clf_pre = clf.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))
    # print('真实值y_test\n', y_test)#真实值
    # print('预测值clf_pre\n', clf_pre)#预测值

    y_test = np.array(y_test)
    with open("xgboost_machine_learing_result2.txt","a") as f:
        for i in range(len(clf_pre)):
            f.write("%.4f,%.4f\n"%(y_test[i],clf_pre[i]))


    '''
    评价指标
    '''

    y_train_pre = clf.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, clf_pre)
    RMSE = metrics.mean_squared_error(y_test, clf_pre) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, clf_pre)
    MAPE = metrics.mean_absolute_percentage_error(y_test, clf_pre)
    print('测试集__MSE',MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, clf_pre) ** 2
    print("测试集__R2:%f" % (R2))
    with open("xgBoost.pkl", "wb") as f:
        pickle.dump(clf, f)


print("###########三、Lasso回归############")
# Lasso 回归 （Least Absolute Shrinkage and Selection Operator）
# Lasso也是惩罚其回归系数的绝对值。
# 与岭回归不同的是，Lasso回归在惩罚方程中用的是绝对值，
# 而不是平方。这就使得惩罚后的值可能会变成0
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.linear_model import Lasso
    # lr=Lasso(alpha=0)
    lasso = Lasso(alpha=0)
    with open("Lasso_machine_learing_result3.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    lasso.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    y_predict_lasso = lasso.predict(X_test)
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))

    y_test = np.array(y_test)
    with open("Lasso_machine_learing_result3.txt","a") as f:
        for i in range(len(y_predict_lasso)):
            f.write("%.4f,%.4f\n"%(y_test[i],y_predict_lasso[i]))
    '''
    评价指标
    '''

    y_train_pre = lasso.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_lasso)
    RMSE = metrics.mean_squared_error(y_test, y_predict_lasso) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_lasso)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_lasso)
    print('测试集__MSE',MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, y_predict_lasso) ** 2
    print("测试集__R2:%f" % (R2))

    plt.title("Lasso")
    plt.xlabel("y_test")
    plt.ylabel("y_predict_lasso")
    plt.scatter(y_test, y_predict_lasso)
    # plt.show()


print("###########四、决策树回归DT############")
# 决策树回归
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(max_depth=2)

    with open("DT_machine_learing_result4.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    tree_reg.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    tree_reg_pre = tree_reg.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))

    y_test = np.array(y_test)
    with open("DT_machine_learing_result4.txt","a") as f:
        for i in range(len(tree_reg_pre)):
            f.write("%.4f,%.4f\n"%(y_test[i],tree_reg_pre[i]))

    '''
    评价指标
    '''

    y_train_pre = tree_reg.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, tree_reg_pre)
    RMSE = metrics.mean_squared_error(y_test, tree_reg_pre) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, tree_reg_pre)
    MAPE = metrics.mean_absolute_percentage_error(y_test, tree_reg_pre)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, tree_reg_pre) ** 2
    print("测试集__R2:%f" % (R2))


    plt.title("DT")
    plt.xlabel("y_test")
    plt.ylabel("tree_reg_pre")
    plt.scatter(y_test, tree_reg_pre)
    # plt.show()



print("###########五、ElasticNet 回归############")
# ElasticNet 回归，ElasticNet回归是Lasso回归和岭回归的组合
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    enet = ElasticNet()
    parameters = {'alpha': [j*0.1 for j in range(1,100)], 'l1_ratio': [j*0.1 for j in range(10)]}
    clf = GridSearchCV(enet, parameters, cv=5, n_jobs= 6)
    clf.fit(X_train, y_train)
    enet = clf.best_estimator_
    with open("ElasticNet_machine_learing_result5.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    enet.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    y_predict_enet = enet.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))


    y_test = np.array(y_test)
    with open("ElasticNet_machine_learing_result5.txt","a") as f:
        for i in range(len(y_predict_enet)):
            f.write("%.4f,%.4f\n"%(y_test[i],y_predict_enet[i]))


    '''
    评价指标
    '''

    y_train_pre = enet.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_enet)
    RMSE = metrics.mean_squared_error(y_test, y_predict_enet) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_enet)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_enet)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, y_predict_enet) ** 2
    print("测试集__R2:%f" % (R2))

    plt.title("ElasticNet")
    plt.xlabel("y_test")
    plt.ylabel("y_predict_enet")
    plt.scatter(y_test, y_predict_enet)
    # plt.show()


print("###########六、随机森林 回归RF############")
# ElasticNet 回归，ElasticNet回归是Lasso回归和岭回归的组合
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.ensemble import RandomForestRegressor
    RFR = RandomForestRegressor()
    with open("RF_machine_learing_result6.txt","w") as f:
        f.write("true,pre\n")
    '''
    时间
    '''
    time1 = time.time()
    RFR.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    y_predict_RFR = RFR.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))

    '''
    评价指标
    '''
    y_test = np.array(y_test)
    with open("RF_machine_learing_result6.txt","a") as f:
        for i in range(len(y_predict_RFR)):
            f.write("%.4f,%.4f\n"%(y_test[i],y_predict_RFR[i]))

    y_train_pre = RFR.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_RFR)
    RMSE = metrics.mean_squared_error(y_test, y_predict_RFR) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_RFR)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_RFR)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor


    R2 = calc_corr(y_test, y_predict_RFR) ** 2
    print("测试集__R2:%f" % (R2))


    plt.title("RF")
    plt.xlabel("y_test")
    plt.ylabel("y_predict_RFR")
    plt.scatter(y_test, y_predict_RFR)
    # plt.show()





 

print("###########六、SVR 回归############") 
# SVR回归
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.svm import SVR

    # 定义SVR模型
    svr = SVR(C=1000, gamma=0.01, kernel='rbf')

    # 训练模型
    time1 = time.time()
    svr.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))

    # 预测
    time1 = time.time()
    y_predict_svr = svr.predict(X_test)
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))

    # 评价指标
    y_test = np.array(y_test)
    with open("SVR_machine_learing_result.txt", "w") as f:
        f.write("true,pre\n")
        for i in range(len(y_predict_svr)):
            f.write("%.4f,%.4f\n" % (y_test[i], y_predict_svr[i]))

    y_train_pre = svr.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_svr)
    RMSE = metrics.mean_squared_error(y_test, y_predict_svr) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_svr)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_svr)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor


    R2 = calc_corr(y_test, y_predict_svr) ** 2
    print("测试集__R2:%f" % (R2))


    plt.title("SVR")
    plt.xlabel("y_test")
    plt.ylabel("y_predict_SVR")
    plt.scatter(y_test, y_predict_svr)
    # plt.show()



























print("###########七、SVR 回归############")
# ElasticNet 回归，ElasticNet回归是Lasso回归和岭回归的组合
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV

    # 定义SVR模型
    svr = SVR()

    # # 定义要搜索的参数
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']

    # #     'C': [0.1],
    # #     'kernel': ['linear'],
    # #     'gamma': ['scale']
    }

    # # # 使用GridSearchCV进行参数搜索
    grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # # 训练模型
    time1 = time.time()
    grid_search.fit(X_train, y_train)
    # svr.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))

    # 获取最佳参数组合
    best_params = grid_search.best_params_
    print("最佳参数组合:", best_params)

    # 使用最佳参数组合训练SVR模型
    best_svr = SVR(**best_params)

    # 训练模型
    time1 = time.time()
    best_svr.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))

    # 预测
    time1 = time.time()
    y_predict_svr = best_svr.predict(X_test)
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))

    # 评价指标
    y_test = np.array(y_test)
    with open("SVR_machine_learing_result.txt", "w") as f:
        f.write("true,pre\n")
        for i in range(len(y_predict_svr)):
            f.write("%.4f,%.4f\n" % (y_test[i], y_predict_svr[i]))

    y_train_pre = best_svr.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_svr)
    RMSE = metrics.mean_squared_error(y_test, y_predict_svr) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_svr)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_svr)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor


    R2 = calc_corr(y_test, y_predict_svr) ** 2
    print("测试集__R2:%f" % (R2))


    plt.title("SVR")
    plt.xlabel("y_test")
    plt.ylabel("y_predict_SVR")
    plt.scatter(y_test, y_predict_svr)
    # plt.show()













print("###########五、多层感知机回归############")
# 多层感知机回归
for i in range(0, 1, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(10), alpha=0.001, learning_rate_init=0.001, max_iter=200)
    with open("MLP_machine_learing_result5.txt","w") as f:
        f.write("true,pre\n")

    '''
    时间
    '''
    time1 = time.time()
    mlp.fit(X_train, y_train)
    time2 = time.time()
    print("训练用时%.6f" % (time2 - time1))
    time1 = time.time()
    y_predict_mlp = mlp.predict(X_test)  # 预测值
    time2 = time.time()
    print("推理用时%.6f" % (time2 - time1))


    y_test = np.array(y_test)
    with open("MLP_machine_learing_result5.txt","a") as f:
        for i in range(len(y_predict_mlp)):
            f.write("%.4f,%.4f\n"%(y_test[i],y_predict_mlp[i]))


    '''
    评价指标
    '''

    y_train_pre = mlp.predict(X_train)  # 预测值
    train_RMSE = metrics.mean_squared_error(y_train, y_train_pre) ** 0.5
    print('训练集__RMSE', train_RMSE)
    MSE = metrics.mean_squared_error(y_test, y_predict_mlp)
    RMSE = metrics.mean_squared_error(y_test, y_predict_mlp) ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_predict_mlp)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_predict_mlp)
    print('测试集__MSE', MSE)
    print('测试集__RMSE', RMSE)
    print('测试集__MAE', MAE)
    print('测试集__MAPE', MAPE)

    def calc_corr(a, b):
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor
    R2 = calc_corr(y_test, y_predict_mlp) ** 2
    print("测试集__R2:%f" % (R2))



