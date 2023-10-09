import pandas as pd
import math
import operator
import json
from sklearn.preprocessing import StandardScaler
import time
import main
import predict
import matplotlib.pyplot as plt

# read data
columns_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
process_label=['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']
# read data
traindf=pd.read_csv('adult.data',names=columns_names)
#trainset 预处理
traindf=main.preprocess(traindf,process_label)

testdf=pd.read_csv('adult.test',names=columns_names)
#去掉第一行
testdf=testdf.drop(testdf.index[0])
testdf=main.preprocess(testdf,process_label)
testdf['income']=testdf['income'].str[:-1]


def optim(traindf,testdf,columns_names,maxdepth):
    labels=columns_names
    dataset_list=traindf.to_numpy().tolist()
    #开始生成决策树
    start_time = time.time()  # 记录开始时间
    mytree=main.createTree(dataset_list,labels,max_depth=maxdepth)
    end_time = time.time()  # 记录结束时间
    training_time = end_time - start_time  # 计算训练所花费的时间

    test_out=predict.predicindex(testdf, mytree)


    return test_out,training_time
    #predict testset



print(optim(traindf,testdf, columns_names, 1))
acc=[]
lowpre=[]
lowrec=[]
lowf1=[]
highpre=[]
highrec=[]
highf1=[]
traintime=[]
for i in range(1,26):
    result,train_time=optim(traindf,testdf, columns_names, i)
    acc.append(result[0][0])#acc
    lowpre.append(result[1][1])#pre
    lowrec.append(result[2][1])#rec
    lowf1.append(result[3][1])#f1
    highpre.append(result[1][0])
    highrec.append(result[2][0])
    highf1.append(result[3][0])
    traintime.append(train_time)#time


depth=list(range(1,26))

plt.plot(depth,acc,color='b',label='acc')
plt.plot(depth,lowpre,marker='o',color='r',label='low income pre')
plt.plot(depth,highpre,marker='o',color='r',linestyle='-',label='high income pre')
plt.plot(depth,lowrec,marker='x',color='g',label='low income rec')
plt.plot(depth,highrec,marker='x',color='g',linestyle='--',label='high income rec')
plt.plot(depth,lowf1,marker='s',color='y',label='low income f1')
plt.plot(depth,highf1,marker='s',color='y',linestyle=':',label='high income f1')
plt.title('The change of decision tree performance index with maximum depth')
plt.xlabel('Maximum depth of the decision tree')
plt.ylabel('Performance index')

# 添加网格线
plt.grid(True)

# 添加图例
plt.legend()


plt.show()