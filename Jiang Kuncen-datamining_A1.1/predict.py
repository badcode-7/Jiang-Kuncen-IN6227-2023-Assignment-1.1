import pandas as pd
import matplotlib.pyplot as plt
import json
import main
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import graphviz
import sklearn

def predicindex(predf,mytree):
    corr=0
    ytrure = predf['income'].tolist()
    numsample=predf.shape[0]
    ypredic=[]
    for index,row in predf.iterrows():
        ypredic.append(predict(mytree, row))

    acc=[accuracy(ytrure,ypredic),accuracy(ytrure,ypredic)]
    print(f'acc:{accuracy(ytrure,ypredic):.2f}\n')
    pres=[]
    recs=[]
    F1s=[]
    # 计算每个类别的支持数、精确度
    for class_label in set(ytrure):
        support = calculate_support(ytrure, class_label)
        pre = precision(ytrure, ypredic, class_label)
        pres.append(pre)
        rec = recall(ytrure, ypredic, class_label)
        recs.append(rec)
        F1 = f1_score(ytrure, ypredic, class_label)
        F1s.append(F1)
        print(f'Class {class_label}: Support = {support}, precision= {pre:.2f},recall={rec:.2f},F1= {F1:.2f}\n')
    # 计算宏平均和加权平均
    print('  \tprecision         \trecall        \tf1_score')
    macro_avg = calculate_macro_avg(ytrure, ypredic)
    weighted_avg = calculate_weighted_avg(ytrure, ypredic)


    print(f'Macro\tAvg Precision = {macro_avg[0]:.2f},\tAvg recall={macro_avg[1]:.2f},\tAvg f1_score={macro_avg[2]:.2f}')
    print(f'Weighted\tAvg Precision = {weighted_avg[0]:.2f},\tAvg recall={weighted_avg[1]:.2f},\tAvg f1_score={weighted_avg[2]:.2f}\n')
    # 计算混淆矩阵
    cm = confusion_matrix(ytrure, ypredic)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    index=[acc,pres,recs,F1s]
    return index


#Accuracy：
def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


#Precision：
def precision(y_true, y_pred, positive_class):
    TP = 0
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] == positive_class and y_pred[i] == positive_class:
            TP += 1
        elif y_pred[i] == positive_class and y_true[i] != positive_class:
            FP += 1
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)
#Recall：
def recall(y_true, y_pred, positive_class):
    TP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == positive_class and y_pred[i] == positive_class:
            TP += 1
        elif y_pred[i] != positive_class and y_true[i] == positive_class:
            FN += 1
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

#F1-Score
def f1_score(y_true, y_pred, positive_class):
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

# 计算支持数（每个类别的样本数量）
def calculate_support(true_labels, class_label):
    return sum(1 for label in true_labels if label == class_label)

# 计算宏平均
def calculate_macro_avg(true_labels, predicted_labels):
    num_classes = len(set(true_labels))
    precision_sum = sum(precision(true_labels, predicted_labels, class_label) for class_label in set(true_labels))
    recall_num=sum(recall(true_labels, predicted_labels, class_label) for class_label in set(true_labels))
    f1_num=sum(f1_score(true_labels, predicted_labels, class_label) for class_label in set(true_labels))
    result=[precision_sum / num_classes,recall_num/num_classes,f1_num/num_classes]
    return result

# 计算加权平均
def calculate_weighted_avg(true_labels, predicted_labels):
    num_classes = len(set(true_labels))
    weighted_precision_sum = sum(precision(true_labels, predicted_labels, class_label) * calculate_support(true_labels, class_label) for class_label in set(true_labels))
    weighted_recall_sum=sum(recall(true_labels, predicted_labels, class_label) * calculate_support(true_labels, class_label) for
        class_label in set(true_labels))
    weighted_F1_sum=sum(f1_score(true_labels, predicted_labels, class_label) * calculate_support(true_labels, class_label) for
        class_label in set(true_labels))
    total_support = sum(calculate_support(true_labels, class_label) for class_label in set(true_labels))
    result=[weighted_precision_sum / total_support,weighted_recall_sum / total_support,weighted_F1_sum / total_support]
    return result

def predict(tree, sample):
    # 递归函数，用于遍历决策树并进行预测
    if isinstance(tree, dict):
        # 获取当前节点的特征
        feature = list(tree.keys())[0]
        if "?" in feature:
            # 定义一个正则表达式模式来匹配这种模式的字符串
            pattern = r'(.+?)(小于|大于)(.+?)\?'

            # 使用正则表达式匹配模式
            match=re.match(pattern,feature)
            really_feature = match.group(1)
            choose_value=float(match.group(3))
            feature_value = sample[really_feature]
            if feature_value>choose_value:
                next_node = tree[feature].get('大于')
            else:
                next_node=tree[feature].get('小于')
            if next_node is None:
                # 如果特征值在树中不存在，返回默认值或进行适当处理
                return ' >50K'
            # 递归调用以继续遍历子树
            return predict(next_node, sample)
        else:
            really_feature=feature
            # 获取样本中的特征值
            feature_value = sample[really_feature]
            # 获取下一个子树或叶子节点
            next_node = tree[feature].get(feature_value)
            if next_node is None:
                # 如果特征值在树中不存在，返回默认值或进行适当处理
                return ' >50K'
            # 递归调用以继续遍历子树
            return predict(next_node, sample)
        #if feature=='income':
            #return 'Notkown'

    else:
        # 如果当前节点是叶子节点，直接返回类别或决策
        return tree

columns_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
process_label=['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']

my_tree='my_tree.json'
with open(my_tree, 'r') as json_file:
    mytree = json.load(json_file)

# 训练集处理
traindf = pd.read_csv('adult.data', names=columns_names)
traindf = main.preprocess(traindf, process_label)
#测试集处理
testdf=pd.read_csv('adult.test',names=columns_names)
#去掉第一行
testdf=testdf.drop(testdf.index[0])
testdf=main.preprocess(testdf,process_label)
testdf['income']=testdf['income'].str[:-1]
#观察分类标签分布
tra_rich=0
tra_pow=0
test_rich=0
test_pow=0
for income in traindf['income']:
    if income==' >50K':
        tra_rich+=1
    if income==' <=50K':
        tra_pow+=1

for income in traindf['income']:
    if income==' >50K':
        test_rich+=1
    if income==' <=50K':
        test_pow+=1

train_income_metrics = {'>50k': tra_rich, '<=50k': tra_pow}
plt.bar(train_income_metrics.keys(), train_income_metrics.values())
plt.ylabel('income')
plt.title('train_set income label separation')
plt.show()

test_income_metrics = {'>50k': test_rich, '<=50k': test_pow}
plt.bar(test_income_metrics.keys(), test_income_metrics.values())
plt.ylabel('income')
plt.title('test_set income label separation')
plt.show()

print('训练集表现：\n')
train_out=predicindex(traindf,mytree)
train_acc=train_out[0][1]
train_pre=train_out[1][1]
train_rec=train_out[2][1]
train_F1=train_out[3][1]

print('验证集表现：\n')
test_out=predicindex(testdf,mytree)
test_acc=test_out[0][1]
test_pre=test_out[1][1]
test_rec=test_out[2][1]
test_F1=test_out[3][1]


#性能评估及可视化
# 创建性能指标图表
train_performance_metrics = {'Accuracy': train_acc, 'Precision': train_pre, 'Recall': train_rec, 'F1 Score': train_F1}
plt.bar(train_performance_metrics.keys(), train_performance_metrics.values())
plt.ylabel('Score')
plt.title('train_set Performance Metrics')
plt.show()

test_performance_metrics = {'Accuracy': test_acc, 'Precision': test_pre, 'Recall': test_rec, 'F1 Score': test_F1}
plt.bar(test_performance_metrics.keys(), test_performance_metrics.values())
plt.ylabel('Score')
plt.title('test_set Performance Metrics')
plt.show()

