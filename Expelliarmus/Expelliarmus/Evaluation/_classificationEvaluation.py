# -*- coding=utf-8 -*-
from ..Utils import getHiveData
from sklearn.metrics import roc_curve,f1_score,roc_auc_score
import pandas as pd

def classificationScore(inputTableName, predProb, predLabel, trueLabel, scoring='auc',executor='server',dt=None):
    """
    params:
    inputTableName : 输入的文件名，如果文件在hive上输入hive表名，如果文件在本地，输入本地文件地址
    dt: 如果输入的是hive表，并且是分区表，就指定分区，默认不带分区
    executor: 'server' 从hive上读数据，'local' 从本地读数据，默认server模式
    predProb: 预测概率值的列名
    predLabel: 预测标签的列名
    trueLabel: 真实标签的列名
    scoring: 评估的方法：支持f1_score,auc,ks, 'all'对应的是上述三个评估方法都使用
    """
    if executor=='server': getHiveData(inputTableName,dt=dt)
    inputTable = pd.read_table(inputTableName)
    y_true = inputTable[trueLabel]
    y_pred = inputTable[predLabel]
    y_score = inputTable[predProb]
    scoreHashTable = {'f1_score':f1_score,
                      'auc':roc_auc_score,
                      'ks':komogrov_smirnov,
                      'all':['f1_score','auc','ks']}
    if scoring not in scoreHashTable:
        print("%s doesn't support"%(scoring))
        return None
    else:
        score_f1 = f1_score(y_true, y_pred)
        score_auc = roc_auc_score(y_true, y_score)
        score_ks = komogrov_smirnov(y_true, y_score)
        if scoring == 'f1_score':
            return score_f1
        elif scoring == 'auc':
            return score_auc
        elif scoring == 'ks':
            return score_ks
        elif scoring == 'all':
            return {'f1_score':score_f1,'auc':score_auc,'ks':score_ks}

def komogrov_smirnov (y_true, y_score):
    fpr,tpr,threshold = roc_curve(y_true, y_score)
    ks = max(tpr-fpr)
    return ks




