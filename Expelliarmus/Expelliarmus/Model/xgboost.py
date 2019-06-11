# -*- coding: UTF-8 -*-
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from ..CrossValidation import stabilityTest
from ..Utils import xgbTranformPy
from ..Utils import hiveTransformUDF
from ..Utils import localPredict, serverPredict
import pandas as pd
import os
import joblib
"""
1. model_selection 
##  grid_search: apply grid search to the parameters
eta:learning_rate;
gamma:min_split_loss(提早停止的条件，gamma越大，不容易过拟合)；
max_depth:树的深度越大，值越大，越容易过拟合；
min_child_weight:最小叶子节点样本权重和
subsample:参数控制每棵树随机采样的比例，减少参数的值，避免过拟合（采样过度或导致模型过拟合）
colsample_bytree: 每棵树column的采样，太多会造成欠拟合
n_estimators:弱学习器的数量

params = {'learning_rate': 0.3, 
          'n_estimators': 500, 
          'gamma': 0, 
          'max_depth': 6, 
          'min_child_weight': 1,
          'colsample_bytree': 1, 
          'subsample': 1}

#kfold: apply stratifiedkfold to the x features

#gridsearchcv and kfold concat
"""

class autoXgboost (object):
    def __init__(self,kfold_num=10,train_test_split=0.2,sample_rowNumber = 10000):
        self.kfold_num = kfold_num
        self.train_test_split = train_test_split
        self.sample_rowNumber = sample_rowNumber
        return

    def xgbtransform(self,data,label,features):
        x = data.loc[:,features]
        y = data.loc[:,label]
        return x,y

    def train(self, inputFileName, model_file, label, featureCols, key, executor='local'):
        if executor == 'local':
            inputDataFrame = pd.read_table(inputFileName)
        else:
            os.system("hive -e 'set hive.cli.print.header=True;select * from %s sort by rand() limit %d' -> %s"%(inputFileName,self.sample_rowNumber,inputFileName))
            inputDataFrame = pd.read_table(inputFileName)
            
        x,y = inputDataFrame.loc[:,featureCols],inputDataFrame.loc[:,label]
        kfold = StratifiedKFold(n_splits=self.kfold_num, shuffle=True, random_state=7)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=self.train_test_split, random_state=7)
        model = XGBClassifier()

        learning_rate =[0.01, 0.05, 0.07, 0.1, 0.2]
        n_estimators = [400,500,600,700,800]
        max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
        min_child_weight = [1, 2, 3, 4, 5, 6]
        gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        subsample = [0.6,0.7,0.8,0.9]
        colsample_bytree = [0.6,0.7,0.8,0.9]

        param_grid = {'learning_rate':learning_rate,
                      'gamma':gamma,
#                       'n_estimators':n_estimators,
#                       'max_depth':max_depth,
#                       'min_child_weight':min_child_weight,
#                       'subsample':subsample,
                      'colsample_bytree':colsample_bytree}

        grid_search = GridSearchCV(model,param_grid,scoring='roc_auc',n_jobs=-1,cv=kfold,verbose=5)
        grid_result = grid_search.fit(X_train,Y_train)
        """
        grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
        best_score_：成员提供优化过程期间观察到的最好的评分
        具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
        注意，“params”键用于存储所有参数候选项的参数设置列表。
        """
        print('\n-------------------------------Best Model-----------------------------')
        print("Best: %f using %s" %(grid_result.best_score_, grid_search.best_params_))

        print('\n--------------------------Result of Each Param------------------------')
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with:   %r" % (mean, param))

        print('\n------------------------Cross Validation of Models--------------------')
        param = grid_search.best_params_     
        xgbModel = grid_result.best_estimator_
        cv_result = stabilityTest(xgbModel, X_train, Y_train, self.kfold_num)
        print(cv_result)
        
        print('\n---------------------Save the Best Model-----------------------')
        joblib.dump(xgbModel,model_file+'.model')
        
        if os.path.exists(inputFileName):
            print('\n-------------------------------model successfully saved----------------------------')
        else:
            print('\n-------------------------------model failed to be saved----------------------------')

    def predict(self, inputTableName, outputTableName, model_file, key, executor='server'):
        """
        :param model_file:
        :param inputTableName:
        :param outputTableName:
        :param executor:
        :return:
        """
        if executor == 'local':
            localPredict(model_file, inputTableName, outputTableName, key)
        if executor == 'server':
            serverPredict(model_file, inputTableName, outputTableName, key)
        print('\n-----------------------predict finished----------------------------')

if __name__ == '__main__':
    autoMLXGB = autoXgboost()
    autoMLXGB.predict()





