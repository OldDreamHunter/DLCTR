from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def gridSearchCV(model,param,scoring='roc_auc'):
    pass

def stabilityTest(model,X_test,Y_test,kfold_num):
    skf = StratifiedKFold(n_splits=kfold_num,shuffle=True, random_state=7)
    X_test.reset_index(drop=True, inplace=True)
    Y_test = np.array(Y_test)
    auc_result = []
    for train_x_range, test_x_range in skf.split(X_test, Y_test):
        test_x = X_test.iloc[test_x_range, :]
        test_y = Y_test[test_x_range]
        predict_y = model.predict(test_x)
        auc_result.append(round(roc_auc_score(test_y, predict_y),3))
    return auc_result
