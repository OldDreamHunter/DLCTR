import pandas as pd
import os
import xgboost as xgb
import joblib
from ..Utils import xgbTranformPy

def hiveTransformUDF(modelFileName,inputTableName,outputTableName,featureCols, key):
    """
    :param modelFileName: complete directory for udf load file
    :param inputTableName: inputTableName contains the DataBase
    :param outputTableName: outputTableName contains the DataBase
    :param featureCols: the model inputFeatureNames
    :return:
    """
    hivesql = "use tmp;\n"
    hivesql += "add file {modelFileName};\n".format(modelFileName=os.getcwd()+'/'+modelFileName)
    hivesql += "drop table if exists {outputTableName};\n".format(outputTableName=outputTableName)
    hivesql += "create table {outputTableName} as".format(outputTableName=outputTableName)
    hivesql += " select transform({featureCols})".format(featureCols=key+','+','.join(featureCols))
    hivesql += " using 'python {modelFileName}'".format(modelFileName=modelFileName)
    hivesql += " as (pin,predictScore)"
    hivesql += " from {inputTableName};".format(inputTableName=inputTableName)
    return hivesql

def serverPredict(modelFileName, inputTableName, outputTableName, key):
    UDFModelFileName = modelFileName.split('.')[0] + '_transform.py'
    xgbModel = joblib.load(modelFileName)
    featureCols = xgbModel.get_booster().feature_names
    xgbTranformPy(xgbModel, UDFModelFileName, key, featureCols)
    hivesql = hiveTransformUDF(UDFModelFileName, inputTableName, outputTableName, featureCols, key)
    print(hivesql)
    print('\n--------------------------------start to hive udf function--------------------------')
    os.system('''hive -e "{hivesql}"'''.format(hivesql=hivesql))
    

def localPredict(modelFileName, inputTableName, outputTableName, key):
    """
    :param modelFileName:
    :param inputTableName: inputTable contains keys and featureCols(keys == user_pin)
    :param outputTableName: outputTableName + '.csv'
    :param featureCols:
    :return:
    """
    inputTable = pd.read_table(inputTableName, sep='\t')
    pinCol = key
    xgbModel = xgb.Booster(model_file=modelFileName)
    featureCols = xgbModel.get_booster().feature_names
    inputPredictTable = inputTable.loc[:, featureCols]
    inputPredictTable = xgb.DMatrix(inputPredictTable)
    predictY= xgbModel.predict(inputPredictTable)
    predictResult = pd.DataFrame({key:inputTable.loc[:,pinCol], 'predictScore':predictY})
    predictResult.to_csv(outputTableName,index=None)
    return predictResult

if __name__ == "__main__":
    modelFileName = 'modelTemp.py'
    inputTableName = 'pf_xgbPredict'
    outputTableName = 'pf_xgbPredictResult'
    featureCols = ','.join(['f1', 'f2', 'f3'])
    result = hiveTransformUDF(modelFileName=modelFileName,
                              inputTableName=inputTableName,
                              outputTableName=outputTableName,
                              featureCols=featureCols)
    print(result)
