# -*- coding:utf-8 -*-
import os

def getHiveData(inputTableName, dt = None, rowNumber = 1000000):
    if dt == None:
        os.system("hive -e 'set hive.cli.print.header=True;select * "
              "from %s order by rand() limit %d' -> %s" %(inputTableName, rowNumber, inputTableName))
    else:
        os.system("hive -e 'set hive.cli.print.header=True;select * "
              "from %s where dt = %s order by rand() limit %d' -> %s" %(inputTableName, dt, rowNumber, inputTableName))
    if os.path.exists(inputTableName): return True
    else: return False


