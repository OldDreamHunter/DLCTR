import re
import sys
import math
import numpy as np
import xgboost as xgb


def string_parser(s):
    out = []
    if len(re.findall(r":leaf=", s)) == 0:
        out_tmp = re.findall(r"[\w.-]+", s)
        if len(out_tmp) > 9:
            for i in range(0, 2):
                out.append(out_tmp[i])
            out.append(out_tmp[2] + "+" + out_tmp[3])
            for i in range(3, len(out_tmp) - 1):
                out.append(out_tmp[i + 1])
        else:
            out = out_tmp
        tabs = re.findall(r"[\t]+", s)
        if (out[4] == out[8]):
            missing_value_handling = (" or ((x['" + out[1] + "'])=='\\N') ")
        else:
            missing_value_handling = ""

        if len(tabs) > 0:
            return (re.findall(r"[\t]+", s)[0].replace('\t', '	') +
                    '		if state == ' + out[0] + ':\n' +
                    re.findall(r"[\t]+", s)[0].replace('\t', '	') +
                    '			state = (' + out[4] +
                    ' if (' + "((x['" + out[1] + "'])!='\\N') and x['" + out[1] + "']<" + out[
                        2] + ')' + missing_value_handling +
                    ' else ' + out[6] + ')\n')

        else:
            return ('		if state == ' + out[0] + ':\n' +
                    '			state = (' + out[4] +
                    ' if (' + "((x['" + out[1] + "'])!='\\N') and x['" + out[1] + "']<" + out[
                        2] + ')' + missing_value_handling +
                    ' else ' + out[6] + ')\n')
    else:
        out = re.findall(r"[\w.-]+", s)
        return (re.findall(r"[\t]+", s)[0].replace('\t', '	') +
                '		if state == ' + out[0] + ':\n	' +
                re.findall(r"[\t]+", s)[0].replace('\t', '	') +
                '		return ' + out[2] + '\n')


def tree_parser(tree, i):
    if i == 0:
        return ('	if num_booster == 0:\n		state = 0\n'
                + "".join([string_parser(tree.split('\n')[i]) for i in range(len(tree.split('\n')) - 1)]))
    else:
        return ('	elif num_booster == ' + str(i) + ':\n		state = 0\n'
                + "".join([string_parser(tree.split('\n')[i]) for i in range(len(tree.split('\n')) - 1)]))


def model_features(trees,key):
    trees_txt = ''.join(trees)
    pattern = re.compile(r'\[\w*<')
    features_tmp = [x.replace('[', '').replace('<', '') for x in pattern.findall(trees_txt)]
    col_name = [key] + features_tmp
    new_col_name = list(set(col_name))
    new_col_name.sort(key=col_name.index)
    return new_col_name

def xgbTranformPy(model,out_file,key,featureCols):
    trees = model.get_booster().get_dump()
    col_name = [key]+featureCols
    result = ["#!/usr/bin/python \n"
              + "#coding:UTF-8 \n\n"
              + "import math\n"
              + "import sys \n"
              + "def xgb_tree(x, num_booster):\n"]
    raw_base_score = 0

    for i in range(len(trees)):
        result.append(tree_parser(trees[i], i))

    with open(out_file, 'w') as the_file:
        the_file.write("".join(result) + "\ndef xgb_predict(x):\n	predict = " + str(raw_base_score) + "\n"
                       + "# initialize prediction with base score\n"
                       + "	for i in range("
                       + str(len(trees))
                       + "):\n		predict = predict + xgb_tree(x, i)"
                       + "\n	score = 1/(1+math.exp(-predict))"
                       + "\n	return float(score)"
                       # main def
                       + "\nfor line in sys.stdin:"
                       + "\n	line = line.strip()"
                       + "\n	line = line.split('\t')"
                       + "\n	features = " + str(col_name)
                       + "\n	if len(line)!=len(features):"
                       + "\n		continue"
                       + "\n	feature = []"
                       + "\n	feature_dir={}"
                       + "\n	for i in range(0,len(features)):"
                       + "\n		if i == 0:"
                       + "\n			feature_dir[features[i]] = line[i]"
                       + "\n		elif line[i]=='\\N':"
                       + "\n			feature_dir[features[i]] = '\\N'"
                       + "\n		else:"
                       + "\n			feature_dir[features[i]] = float(line[i])\n"
                       + "\n	score = xgb_predict(feature_dir)"
                       + "\n	print '\t'.join([feature_dir['{key}'],str(score)])".format(key=key)
                       )


if __name__ == '__main__':
    xgbTranformPy(model, out_file)
