import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import scipy.stats as sp
import sys
import random
from collections import *

argToName = {'dia': 'BP Dia_mmHg', 'sys': 'LA Systolic BP_mmHg',
             'eda': 'EDA_microsiemens', 'res': 'Respiration Rate_BPM'}


def formatdata(df, arg):
    data = df.copy(deep=True)
    columns = ['ID', 'Type', 'Class', 'Data']
    data[columns] = data.Data.str.split(',', n=3, expand=True)
    data.Data = data.Data.str.split(',')
    if arg != 'all':
        fullName = argToName[arg]
        data = data[data['Type'] == fullName]
    return data


def normalizeSignals(dataFrame):
    for index, row in dataFrame.iterrows():
        row["Data"] = normalize([row["Data"]])[0]
    return dataFrame


def getNormalizedDataFrame(filteredDf, downSampleLength):
    for index, row in filteredDf.iterrows():
        data = row["Data"]
        newData = []
        ratio = int(len(data)/downSampleLength)
        for index in range(0, len(data), ratio):
            miniData = [float(i) for i in data[index:index+ratio]]
            newData.append(sum(miniData)/ratio)
        if(len(newData) > downSampleLength):
            newData = newData[0:downSampleLength]
        elif(len(newData) < downSampleLength):
            while(len(newData) != downSampleLength):
                newData.append(newData[-1])
        row["Data"] = newData

    normalizeSignals(filteredDf)
    # print(filteredDf)


def convertDftoDataArray(filiteredDf):
    x_train = []

    for index, row in filteredDfdia.iterrows():
        thisrow = []
        for i in row["Data"]:
            thisrow.append(i)
        x_train.append(thisrow)

    return np.array(x_train)


def getPrediction(train, test):
    x_train = convertDftoDataArray(train)
    x_test = convertDftoDataArray(test)
    # get class
    y_train = train.iloc[:, 3].values
    y_test = test.iloc[:, 3].values
    #print(x_train.shape, x_train, y_train, type(x_train), "len", len(x_train[0]))
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("recall: ", recall_score(y_pred=y_pred,
        y_true=y_test, pos_label='Pain'))
    print("precision: ", precision_score(y_pred=y_pred,
        y_true=y_test, pos_label='Pain'))
    return y_pred

pd.set_option("display.max_rows", None, "display.max_columns", None)
# .sample(frac=1).reset_index(drop=True)
df = pd.read_csv(sys.argv[1],  delimiter='|', names=['Data'])
# .sample(frac=1).reset_index(drop=True)
df_test = pd.read_csv(sys.argv[2], delimiter='|', names=['Data'])
np.random.shuffle(df_test.values.reshape(-1, 4, df.shape[1]))
np.random.shuffle(df.values.reshape(-1, 4, df.shape[1]))

#df_origtest = pd.read_csv('data2_orig.csv', delimiter='|', names=['Data'])
#filterOrig = formatdata(df_origtest, "all")
# print("filterorig0",filterOrig[0])
downSampleLength = 5000
filteredDf = formatdata(df_test, "all")
filteredDfdia = formatdata(df, "dia")
filteredDfdiaTest = formatdata(df_test, "dia")
filteredDfsys = formatdata(df, "sys")
filteredDfsysTest = formatdata(df_test, "sys")
filteredDfeda = formatdata(df, "eda")
filteredDfedaTest = formatdata(df_test, "eda")
filteredDfres = formatdata(df, "res")
filteredDfresTest = formatdata(df_test, "res")
getNormalizedDataFrame(filteredDfdia, downSampleLength)
getNormalizedDataFrame(filteredDfdiaTest, downSampleLength)
getNormalizedDataFrame(filteredDfsys, downSampleLength)
getNormalizedDataFrame(filteredDfsysTest, downSampleLength)
getNormalizedDataFrame(filteredDfeda, downSampleLength)
getNormalizedDataFrame(filteredDfedaTest, downSampleLength)
getNormalizedDataFrame(filteredDfres, downSampleLength)
getNormalizedDataFrame(filteredDfresTest, downSampleLength)

# print(filteredDf)
# dia
# get data
#x_train = filteredDfdia.iloc[:,0].to_numpy()
print("DiaPredictions")
DiaPredictions = getPrediction(filteredDfdia, filteredDfdiaTest)
print("SysPredictions")
SysPredictions = getPrediction(filteredDfsys, filteredDfsysTest)
print("EdaPredictions")
EdaPredictions = getPrediction(filteredDfeda, filteredDfedaTest)
print("ResPredictions")
ResPredictions = getPrediction(filteredDfres, filteredDfresTest)

# print(filteredDf)
combinedPredictions = []

for dia, sys, eda, res in zip(DiaPredictions, SysPredictions, EdaPredictions, ResPredictions):
    count = [dia, sys, eda, res]
    count = Counter(count)
    if count["No Pain"] > count["Pain"]:
        combinedPredictions.append("No Pain")
    elif count["No Pain"] < count["Pain"]:
        combinedPredictions.append("Pain")
    else:
        combinedPredictions.append((random.choice(["No Pain", "Pain"])))

# correct predictions

correctPredictions = []

for index, row in filteredDf.iterrows():
    if index % 4 == 0:
        correctPredictions.append(row["Class"])


print("accuracy: ", accuracy_score(correctPredictions, combinedPredictions))
print("recall: ", recall_score(y_pred=combinedPredictions,
      y_true=correctPredictions, pos_label='Pain'))
print("precision: ", precision_score(y_pred=combinedPredictions,
      y_true=correctPredictions, pos_label='Pain'))
