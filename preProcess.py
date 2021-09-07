import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy.stats as sp
import sys



argToName = {'dia':'BP Dia_mmHg', 'sys':'LA Systolic BP_mmHg', 'eda':'EDA_microsiemens', 'res': 'Respiration Rate_BPM'}
def formatdata(data, arg):
  columns = ['ID', 'Type', 'Class', 'Data']
  data[columns] = data.Data.str.split(',', n=3, expand=True)
  data.Data = data.Data.str.split(',')
  if arg != 'all':
    fullName = argToName[arg]
    data = data[data['Type'] == fullName]
  return data

df = pd.read_csv('data1.csv',  delimiter='|', names = ['Data'])
filteredDf = formatdata(df, "sys")
features = []

#mean, variance, entropy, min, and max
for index, row in filteredDf.iterrows():
  row.Data = [float(i) for i in row.Data]

  features.append([sum(row.Data)/len(row.Data), 
                  np.var(row.Data), 
                  sp.entropy(pd.Series(row.Data).value_counts()), 
                  min(row.Data), 
                  max(row.Data),
                  row.Class
                  ])


table = pd.DataFrame(features, index=list(range(len(features))), columns=['mean', 'variance', 'entropy', 'min', 'max', 'class'])



#get data
x = table.iloc[:,0:5].values
#get class
y = table.iloc[:, 5].values

########################################
kf = KFold(n_splits=10)
rf = RandomForestClassifier()


confMatrices = np.ndarray(shape=(2,2))
accuracy = 0
recallScore = 0
precisionScore = 0

for train_index, test_index in kf.split(x):
  x_train, x_test = x[train_index], x[test_index]
  y_train, y_test = y[train_index], y[test_index]  
  rf.fit(x_train, y_train)
  print("xtrain", x_train, y_train)
  y_pred = rf.predict(x_test)
  confMatrices += confusion_matrix(y_test, y_pred)
  accuracy+=accuracy_score(y_test, y_pred)
  recallScore += recall_score(y_pred=y_pred, y_true=y_test, pos_label='Pain')
  precisionScore+=precision_score(y_pred=y_pred, y_true=y_test, pos_label='Pain')


#########################################
print(f"Conf. Matrix: \n {confMatrices/10}")
print(f"Accuracy Score: {accuracy/10}")
print(f"Recall Score: {recallScore/10}")
print(f"Precision Score: {precisionScore/10}")


