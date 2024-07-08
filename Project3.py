from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import minmax_scale
import scipy.signal as signal
from sklearn import metrics
import csv, sys, os
import numpy as np
import math

#Read in files for training and testing
DATASET1 = sys.argv[1]
DATASET2 = sys.argv[2]

X_train = []
Y_train = []
X_test = []
Y_test = []

def read_data(data):
    arr1 = []
    arr2 = []
    arr3 = []
    with open(data) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for row in readCSV:
            #if(row[1] == 'EDA_microsiemens'):
            if(row[2] == 'No Pain'):
                for i in range(3, len(row)):
                    num = float(row[i])
                    arr1.append(num)
                arr2.append(arr1)
                arr1 = []
                arr3.append(0)
            if(row[2] == 'Pain'):
                for i in range(3, len(row)):
                    num = float(row[i])
                    arr1.append(num)
                arr2.append(arr1)
                arr1 = []
                arr3.append(1)
    return arr2, arr3

def normalize_data(dataset):
    new_arr = []
    temp_arr = []
    for list in dataset:
        for j in list:
            temp_arr.append(j)
        ratio = math.floor((len(temp_arr) / 5000))
        X = signal.resample_poly(temp_arr, 1, ratio)
        X = minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
        X = np.resize(X, 5000)
        new_arr.append(X)
        temp_arr = []
    return new_arr

#Read data into lists and convert to array
X_train, Y_train = read_data(DATASET1)
X_test, Y_test = read_data(DATASET2)

#Downsample and normalize data
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

clf = RandomForestClassifier(n_estimators=100)

eclf = VotingClassifier(estimators=[('rf', clf)], voting='hard')

eclf = eclf.fit(X_train, Y_train)
#print(eclf.predict(X_test))

Y_pred = eclf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))

#Information used as experimental details in report
#print("Confusion Matrix:")
#print(confusion_matrix(Y_test, Y_pred))
#print("Accuracy:", accuracy_score(Y_test, Y_pred))
#print("Precision:", precision_score(Y_test, Y_pred))
#print("Recall:", recall_score(Y_test, Y_pred))

# =============================================================================
# The following section is simply for data interpretation, not actual code
# =============================================================================

#'BP Dia_mmHg'
# DS1 DS2
#[0 1 0 1 1 1 1 0 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 1
# 1 0 1 0 1 1 1 1 0 0 1 0 0 1 1 0 1 0 0 0 1 1 1]

# DS2 DS1
#[0 1 1 0 0 1 0 0 0 1 1 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 1 1 0 1 0 1 1
# 1 1 0 0 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 1 0 0]

# =============================================================================
# Confusion Matrix (DS1 DS2):
# [[20 10]
#  [ 6 24]]
# Accuracy: 0.7333333333333333
# Precision: 0.7058823529411765
# Recall: 0.8

# Confusion Matrix (DS2 DS1):
# [[20 10]
#  [ 8 22]]
# Accuracy: 0.7
# Precision: 0.6875
# Recall: 0.7333333333333333
# =============================================================================

#'EDA_microsiemens'
# DS1 DS2
#[0 1 0 0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 1 1 1 0 1 0 0
# 1 1 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0]

# DS2 DS1
#[1 1 1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 1
# 1 1 1 1 1 0 1 1 1 1 0 0 0 0 1 0 0 1 0 1 1 1 0]

# =============================================================================
# Confusion Matrix (DS1 DS2):
# [[23  7]
#  [14 16]]
# Accuracy: 0.65
# Precision: 0.6956521739130435
# Recall: 0.5333333333333333

# Confusion Matrix (DS2 DS1):
# [[12 18]
#  [11 19]]
# Accuracy: 0.5166666666666667
# Precision: 0.5135135135135135
# Recall: 0.6333333333333333
# =============================================================================

#'LA Systolic BP_mmHg'
# DS1 DS2
#[1 0 0 0 0 1 0 1 0 1 1 1 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0
# 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 1 0 0]

# DS2 DS1
#[0 1 1 1 0 1 0 1 0 0 1 0 0 1 0 1 1 1 1 1 0 1 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0
# 1 0 1 1 1 1 0 1 1 0 1 0 1 0 0 0 1 0 1 1 0 1 1]

# =============================================================================
# Confusion Matrix (DS1 DS2):
# [[24  6]
#  [14 16]]
# Accuracy: 0.6666666666666666
# Precision: 0.7272727272727273
# Recall: 0.5333333333333333

# Confusion Matrix (DS2 DS1):
# [[18 12]
#  [ 5 25]]
# Accuracy: 0.7166666666666667
# Precision: 0.6756756756756757
# Recall: 0.8333333333333334
# =============================================================================
    
#'Respiration Rate_BPM'
# DS1 DS2
#[0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 0 0 1 1 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 0 0
# 0 0 0 1 1 1 1 1 1 0 0 0 1 0 1 0 1 1 0 1 0 0 1]

# DS2 DS1
#[0 0 1 0 1 1 0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 1 0 0 1 1 0 1 0 0 1 0 1
# 1 0 1 0 1 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0 0]

# =============================================================================
# Confusion Matrix (DS1 DS2):
# [[23  7]
#  [13 17]]
# Accuracy: 0.6666666666666666
# Precision: 0.7083333333333334
# Recall: 0.5666666666666667

# Confusion Matrix (DS2 DS1):
# [[10 20]
#  [10 20]]
# Accuracy: 0.5
# Precision: 0.5
# Recall: 0.6666666666666666
# =============================================================================

# DS1 DS2 Majority Voting Results (S represents 2 pain and 2 no pain)
#[0 S 0 0 0 1 S 1 0 1 S 1 0 1 0 1 0 1 S 1 0 1 0 1 0 S 0 0 0 S 0 1 0 S S S 0
# 1 0 S 0 1 S 1 1 0 0 0 0 0 S S 0 1 S 0 0 1 0 S]

# DS2 DS1 Majority Voting Results (S represents 2 pain and 2 no pain)
#[0 1 1 S 0 1 0 S 0 S 1 S 0 S 1 1 1 1 S 1 0 1 1 0 S 1 0 S S S S S 0 1 S 0 1
# 1 S 1 S 1 0 1 S 1 S S 0 0 S 1 S S 0 1 S 1 S 0]
