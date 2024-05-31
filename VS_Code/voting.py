import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier

# Importing the dataset
dataset = pd.read_csv("Dataset/hpc_io_data.csv")
dataset.fillna(0, inplace=True)  # Replace missing values

# Displaying dataset values
print(dataset)

# Plotting graph of ransomware and benign from dataset where 0 label refers to benign and 1 refers to ransomware
labels, count = np.unique(dataset['label'], return_counts=True)
labels = ['Benign', 'Ransomware']
height = count
bars = labels
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Dataset Class Label Graph")
plt.ylabel("Count")
plt.show()

# Splitting dataset into features (X) and target (Y)
X = dataset[['instructions', 'LLC-stores', 'L1-icache-load-misses',
             'branch-load-misses', 'node-load-misses', 'rd_req', 'rd_bytes',
             'wr_req', 'wr_bytes', 'flush_operations', 'rd_total_times',
             'wr_total_times', 'flush_total_times']]
Y = dataset['label']

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Defining classifiers
clf1 = AdaBoostClassifier(n_estimators=10, random_state=0)
clf2 = RandomForestClassifier(n_estimators=5, random_state=1)

# Defining Voting Classifier
eclf = VotingClassifier(estimators=[('ad', clf1), ('rf', clf2)], voting='soft')

# Training Voting Classifier
eclf.fit(X_train, y_train)

# Making predictions
predict = eclf.predict(X_test)

# Calculating metrics
p = precision_score(y_test, predict, average='macro') * 100
r = recall_score(y_test, predict, average='macro') * 100
f = f1_score(y_test, predict, average='macro') * 100
a = accuracy_score(y_test, predict) * 100

print("Voting Classifier Accuracy  : " + str(a))
print("Voting Classifier Precision : " + str(p))
print("Voting Classifier Recall    : " + str(r))
print("Voting Classifier F1 Score  : " + str(f))

# Plotting confusion matrix
conf_matrix = confusion_matrix(y_test, predict)
plt.figure(figsize=(5, 5))
ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
ax.set_ylim([0, len(labels)])
plt.title("Voting Classifier Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
#plt.show()