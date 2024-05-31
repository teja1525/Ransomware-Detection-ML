import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Importing the dataset
dataset = pd.read_csv("Dataset/hpc_io_data.csv")
dataset.fillna(0, inplace=True)  # Replace missing values

# Displaying dataset values
print(dataset)

# Splitting dataset into features (X) and target (Y)
X = dataset[['instructions', 'LLC-stores', 'L1-icache-load-misses',
             'branch-load-misses', 'node-load-misses', 'rd_req', 'rd_bytes',
             'wr_req', 'wr_bytes', 'flush_operations', 'rd_total_times',
             'wr_total_times', 'flush_total_times']]
Y = dataset['label']

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Normalizing features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshaping features for CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

# Encoding target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Defining CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(64, (1, 1), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(1, 1)))
cnn_model.add(Conv2D(32, (1, 1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(units=256, activation='relu'))
cnn_model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Compiling CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training CNN model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Evaluating CNN model
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print("CNN2D Model Accuracy:", accuracy)

# Making predictions
predictions = cnn_model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Calculating metrics
precision = precision_score(np.argmax(y_test, axis=1), predicted_labels, average='macro')
recall = recall_score(np.argmax(y_test, axis=1), predicted_labels, average='macro')
f1 = f1_score(np.argmax(y_test, axis=1), predicted_labels, average='macro')
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_labels)
print("CNN2D Model Precision:", precision)
print("CNN2D Model Recall:", recall)
print("CNN2D Model F1 Score:", f1)
print("CNN2D Model Confusion Matrix:\n", conf_matrix)
