import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
# import talib
import random


# Reading the dataset
dataset = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\US States Data.csv')
dataset = dataset.dropna(axis=1, how='all')

data = dataset.drop(['States','Predicted cases (tests & population)'], axis=1)


print(data.columns)

#Dimensions of Data
n = data.shape[0]
p = data.shape[1]

print(data.shape)

X = data.drop('Actual cases (measured)', axis=1)
y = data['Actual cases (measured)']

X = X.values
y = y.values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Test train splitting

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_features=X_train.shape[1]


def preprocess(x, y):
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.int64)

  return x, y

def create_dataset(xs, ys, n_classes=10):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)


train_dataset = create_dataset(X_train, y_train)
val_dataset = create_dataset(X_test, y_test)


model = keras.Sequential([
    keras.layers.Reshape(target_shape=(1), input_shape=(9,)),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=192, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(),
    validation_steps=2
)