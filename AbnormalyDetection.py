import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt


def showDataset (dataset):
    print(dataset.head())
    fig, ax = plt.subplots()
    dataset.plot(legend=False, ax=ax)
 #   plt.show()

def loadDataset (filename):
    master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    return pd.read_csv(
    master_url_root+filename, parse_dates=True, index_col="timestamp"
    )


def normalizeDataset(dataset):
    training_mean = dataset.mean()
    training_std = dataset.std()
    return (dataset - training_mean) / training_std

# 1 datapoint every 5 minutes => 288 datapoints a day
def create_sequences(values, time_steps=288):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


df_timeserie_no_anomaly = loadDataset("artificialNoAnomaly/art_daily_small_noise.csv")
df_timeserie_with_anomaly = loadDataset("artificialWithAnomaly/art_daily_jumpsup.csv")
df_training_normalized=normalizeDataset(df_timeserie_no_anomaly)

x_train = create_sequences(df_training_normalized.values)

showDataset(df_timeserie_no_anomaly)
showDataset(df_timeserie_with_anomaly)
showDataset(df_training_normalized)
# print(x_train)
print("Number of training samples:", len(df_training_normalized))

print("Training input shape: ", x_train.shape)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

