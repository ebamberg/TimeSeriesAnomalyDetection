import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

TIME_STEPS=288

def showDataset (dataset,title=""):
    print(dataset.head())
    fig, ax = plt.subplots()
    dataset.plot(legend=False, ax=ax, label=title)
    plt.show()

def loadDataset (filename):
    master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    return pd.read_csv(
    master_url_root+filename, parse_dates=True, index_col="timestamp"
    )


def normalizeDataset(dataset, baseDataset):
    training_mean = baseDataset.mean()
    training_std = baseDataset.std()
    return (dataset - training_mean) / training_std

# 1 datapoint every 5 minutes => 288 datapoints a day
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def plotTrainingResults(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

def findThreshold(trained_dataset,x_train_pred):
    # calculate the mae loss of our model -> what is the max error our model has while reconstructing our training data
    train_mae_loss = np.mean(np.abs(x_train_pred - trained_dataset), axis=1)
    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()
    # Get reconstruction loss threshold - everything heighter then the error that our model do during reconstruction have to be a Anomaly
    threshold = np.max(train_mae_loss)
    print("error threshold after sequence reconstruction: ", threshold)
    return threshold

def compareOriginalAndReconstructedSequences(dataset,dataset_reconstructed):
    plt.plot(x_train[0])
    plt.plot(x_train_pred[0])
    plt.show()

# --------------------------------------------------
#
#           Train our model with normal sequence data
#
# --------------------------------------------------

df_timeserie_no_anomaly = loadDataset("artificialNoAnomaly/art_daily_small_noise.csv")
df_training_normalized=normalizeDataset(df_timeserie_no_anomaly,df_timeserie_no_anomaly)

x_train = create_sequences(df_training_normalized.values)

showDataset(df_timeserie_no_anomaly,title="Dataset with no anomaly")
# showDataset(df_training_normalized)
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

#source and target is the same, cause we use a reconstruction model
history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

plotTrainingResults(history)
x_train_pred = model.predict(x_train)
threshold =findThreshold(x_train,x_train_pred)

compareOriginalAndReconstructedSequences(x_train,x_train_pred)



# --------------------------------------------------
#
#           find Anomalies
#
# --------------------------------------------------

df_timeserie_with_anomaly = loadDataset("artificialWithAnomaly/art_daily_jumpsup.csv")

df_withAnomaly_normalized=normalizeDataset(df_timeserie_with_anomaly,df_timeserie_no_anomaly)
x_withAnomaly = create_sequences(df_withAnomaly_normalized.values)

showDataset(df_timeserie_with_anomaly,title="Dataset with anomaly")

# Get test MAE loss.
x_withAnomaly_pred = model.predict(x_withAnomaly)
withAnomaly_mae_loss = np.mean(np.abs(x_withAnomaly_pred - x_withAnomaly), axis=1)
withAnomaly_mae_loss = withAnomaly_mae_loss.reshape((-1))

plt.hist(withAnomaly_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = withAnomaly_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))


anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_withAnomaly_normalized) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

df_subset = df_timeserie_with_anomaly.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_timeserie_with_anomaly.plot(legend=False, ax=ax,label="detected anomalies overlay to dataset ")
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()

