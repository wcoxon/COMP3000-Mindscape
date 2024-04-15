
import tensorflow as tf
from tensorflow.keras import losses, metrics

import preprocessing
from preprocessing import dataset
from interface import DataBrowser, selectArchitecture
from architectures import buildModel
from env import batchSize

preprocessing.debug = True

test = DataBrowser(dataset,4)


preprocessing.debug = False
architecture = selectArchitecture()


model = buildModel(architecture)

model.compile(
    optimizer = 'adam',
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [metrics.SparseCategoricalAccuracy()]
)
model.summary()

train_dataset = dataset # split into train and validation sets


from interface import MetricsFigure

profiler = MetricsFigure()

loss = []
accuracy = []
class recordMetrics(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):

        accuracy.append(logs["sparse_categorical_accuracy"])
        profiler.plotAccuracy(accuracy)

        loss.append(logs["loss"])
        profiler.plotLoss(loss)

        profiler.updateCanvas()


model.fit(
    x=train_dataset.batch(batchSize),
    epochs=2,
    batch_size=batchSize,
    callbacks=[recordMetrics()],
    shuffle=True
)

import matplotlib.pyplot as plt
plt.show(block=True)