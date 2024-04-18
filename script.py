
import tensorflow as tf
from tensorflow.keras import losses, metrics

import preprocessing
from preprocessing import class_distribution, generator_dataset, write_tfrecord, _parse_function
from interface import DataBrowser, selectArchitecture, PerformanceProfiler
from architectures import buildModel
import env

dataset = generator_dataset()
#write_tfrecord(dataset,'data/ADNI1_dataset_2000.tfrecord')
#input("finished")

#dataset = tf.data.TFRecordDataset('data/ADNI1_dataset_2000.tfrecord').map(_parse_function)

class_dist = class_distribution(dataset)
class_weights = {i:1/d for i, d in enumerate(class_dist)}

DataBrowser(dataset,4,class_dist)
selectArchitecture()

model = buildModel()

boundaries = [500/env.batchSize, 1000/env.batchSize, 2000/env.batchSize]
values = [0.01, 0.005, 0.001, 0.0001]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [metrics.SparseCategoricalAccuracy()]
)
model.summary()


env.debug = False




model.fit(
    x=dataset.batch(env.batchSize),
    epochs=env.epochs,
    batch_size=env.batchSize,
    callbacks=[PerformanceProfiler()],
    shuffle=True,
    class_weight=class_weights
)

import matplotlib.pyplot as plt
plt.show(block=True)