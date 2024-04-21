
import tensorflow as tf
from tensorflow.keras import losses, metrics
import preprocessing
from interface import DataBrowser,selectDataset, selectArchitecture, PerformanceProfiler
from architectures import buildModel
import env

dataset_manifest = selectDataset()

dm = preprocessing.dataset_manager(dataset_manifest)

DataBrowser(dm,4)
architecture = selectArchitecture()

model = buildModel(dm,architecture)

epoch_samples = 1024

boundaries = [0.25*epoch_samples/env.batchSize, 0.5*epoch_samples/env.batchSize, 0.75*epoch_samples/env.batchSize]
values = [1e-4, 5e-5, 1e-5, 1e-6]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [metrics.SparseCategoricalAccuracy()]
)
model.summary()

training_data = dm.dataset.take(epoch_samples)#.cache("training_cache2")

stats = [
    "dataset: %s" % dm.manifest["name"],
    "architecture: %s" % architecture,
    "epochs: %s" % env.epochs,
    "batch size: %s" % env.batchSize
]

model.fit(
    x=training_data.batch(env.batchSize),
    steps_per_epoch=epoch_samples,
    epochs=env.epochs,
    batch_size=env.batchSize,
    callbacks=[PerformanceProfiler(stats)],
    shuffle=True,
    class_weight=dm.class_weight
)

import matplotlib.pyplot as plt
plt.show(block=True) # to prevent it from closing when training is done