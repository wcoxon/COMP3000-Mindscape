import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models, preprocessing
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pydicom as dicom

class_names = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

directory = "OASIS archive/Data/"

def filePathToImage(path):
    imageFile = tf.io.read_file(path)
    # read the image just 1 (r) value per pixel since they are greyscale, r g and b are the same value
    
    return tf.io.decode_jpeg(imageFile,1)

def generate3dDataset():
    
    for filePath in tf.data.Dataset.list_files(directory+"*/*100.jpg"):
        
        pathString = filePath.numpy().decode("ascii")

        [subDirectoryName, fileName] = pathString.split("\\")[-2:]

        classification = class_names.index(subDirectoryName)

        patient_ID = fileName.split("_")[1]
        mpr_number = fileName.split("_")[3]

        scanPrefix = "%s%s/OAS1_%s_MR1_%s_" % (directory, subDirectoryName, patient_ID, mpr_number)

        imagePaths = ["%s%i.jpg" %(scanPrefix, layer) for layer in range(100,161)]

        imageArray = [filePathToImage(path) for path in imagePaths]
        
        yield (
            imageArray,
            classification
        )

dataset3d = tf.data.Dataset.from_generator(
    generate3dDataset,
    output_types=(
        tf.uint8,
        tf.uint8
    ),
    output_shapes=(
        (61,248,496,1),
        ()
    )
)


def displayFigure():
    figureCount=1

    plt.figure(figsize=(10,4))

    dataIterator = dataset3d.take(figureCount).as_numpy_iterator()

    figBrains=[]
    figImages=[]
    figTitles=[]

    for i, (_slices, _classification) in enumerate(dataIterator):
        plt.subplot(1,figureCount,1+i)
        plt.axis("off")

        #add volume to array so i can read it later to show a different layer
        figBrains.append(_slices)

        #draw the image as greyscale, only use r value for pixels
        #add image graphic to array to update it later
        figImages.append(plt.imshow(_slices[0],"Greys_r"))
    
        plt.title(class_names[_classification])

    #create slider
    layer_slider = Slider(
        ax=plt.axes([0.1,0.1,0.8,0.05]), 
        label="layer",
        valmin=0,
        valmax=60,
        valinit=0,
        valstep=1
    )

    def updateImageLayers(layer):
        for i in range(figureCount):
            figImages[i].set_data(figBrains[i][layer])

    layer_slider.on_changed(updateImageLayers)

    plt.show()

#displayFigure()

checkpoint_path = "training checkpoints/checkpoint2.ckpt"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

batchSize=2

depth = 61
width = 248
height = 496
channels = 1

def buildVGG():
    return models.Sequential([
        layers.Conv3D(filters=16, kernel_size=3, activation='relu', input_shape=(depth, height, width, channels)),
        layers.Conv3D(filters=16, kernel_size=3, activation='relu', input_shape=(depth, height, width, channels)),

        
        layers.MaxPooling3D(pool_size=2),

        layers.Conv3D(filters=32, kernel_size=3, activation='relu'),
        layers.Conv3D(filters=32, kernel_size=3, activation='relu'),

        layers.MaxPooling3D(pool_size=2),

        layers.Conv3D(filters=32, kernel_size=3, activation='relu'),
        layers.Conv3D(filters=32, kernel_size=3, activation='relu'),
        layers.Conv3D(filters=32, kernel_size=3, activation='relu'),

        layers.MaxPooling3D(pool_size=2),
        
        layers.Flatten(),

        layers.Dense(4, activation='softmax')
    ])


#make model
model = buildVGG()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
model.summary()


train_dataset = dataset3d


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

losses = []

class myCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        #update graph and display

        #print(logs.keys())
        
        ## loss or sparse_categorical_accuracy
        losses.append(logs["sparse_categorical_accuracy"])
        ax.plot(range(len(losses)),losses)
        fig.canvas.draw()
        fig.canvas.flush_events()

model.fit(
    train_dataset.batch(batchSize),
    epochs=2,
    batch_size=batchSize,
    callbacks=[checkpoint_callback,myCallback()],
    shuffle=True
)