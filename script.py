import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import pydicom as dicom
import csv

import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.layers import Input,  concatenate,Cropping3D, Flatten, Dense, Concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, BatchNormalization, GlobalAveragePooling3D, Activation, Add
from tensorflow.keras.models import Model, Sequential

from scipy.ndimage import zoom

# ADNI/[collection name]/ADNI/[subjectID]/[series description]/[scan date]/[image id]/[images]

ADNI3_set = ["ADNI3 T1 AXIAL 54 DEPTH","ADNI3_T1_AXIAL_54_DEPTH_4_07_2024"]


setpaths = ADNI3_set

directory = "data/%s/ADNI"%setpaths[0]
imageCSVPath ="data/%s/%s.csv"%(setpaths[0], setpaths[1])

sex_map = {
    'M':0, # male
    'F':1 # female
}

label_map = {
    'CN':0, # cognitively normal
    'SMC':1, # subjective memory complaints
    'EMCI':2, # early mild cognitive impairment
    'MCI':3, # mild cognitive impairment
    'LMCI':4, # late mild cognitive impairment
    'AD':5, # alzheimers disease
}

num_classes = len(set(label_map.values()))

image_shape = (
    54, # depth
    78, # width
    78, # height
    1    # channels
)

batchSize = 1

def getMeta(imageID):
    with open(imageCSVPath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Image Data ID'] == imageID:
                return {
                    "group":label_map[row['Group']],
                    "age":int(row["Age"]),
                    "sex":sex_map[row["Sex"]]
                }


def generate3dDataset():
    for subjectDir in tf.data.Dataset.list_files("%s/*/*/*/*" % directory): #for each scan
        
        
        scanDir = subjectDir.numpy().decode("ascii") # image directory path

        imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.dcm"%scanDir,shuffle=False) ] # collect all the dcm file paths
        
        imagePaths.sort(key=lambda f: int(f.split('_')[-3]))
        
        imageArray = np.array([ dicom.dcmread(path).pixel_array for path in imagePaths ]) # load up the image array
        
        # set shape
        imageArray = np.reshape(imageArray,imageArray.shape[-3:])

        # add channels
        imageArray = np.expand_dims(imageArray, axis=-1)

        # resize images
        if(imageArray.shape != image_shape):
            imageArray = np.array([ tf.image.resize(i, image_shape[1:3]) for i in imageArray])
        
        # normalize values
        imageArray = imageArray/imageArray.max()

        imageID = scanDir.split("\\")[-1] # extract image ID


        metadata = getMeta(imageID) # query CSV data
        age = metadata["age"]
        sex = metadata["sex"]

        # some scans dont have weight
        weight = dicom.dcmread(imagePaths[0]).get("PatientWeight")

        diagnosis = metadata["group"]

        yield (
            (
                imageArray, 
                age, 
                sex,
                weight
            ),
            diagnosis
        )


dataset3d = tf.data.Dataset.from_generator(
    generate3dDataset,
    output_signature=(
        (
            tf.TensorSpec(shape=image_shape, dtype=tf.float32), 
            tf.TensorSpec(shape=(), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.float16)
        ),
        tf.TensorSpec(shape=(), dtype=tf.uint8)
    )
)

def displayFigure():
    figureCount=1

    plt.figure(figsize=(10,4))

    dataIterator = dataset3d.take(figureCount).as_numpy_iterator()

    imageArrays = []
    imshows = []

    for i, (_inputs, _output) in enumerate(dataIterator):
        plt.subplot(1,figureCount,1+i)
        plt.axis("off")

        imageInput = _inputs[0]
        imageArrays.append(imageInput)
        imshows.append(plt.imshow(imageInput[0],cmap='gray'))

        plt.title(_output)

    #create slider
    layer_slider = Slider(
        ax=plt.axes([0.1,0.1,0.8,0.05]), 
        label="layer",
        valmin=0, valmax=image_shape[0]-1, valinit=0, valstep=1
    )

    def updateImageLayers(layer):
        for i in range(figureCount):
            imshows[i].set_data(imageArrays[i][layer])
            

    layer_slider.on_changed(updateImageLayers)

    plt.show()

displayFigure()


#the vgg-16 base, without the top
def VGG_16_3D():
    t = 64 #t for test lol just a scale of model depth, but hardcoding the proportional sizes if that makes sense like this layer will be the same relative scale from others
    pool = (1,2,2)
    
    return Sequential([

        Conv3D(filters=t, kernel_size=3, activation='relu', padding="same", input_shape=image_shape),
        Conv3D(filters=t, kernel_size=3, activation='relu', padding="same"),
        MaxPooling3D(pool_size=pool),

        Conv3D(filters=2*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=2*t, kernel_size=3, activation='relu', padding="same"),
        MaxPooling3D(pool_size=pool),

        Conv3D(filters=4*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=4*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=4*t, kernel_size=3, activation='relu', padding="same"),
        MaxPooling3D(pool_size=pool),

        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        MaxPooling3D(pool_size=pool),

        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        Conv3D(filters=8*t, kernel_size=3, activation='relu', padding="same"),
        MaxPooling3D(pool_size=pool),

        Flatten(),

        Dense(units=16*t, activation='relu'),
        Dense(units=16*t, activation='relu'),
        Dense(units=16*t, activation='relu'),
        
    ])
# unet
def conv_block(input_tensor, num_filters):
    encoder = Conv3D(num_filters, 3, padding='same')(input_tensor)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Conv3D(num_filters, 3, padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling3D(2, strides=2)(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_tensor)
    # Calculate the necessary padding
    target_shape = tf.shape(concat_tensor)
    decoder_shape = tf.shape(decoder)
    pads = target_shape[1:4] - decoder_shape[1:4]
    decoder = tf.pad(decoder, paddings=[[0, 0], [pads[0]//2, pads[0] - pads[0]//2], [pads[1]//2, pads[1] - pads[1]//2], [pads[2]//2, pads[2] - pads[2]//2], [0, 0]])
    decoder = layers.concatenate([decoder, concat_tensor], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = conv_block(decoder, num_filters)
    return decoder

def unet_3d(input_shape):
    inputs = layers.Input(shape=input_shape)

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)

    center = conv_block(encoder4_pool, 1024)

    decoder4 = decoder_block(center, encoder4, 512)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)

    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(decoder0)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


#resnet
def conv3d_bn(x, filters, kernel_size, strides=(1, 1, 1), padding='same', activation='relu'):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x
def residual_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu'):
    shortcut = x

    x = conv3d_bn(x, filters, kernel_size, strides, activation=activation)
    x = conv3d_bn(x, filters, kernel_size, activation=None)

    shortcut = Conv3D(filters, (1, 1, 1), strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x
def resnet_3d():
    inputs = Input(image_shape)

    x = conv3d_bn(inputs, 64, (7, 7, 7), strides=(2, 2, 2))
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, strides=(2, 2, 2))
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = GlobalAveragePooling3D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def buildModel():

    age_input = Input(shape=(1,))
    age_dense = Dense(256, activation='relu')(age_input)

    sex_input = Input(shape=(1,))
    sex_dense = Dense(256, activation='relu')(sex_input)

    weight_input = Input(shape=(1,))
    weight_dense = Dense(256, activation='relu')(weight_input)

    image_input = Input(shape=image_shape)
    image_output =  unet_3d(image_shape)(image_input)
    image_output = Flatten()(image_output)

    concatenated = Concatenate()([image_output, age_dense, sex_dense, weight_dense])
    concat_dense = Dense(units=512,activation='relu')(concatenated)
    concat_dense = Dense(units=512,activation='relu')(concat_dense)

    output = Dense(units=num_classes)(concat_dense)

    return Model(inputs=[image_input, age_input,sex_input,weight_input], outputs=output)


model = buildModel()

model.compile(
    optimizer = 'adam',
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [metrics.SparseCategoricalAccuracy()]
)

model.summary()

#split into train and validation sets
train_dataset = dataset3d
#train_dataset = dataset3d.take(1000)
#test_dataset = dataset3d.skip(1000)


loss = []
accuracy = []

plt.ion()
fig = plt.figure()

accuracyPlot = fig.add_subplot(211)
plt.grid()
lossPlot = fig.add_subplot(212)
plt.grid()


class myCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        #update graph and redraw
        loss.append(logs["loss"])
        accuracy.append(logs["sparse_categorical_accuracy"])

        accuracyPlot.plot(range(batch+1),accuracy)
        lossPlot.plot(range(batch+1),loss)

        fig.canvas.draw()
        fig.canvas.flush_events()

model.fit(
    x=train_dataset.batch(batchSize),
    epochs=1,
    batch_size=batchSize,
    callbacks=[myCallback()],
    shuffle=True
)

plt.show(block=True)