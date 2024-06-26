

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate, Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, Activation, Add
from tensorflow.keras.models import Model, Sequential

import env

#the vgg-16 base, without the top
def VGG_16_3D(inputs):
    t = 16
    pool = 2
    
    return Sequential([
        inputs,
        Conv3D(filters=t, kernel_size=3, activation='relu', padding="same"),
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
        #Dense(units=num_classes, activation='softmax')
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
    decoder = Concatenate()([decoder, concat_tensor])
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = conv_block(decoder, num_filters)
    return decoder

def unet_3d(inputs):

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

    x = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(decoder0)

    outputs = Flatten()(x)
    #outputs = Dense(256, activation='relu')(outputs)
    #outputs = Dense(128, activation='relu')(outputs)

    return Model(inputs=[inputs], outputs=[outputs])


#resnet
def conv3d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu'):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)

    x = Activation(activation)(x)
    return x

def residual_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    shortcut = x

    x = conv3d_bn(x, filters, kernel_size, strides, activation=activation)
    x = conv3d_bn(x, filters, kernel_size, activation=None)

    shortcut = Conv3D(filters, (1, 1, 1), strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x

#import tensorflow_models.vision as vision

def resnet_3d(inputs):

    x = conv3d_bn(inputs, filters=64, kernel_size=7, strides=2)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    return Model(inputs=inputs, outputs=x)


#full model
def buildModel(dm, architecture):

    csv_inputs = [Input(shape=feature["input_shape"]) for feature in dm.manifest["features"]]

    numerical_data = Concatenate()(csv_inputs)
    numerical_branch = Dense(64, activation='relu')(numerical_data)
    numerical_branch = Dense(32, activation='relu')(numerical_data)
    

    image_input = Input(shape=dm.image_shape)

    if(architecture=='UNet'):
        image_output =  unet_3d(image_input)
    elif(architecture=='VGG-16'):
        image_output =  VGG_16_3D(image_input)
    elif(architecture=='ResNet'):
        image_output =  resnet_3d(image_input)


    feature_inputs = [
        image_input, 
        *csv_inputs
    ]

    combined = Concatenate()([numerical_branch, image_output.output])

    concat_dense = Dense(64, activation='relu')(combined)
    concat_dense = Dense(32, activation='relu')(concat_dense)
    output = Dense(units=len(dm.classes))(concat_dense)

    return Model(inputs=feature_inputs, outputs=output)