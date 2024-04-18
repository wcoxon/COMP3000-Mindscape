import numpy as np

import pydicom as dicom
import nibabel as nib
import csv

import tensorflow as tf

from scipy.ndimage import zoom


import env
from env import images_directory, ADNI_merge, image_shape, label_map, dataset_props

import datetime
from dateutil.relativedelta import relativedelta
import time



def class_distribution(dataset):

    class_counts = np.array([0.0]*env.num_classes)

    links = dataset_props.get("link")
    if(links==None):
        for (_input, _output) in dataset:
            dx = _output.numpy()
            class_counts[dx]+=1
        return class_counts


    with open(links[0]["table"]["path"], mode='r') as file:
        for row in csv.DictReader(file):
            feature = links[0]["table"]["features"]["Group"]
            dx = feature["map"][row[feature["column"]]]
            class_counts[dx]+=1
    
    total_samples = sum(class_counts)
    class_counts /= float(total_samples)

    #class_weights = 1 / class_counts

    return class_counts #{i: v for i, v in enumerate(class_weights)}

def getMeta(constraint):

    metadata = constraint

    for link in dataset_props["link"]:
        with open(link["table"]["path"], mode='r') as file:
            reader = csv.DictReader(file)
            column_values = [[link["table"]["features"][feature]["column"],metadata[feature]] for feature in link["keys"]]

            #match_values = [ metadata[feature] for feature in link["keys"]]

            for row in reader:
                matches = [row[column]==value for column,value in column_values]
                if all(matches):
                    for feature in link["table"]["features"].values():
                        if metadata.get(feature["name"]):
                            continue

                        feature_value = row[feature["column"]]
                        if feature.get("map"):
                            feature_value = feature["map"].get(feature_value)

                        metadata[feature["name"]] = feature_value
    
    return metadata

def normalizePixels(pixelArray):
    return pixelArray / pixelArray.max()


def loadDicom(imageDir):

    imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.dcm"%imageDir,shuffle=False) ] # collect all the dcm file paths
    dcmArray = [ dicom.dcmread(path) for path in imagePaths ]

    dcmData = dcmArray[0]
    if env.debug:
        imagePosition = (0x0020,0x0032)
        imageOrientation = (0x0020,0x0037)
        sliceLocation = (0x0020, 0x1041)
        slicePosition = (0x0019, 0x1015)
        windowCenter = (0x0028, 0x1050)
        print(dcmData.get(imagePosition))
        print(dcmData.get(imageOrientation))
        print(dcmData.get(sliceLocation))
        print(dcmData.get(slicePosition))
        print(dcmData.get(windowCenter))
        #print()

    try:
        dcmArray.sort(key=lambda dcm: float(dcm.get((0x0020, 0x1041)).repval[1:-1]))
    except AttributeError:
        if(env.debug): print("dicom missing positions")

    imageArray = np.array([dcm.pixel_array for dcm in dcmArray])

    imageArray = np.reshape(imageArray,imageArray.shape[-3:])
    imageArray = np.expand_dims(imageArray, axis=-1)
    if(imageArray.shape != image_shape):
        imageArray = np.array([ tf.image.resize(i, image_shape[1:3]) for i in imageArray]) # resize images
    imageArray = normalizePixels(imageArray)

    return imageArray


def resize_volume_by_padding_and_cropping(volume, target_size):
    depth, height, width = volume.shape
    target_depth, target_height, target_width = target_size
    
    # Calculate padding amounts
    pad_depth = max((target_depth - depth) // 2, 0)
    pad_height = max((target_height - height) // 2, 0)
    pad_width = max((target_width - width) // 2, 0)
    
    # Apply padding
    volume_padded = np.pad(volume, ((pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width)), 'constant')
    
    # Calculate cropping indices
    crop_depth_start = (volume_padded.shape[0] - target_depth) // 2
    crop_height_start = (volume_padded.shape[1] - target_height) // 2
    crop_width_start = (volume_padded.shape[2] - target_width) // 2
    
    # Apply cropping
    volume_resized = volume_padded[
        crop_depth_start:crop_depth_start + target_depth,
        crop_height_start:crop_height_start + target_height,
        crop_width_start:crop_width_start + target_width
    ]

    print(volume.shape)
    print(target_size)
    print(pad_depth,pad_height,pad_width)
    print(crop_depth_start,crop_height_start,crop_width_start)
    print(volume_resized.shape)
    
    return np.array(volume_resized)

def loadNii(imageDir):
    imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.nii"%imageDir,shuffle=False) ]

    nii = nib.load(imagePaths[0])

    image = nii.get_fdata()
    #image = resize_volume_by_padding_and_cropping(image,image_shape[:3])
    image = zoom(image, (image_shape[0]/image.shape[0], image_shape[1]/image.shape[1], image_shape[2]/image.shape[2]))
    
    #image = normalizePixels(image)
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    volume_normalized = (image - p1) / (p99 - p1)
    image = np.clip(volume_normalized, 0, 1)


    image = np.expand_dims(image,axis=-1)

    return image

def generateDataset():
    for subjectDir in tf.data.Dataset.list_files("%s/*/*/*/*" % images_directory): #for each scan
        
        scanDir = subjectDir.numpy().decode("ascii") # image directory path
        imageID = scanDir.split("\\")[-1] # extract image ID from file path

        
        if(dataset_props["image_format"]=="nii"): volume = loadNii(scanDir)
        elif(dataset_props["image_format"]=="dcm"): volume = loadDicom(scanDir)


        metadata = getMeta({"Image ID" : imageID}) # query CSV data

        if env.debug:
            print(volume.shape)
            print(scanDir)
            print(metadata,"\n")

        yield (
            (
                volume, 
                metadata["Age"],
                metadata["Sex"],
                metadata["Race"],
                #metadata["weight"]
            ),
            metadata["Group"]
        )

def generator_dataset():
    return tf.data.Dataset.from_generator(
        generateDataset,
        output_signature=(
            (
                tf.TensorSpec(shape=image_shape, dtype=tf.float32), # image
                tf.TensorSpec(shape=(), dtype=tf.float16), # age
                tf.TensorSpec(shape=(2,), dtype=tf.uint8), # sex
                tf.TensorSpec(shape=(7,), dtype=tf.uint8), # race
                #tf.TensorSpec(shape=(), dtype=tf.float16), # weight
            ),
            tf.TensorSpec(shape=(), dtype=tf.uint8)
        )
)



# Define functions to convert data to tf.train.Feature
_bytes_feature = lambda value : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])) #Returns a bytes_list from a string / byte.
_float_feature = lambda value : tf.train.Feature(float_list=tf.train.FloatList(value=[value])) #Returns a float_list from a float / double.
_int64_feature = lambda value : tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) #Returns an int64_list from a bool / enum / int / uint.

# Serialize the dataset elements
def serialize_example(image, age, sex, label):
    feature = {
        'image': _bytes_feature(image),
        'age': _float_feature(age),
        'sex': _int64_feature(sex),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Write the serialized data to TFRecord
def write_tfrecord(dataset, filename):
    writer = tf.io.TFRecordWriter(filename)

    record_count = 2000

    for i,((image, age, sex), label) in enumerate(dataset.take(record_count).as_numpy_iterator()):
        print(i+1,"/",record_count)
        example = serialize_example(image, age, sex, label)
        writer.write(example)
    writer.close()



# Define the feature description, which will be used to parse the TFRecord file
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'age': tf.io.FixedLenFeature([], tf.float32),
    'sex': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the feature description.
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the image data
    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
    image = tf.reshape(image, image_shape)  # Reshape the image to its original shape

    # Get the age, sex, and label
    age = parsed_features['age']
    sex = parsed_features['sex']
    label = parsed_features['label']

    return (image, age, sex), label
