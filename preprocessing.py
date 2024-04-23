import numpy as np

import pydicom as dicom
import nibabel as nib
import csv

import tensorflow as tf

from scipy.ndimage import zoom

debug = False

class dataset_manager():

    def __init__(self, manifest):
        self.manifest = manifest
        self.classes = manifest["classes"]
        self.image_shape = manifest["image_input_shape"]

        self.class_distribution = self.get_class_distribution()
        self.class_weight = {i:1/d for i, d in enumerate(self.class_distribution)}

        self.dataset = self.generator_dataset()

    def get_class_distribution(self):

        class_counts = np.array([0.0]*len(self.classes))

        refs = self.manifest.get("link")
        if(refs==None):
            for (_input, _output) in self.dataset:
                dx = _output.numpy()
                class_counts[dx]+=1
            return class_counts


        with open(refs[0]["table"]["path"], mode='r') as file:
            for row in csv.DictReader(file):
                feature = refs[0]["table"]["features"]["Group"]
                dx = feature["map"](row[feature["column"]])
                class_counts[dx]+=1

        total_samples = sum(class_counts)
        class_counts /= float(total_samples)

        return class_counts
    

    def getMeta(self,constraint):

        metadata = constraint

        for ref in self.manifest["link"]:
            with open(ref["table"]["path"], mode='r') as file:
                reader = csv.DictReader(file)

                key_features = [ref["table"]["features"][key_feature_name] for key_feature_name in ref["keys"]]

                for row in reader:

                    matches = [ (row[feature["column"]] if "map" not in feature else feature["map"](row[feature["column"]]))==metadata[feature["name"]] for feature in key_features]

                    if all(matches):
                        for feature in ref["table"]["features"].values():
                            if metadata.get(feature["name"]):
                                # if we already have this feature, skip it
                                continue

                            feature_value = row[feature["column"]]
                            if feature_value=="":
                                #data is missing from dataset
                                feature_value = None
                            elif feature.get("map"):
                                feature_value = feature["map"](feature_value)

                            metadata[feature["name"]] = feature_value
            
            if debug : print(ref["table"]["path"],":",metadata)

        return metadata
    
    def generate_sample(self,path):
        
        path_string = path.numpy().decode("ascii") # image directory path
        imageID = self.manifest["getImageID"](path_string) # extract image ID from file path
        
        if(self.manifest["image_format"]=="nii"): volume = loadNii(path_string,imageID,self.manifest["image_transformations"])
        elif(self.manifest["image_format"]=="dcm"): volume = loadDicom(path_string,self.manifest["image_input_shape"])
        metadata = self.getMeta({"Image ID" : imageID}) # query CSV data

        return (
            (
                volume, 
                *[metadata[feature["name"]] for feature in self.manifest["features"]]
            ),
            metadata["Group"]
        )

    def generateDataset(self):

        for image_path in tf.data.Dataset.list_files(self.manifest["images_path"],shuffle=False): #for each image
            yield self.generate_sample(image_path)

    def generator_dataset(self):


        return tf.data.Dataset.from_generator(
            self.generateDataset,
            output_signature=(
                (
                    tf.TensorSpec(shape=self.manifest["image_input_shape"], dtype=tf.float32), # image
                    *[feature["signature"] for feature in self.manifest["features"]]
                ),
                tf.TensorSpec(shape=(), dtype=tf.uint8)
            )
        )


def normalizePixels(pixelArray):
    return pixelArray / pixelArray.max()


def loadDicom(imagePath,image_shape):

    imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.dcm"%imagePath,shuffle=False) ] # collect all the dcm file paths
    dcmArray = [ dicom.dcmread(path) for path in imagePaths ]

    dcmData = dcmArray[0]
    if debug:
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
        if(debug): print("dicom missing positions")

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

def resize_process(shape):
    def resize_volume(v):
        return zoom(v, (shape[0]/v.shape[0], shape[1]/v.shape[1], shape[2]/v.shape[2]))

    return resize_volume

def normalize_process(v):
    p1 = np.percentile(v, 1)
    p99 = np.percentile(v, 99)
    volume_normalized = (v - p1) / (p99 - p1)
    return np.clip(volume_normalized, 0, 1)

#import os
def loadNii(imagePath,imageID,processes):
    
    nii = nib.load(imagePath)
    image = nii.get_fdata()

    for process in processes:
        image = process(image)

    #if "write" in processes:
        #processed_image = nib.Nifti1Image(image, nii.affine)
        #nib.save(processed_image, "data/preprocessed/%s_%s_%s/%s.nii"%(*image_shape[:3],imageID))

    image = np.expand_dims(image,axis=-1)

    return image


#
## Define functions to convert data to tf.train.Feature
#_bytes_feature = lambda value : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])) #Returns a bytes_list from a string / byte.
#_float_feature = lambda value : tf.train.Feature(float_list=tf.train.FloatList(value=[value])) #Returns a float_list from a float / double.
#_int64_feature = lambda value : tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) #Returns an int64_list from a bool / enum / int / uint.
#
## Serialize the dataset elements
#def serialize_example(image, age, sex, label):
#    feature = {
#        'image': _bytes_feature(image),
#        'age': _float_feature(age),
#        'sex': _int64_feature(sex),
#        'label': _int64_feature(label),
#    }
#    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#    return example_proto.SerializeToString()
#
## Write the serialized data to TFRecord
#def write_tfrecord(dataset, filename):
#    writer = tf.io.TFRecordWriter(filename)
#
#    record_count = 2000
#
#    for i,((image, age, sex), label) in enumerate(dataset.take(record_count).as_numpy_iterator()):
#        print(i+1,"/",record_count)
#        example = serialize_example(image, age, sex, label)
#        writer.write(example)
#    writer.close()
#
#
#
## Define the feature description, which will be used to parse the TFRecord file
#feature_description = {
#    'image': tf.io.FixedLenFeature([], tf.string),
#    'age': tf.io.FixedLenFeature([], tf.float32),
#    'sex': tf.io.FixedLenFeature([], tf.int64),
#    'label': tf.io.FixedLenFeature([], tf.int64),
#}
#
#def _parse_function(example_proto):
#    # Parse the input `tf.train.Example` proto using the feature description.
#    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
#
#    # Decode the image data
#    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
#    image = tf.reshape(image, image_shape)  # Reshape the image to its original shape
#
#    # Get the age, sex, and label
#    age = parsed_features['age']
#    sex = parsed_features['sex']
#    label = parsed_features['label']
#
#    return (image, age, sex), label
#