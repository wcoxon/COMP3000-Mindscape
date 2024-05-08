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

    def get_class_distribution(self):

        class_counts = np.array([0.0]*len(self.classes))

        refs = self.manifest.get("refs")
        if(refs==None):
            for (_input, _output) in self.dataset:
                dx = _output.numpy()
                class_counts[dx]+=1
            return class_counts


        with open(refs[0]["path"], mode='r') as file:
            for row in csv.DictReader(file):
                feature = refs[0]["features"]["Group"]
                dx = feature["map"](row[feature["column"]])
                class_counts[dx]+=1

        total_samples = sum(class_counts)
        class_counts /= float(total_samples)

        return class_counts
    

    def getMeta(self,constraint):
        metadata = constraint
        
        for ref in self.manifest["refs"]:
            with open(ref["path"], mode='r') as file:
                reader = csv.DictReader(file)

                #the ref's features denoted as keys to match to
                key_features = [[feature_name, ref["features"][feature_name]] for feature_name in ref["keys"]]

                for row in reader:

                    matches = [(row[feature["column"]] if "map" not in feature else feature["map"](row[feature["column"]]))==metadata[feature_name] for feature_name, feature in key_features]

                    if all(matches):
                        for feature_name,feature in ref["features"].items():
                            if metadata.get(feature_name):
                                # if we already have this feature, skip it
                                continue

                            feature_value = row[feature["column"]]
                            if feature_value=="":
                                #data is missing from dataset
                                feature_value = None
                            elif feature.get("map"):
                                feature_value = feature["map"](feature_value)

                            metadata[feature_name] = feature_value
            
            if debug : print(ref["path"],":",metadata)

        return metadata
    
    def generate_sample(self,path):
        
        path_string = path.numpy().decode("ascii") # image directory path
        imageID = self.manifest["getImageID"](path_string) # extract image ID from file path
        
        if(self.manifest["image_format"]=="nii"): volume = loadNii(path_string,self.manifest["image_transformations"])
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
def loadNii(imagePath,processes):
    
    nii = nib.load(imagePath)
    image = nii.get_fdata()

    for process in processes:
        image = process(image)

    image = np.expand_dims(image,axis=-1)

    return image

def loadDicom(imagePath,image_shape):

    imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.dcm"%imagePath,shuffle=False) ] # collect all the dcm file paths
    dcmArray = [ dicom.dcmread(path) for path in imagePaths ]
    
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
