import numpy as np

import pydicom as dicom
import nibabel as nib
import csv

import tensorflow as tf

from scipy.ndimage import zoom

from env import image_shape,label_map,sex_map,viscode_map,imageCSVPath,ADNI_merge,directory

def getMeta(imageID):

    age = None
    sex = None
    dx = None

    patient_ID = None
    viscode = None

    with open(imageCSVPath, mode='r') as file:
        for row in csv.DictReader(file):
            if row['Image Data ID'] == imageID:

                patient_ID = row["Subject"]
                viscode = viscode_map[row["Visit"]]

                age = int(row["Age"])
                sex = sex_map[row["Sex"]]
                dx = label_map[row['Group']]
                break

    with open(ADNI_merge,mode='r') as mergeFile:
        for row in csv.DictReader(mergeFile):
            if row["PTID"] == patient_ID and row["VISCODE"] == viscode:
                if row["AGE"]!="": 
                    age = float(row["AGE"])
                break

    return { "group": dx, "age": age, "sex":sex}

def normalizePixels(pixelArray):
    return pixelArray / pixelArray.max()

debug = False

def generateDataset():
    for subjectDir in tf.data.Dataset.list_files("%s/*/*/*/*" % directory): #for each scan
        
        scanDir = subjectDir.numpy().decode("ascii") # image directory path

        imagePaths = [ filePath.numpy().decode("ascii") for filePath in tf.data.Dataset.list_files("%s\\*.dcm"%scanDir,shuffle=False) ] # collect all the dcm file paths
        
        # read Dicom images and sort by depth
        dcmArray = [ dicom.dcmread(path) for path in imagePaths ]
        try:
            dcmArray.sort(key=lambda dcm: float(dcm.get((0x0020, 0x1041)).repval[1:-1]))
        except AttributeError:
            if(debug): print("dicom missing positions")


        # convert dicoms to pixel arrays
        imageArray = np.array([dcm.pixel_array for dcm in dcmArray]) 

        # set shape
        imageArray = np.reshape(imageArray,imageArray.shape[-3:])

        # add channels dimension
        imageArray = np.expand_dims(imageArray, axis=-1)

        if(imageArray.shape != image_shape):
            imageArray = np.array([ tf.image.resize(i, image_shape[1:3]) for i in imageArray]) # resize images
        
        imageArray = normalizePixels(imageArray) # normalize pixel values

        imageID = scanDir.split("\\")[-1] # extract image ID from file path

        metadata = getMeta(imageID) # query CSV data
        age = metadata["age"]
        sex = metadata["sex"]
        diagnosis = metadata["group"]

        dcmData = dcmArray[0]
        # get weight from dicom properties (some are missing this though)
        #weight = dcmData.get("PatientWeight")

        if debug:
            print(scanDir.replace('\\','\\\\'))

            print(metadata)

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
            print()

        yield (
            (
                imageArray, 
                age, 
                sex,
                #weight
            ),
            diagnosis
        )


dataset = tf.data.Dataset.from_generator(
    generateDataset,
    output_signature=(
        (
            tf.TensorSpec(shape=image_shape, dtype=tf.float32), # image
            tf.TensorSpec(shape=(), dtype=tf.float16), # age
            tf.TensorSpec(shape=(), dtype=tf.uint8), # sex
            #tf.TensorSpec(shape=(), dtype=tf.float16) # weight
        ),
        tf.TensorSpec(shape=(), dtype=tf.uint8)
    )
)
