
#import tensorflow as tf
#from preprocessing import _parse_function, generator_dataset


ADNIMerge_properties = {
    "path":"adni documentation/ADNIMERGE_13Apr2024.csv",
    "features":[
        {
            "name":"ID",
            "column":"PTID"
        },
        {
            "name":"Visit Code 2",
            "column":"VISCODE",
            #adni merge combines sc and bl visits into just bl because they're both initial visits

        },
        {
            "name":"Age",
            "column":"AGE",
        },
        {
            "name":"Race",
            "column":"PTRACCAT",
            "map":{
                "Black":0,
                "White":1,
                "Asian":2,
                "More than one":3,
                "Am Indian/Alaskan":4,
                "Unknown":5
            }
        }
    ]
}

vitals_properties = {
    "path":"adni documentation/VITALS_16Apr2024.csv",
    "features":{
        "Patient ID":{
            "name":"Patient ID",
            "column":"PTID"
        },
        "Visit Code":{
            "name":"Visit Code",
            "column":"VISCODE",
        },
        "Visit Code 2":{
            "name":"Visit Code 2",
            "column":"VISCODE2",
        },
        "Weight":{
            "name":"Weight",
            "column":"VSWEIGHT"
        },
        "Weight Unit":{
            "name":"Weight Unit",
            "column":"VSWTUNIT"
        }
    }
    #"features":[
    #    "PTID",
    #    "VISCODE", #also viscode2 but idk the difference
    #    "VSWEIGHT",
    #    "VSWTUNIT", #gotta multiply this to weight
    #    "VSHEIGHT",
    #    "VSHTUNIT", # this one is weird af, do u divide here?
    #    "VSBPSYS", # blood pressure, systolic
    #    "VSBPDIA", #blood pressure, diastolic
    #    "VSPULSE", #bpm
    #]
}

demographic_properties = {
    "path": "adni documentation/PTDEMOG_16Apr2024.csv",
    "features":{
        "Patient ID":{
            "name":"Patient ID",
            "column":"PTID"
        },
        "Sex":{
            "name":"Sex",
            "column":"PTGENDER",
            "map":{
                "1":[1,0],
                "2":[0,1]
            }
        },
        "Race":{
            "name":"Race",
            "column":"PTRACCAT",
            "map":{
                "-4":[0,0,0,0,0,0,0], #0,
                "1":[1,0,0,0,0,0,0],  #1,
                "2":[0,1,0,0,0,0,0],  #2,
                "3":[0,0,1,0,0,0,0], #3,
                "4":[0,0,0,1,0,0,0], #4,
                "5":[0,0,0,0,1,0,0], #5,
                "6":[0,0,0,0,0,1,0], #6,
                "7":[0,0,0,0,0,0,1], #7,
                "4|5":[0,0,0,1,1,0,0], #5,
                "1|5":[1,0,0,0,1,0,0], #5,
                "3|4|5":[0,0,1,1,1,0,0]#5
            }
        },
    }
    #"features":[
    #    "PTID",
    #    "PTGENDER",
    #    "PTDOB", # MMM-YY, e.g. Apr-51
    #    "PTHAND", # [right, left]=[1,2]
    #    "PTMARRY",
    #    "PTEDUCAT",
    #    "PTETHCAT", #ethnicity category
    #    "PTRACCAT", #race category # 4, 5, 4|5, -4
    #    "PTWORKHS",
    #    "PTNOTRT", # not retired
    #    "PTRTYR", # retirement year YYYY
    #]
}


ADNI3_set = {
    "name": "ADNI3 T1 AXIAL 54 DEPTH",

    "link":[
        {
            "table":{
                "path": "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI3_T1_AXIAL_54_DEPTH_4_07_2024.csv",
                "features":{
                    "Image ID": {
                        "name":"Image ID",
                        "column":"Image Data ID"
                    },
                    "Patient ID":{
                        "name":"Patient ID",
                        "column":"Subject"
                    },
                    "Visit Code":{
                        "name":"Visit Code",
                        "column":"Visit"
                    },
                    "Age":{
                        "name":"Age",
                        "column":"Age"
                    },
                    "Sex":{
                        "name":"Sex",
                        "column":"Sex",
                        "map":{
                            'M':[1,0], 'Male':[1,0], # male
                            'F':[0,1], 'Female':[0,1] # female
                        }
                    },
                    "Group":{
                        "name":"Group",
                        "column":"Group",
                        "map":{
                            'CN':0, # cognitively normal
                            'SMC':1, # subjective memory complaints
                            'EMCI':2, # early mild cognitive impairment
                            'MCI':3, # mild cognitive impairment
                            'LMCI':4, # late mild cognitive impairment
                            'AD':5, 'Dementia':5 # alzheimers dementia
                        }
                    }
                }
            },
            "keys":["Image ID"]
        },
        {
            "table":vitals_properties,
            "keys":["Patient ID", "Visit Code"]
        },
        {
            "table":demographic_properties,
            "keys":["Patient ID"]
        }
    ],

    "images_directory" : "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI",
    "image_format" : "dcm",
    "image_shape" : (54, 78, 78, 1), #(depth, width, height, channels)

    #"csv_path" : "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI3_T1_AXIAL_54_DEPTH_4_07_2024.csv",
    "classes" : ["CN","SMC","EMCI","MCI","LMCI","AD"],
    #"label_map" : {
    #    'CN':0, # cognitively normal
    #    'SMC':1, # subjective memory complaints
    #    'EMCI':2, # early mild cognitive impairment
    #    'MCI':3, # mild cognitive impairment
    #    'LMCI':4, # late mild cognitive impairment
    #    'AD':5, 'Dementia':5 # alzheimers dementia
    #}
}

ADNI1_set = {
    "name":"ADNI1_Complete 1Yr 1.5T",

    "link":[
        {
            "table":{
                "path": "data/ADNI1_Complete 1Yr 1.5T/ADNI1_Complete_1Yr_1.5T_4_14_2024.csv",
                "features":{
                    "Image ID": {
                        "name":"Image ID",
                        "column":"Image Data ID"
                    },
                    "Patient ID":{
                        "name":"Patient ID",
                        "column":"Subject"
                    },
                    "Visit Code":{
                        "name":"Visit Code",
                        "column":"Visit"
                    },
                    "Age":{
                        "name":"Age",
                        "column":"Age"
                    },
                    "Sex":{
                        "name":"Sex",
                        "column":"Sex",
                        "map":{
                            'M':[1,0], 'Male':[1,0], # male
                            'F':[0,1], 'Female':[0,1] # female
                        }
                    },
                    "Group":{
                        "name":"Group",
                        "column":"Group",
                        "map":{
                            "CN":0,
                            "MCI":1,
                            "AD":2, "Dementia":2
                        }
                    }
                }
            },
            "keys":["Image ID"]
        },
        {
            "table":vitals_properties,
            "keys":["Patient ID", "Visit Code"]
        },
        {
            "table":demographic_properties,
            "keys":["Patient ID"]
        }
    ],

    "images_directory" : "data/ADNI1_Complete 1Yr 1.5T/ADNI",
    "image_format" : "nii",
    "image_shape" : (128, 128, 90, 1), # not source, what the images are resized to.     #(256, 256, 180, 1), # (width, height, depth, channels)

    #"csv_path" : "data/ADNI1_Complete 1Yr 1.5T/ADNI1_Complete_1Yr_1.5T_4_14_2024.csv",
    "classes" : ["CN","MCI","AD"],
    #"label_map" : {
    #    "CN":0,
    #    "MCI":1,
    #    "AD":2, "Dementia":2
    #},

    #"dataset" : lambda : generator_dataset()
}

preprocessed_set = {
    "name":"ADNI1_Complete 1Yr 1.5T (preprocessed)",
    "file_location" : 'data/my_dataset.tfrecord',
    "image_shape" : (128, 128, 90, 1),

    "classes" : ["CN","MCI","AD"],

    #"dataset": lambda : tf.data.TFRecordDataset('data/ADNI1_dataset_2000.tfrecord').map(_parse_function)
}


dataset_props = ADNI1_set

image_shape = dataset_props.get("image_shape")
classes = dataset_props.get("classes")
num_classes = len(classes)

images_directory = dataset_props.get("images_directory")

#csv_path = dataset_props.get("csv_path")

ADNI_merge = "adni documentation/ADNIMERGE_13Apr2024.csv"
vitals = "adni documentation/VITALS_16Apr2024.csv"
demographic = "adni documentation/PTDEMOG_16Apr2024.csv"

label_map = dataset_props.get("label_map")

batchSize = 2
epochs = 1
debug = True
architecture = 'ResNet'