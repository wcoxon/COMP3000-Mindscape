
import tensorflow as tf
#from preprocessing import _parse_function, generator_dataset
import preprocessing

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
        },
        "Height":{
            "name":"Height",
            "column":"VSHEIGHT"
        },
        "Height Unit":{
            "name":"Height Unit",
            "column":"VSHTUNIT"
        },
        "Systolic BP":{
            "name":"Systolic BP",
            "column":"VSBPSYS"
        },
        "Diastolic BP":{
            "name":"Diastolic BP",
            "column":"VSBPDIA"
        },
        "Pulse":{
            "name":"Pulse",
            "column":"VSPULSE"
        }
    }
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
            "map":lambda x:{"1":[1,0],"2":[0,1]}[x]
            #1=Male; 2=Female
        },
        "Handedness":{
            "name":"Handedness",
            "column":"PTHAND"
            #1=Right;2=Left	
        },
        "Marital Status":{
            "name":"Marital Status",
            "column":"PTMARRY"
            #1=Married; 2=Widowed; 3=Divorced; 4=Never married; 5=Unknown
        },
        "Work History":{
            "name":"Work History",
            "column":"PTWORKHS"
            #1=Yes; 0=No
        },
        "Education":{
            "name":"Education",
            "column":"PTEDUCAT"
            #0..20
        },
        "Ethnic Category":{
            "name":"Ethnic Category",
            "column":"PTETHCAT"
            #1=Hispanic or Latino; 2=Not Hispanic or Latino; 3=Unknown
        },
        "Race":{
            "name":"Race",
            "column":"PTRACCAT",
            "map": lambda x : {
                "1":[1,0,0,0,0,0], 
                "2":[0,1,0,0,0,0], 
                "3":[0,0,1,0,0,0], 
                "4":[0,0,0,1,0,0], 
                "5":[0,0,0,0,1,0], 
                "6":[0,0,0,0,0,1], 
                "4|5":[0,0,0,1,1,0],
                "1|5":[1,0,0,0,1,0],
                "3|4|5":[0,0,1,1,1,0],
                "-4":[0,0,0,0,0,0],
                "7":[0,0,0,0,0,0], 
            }[x]
            #1=American Indian or Alaskan Native; 2=Asian; 3=Native Hawaiian or Other Pacific Islander; 4=Black or African American; 5=White; 6=More than one race; 7=Unknown
        },

    }
    #"features":[
    #    "PTNOTRT", # not retired
    #    "PTRTYR", # retirement year YYYY
    #]
}


ADNI1_links = [
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
                    "Roster ID":{
                        "name":"Roster ID",
                        "column":"Subject",
                        "map": lambda x : str(int(x.split("_")[-1]))
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
                        "map": lambda x : {'M':[1,0], 'Male':[1,0], 'F':[0,1], 'Female':[0,1]}[x]
                    },
                    "Group":{
                        "name":"Group",
                        "column":"Group",
                        "map": lambda x : {"CN":0,"MCI":1, "AD":2, "Dementia":2}[x]
                    }
                }
            },
            "keys":["Image ID"]
        },
        {
            "table":{
                "path":"adni documentation/UWNPSYCHSUM_19Apr2024.csv",
                "features":{
                    "Roster ID":{
                        "name":"Roster ID",
                        "column":"RID"
                    },
                    "Visit Code":{
                        "name":"Visit Code",
                        "column":"VISCODE",
                        "map":lambda x : ("sc" if x=="bl" else x)

                    },
                    "Visit Code 2":{
                        "name":"Visit Code 2",
                        "column":"VISCODE2",
                        "map":lambda x : ("sc" if x=="bl" else x)
                    },
                    "Memory Score":{
                        "name":"Memory Score",
                        "column":"ADNI_MEM"
                    },
                    "Executive Function":{
                        "name":"Executive Function",
                        "column":"ADNI_EF"
                    },
                    "Executive Function 2":{
                        "name":"Executive Function 2",
                        "column":"ADNI_EF2"
                    },
                    "Language Cognition":{
                        "name":"Language Cognition",
                        "column":"ADNI_LAN"
                    },
                    "Visuo-spatial Score":{
                        "name":"Visuo-spatial Score",
                        "column":"ADNI_VS"
                    }
                }
            },
            "keys":["Roster ID", "Visit Code"]
        },
        {
            "table":vitals_properties,
            "keys":["Patient ID", "Visit Code"]
        },
        {
            "table":demographic_properties,
            "keys":["Patient ID"]
        }
    ]


memory_score = {
    "name":"Memory Score",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float32),
    "input_shape":(1,)
}
executive_function = {
    "name": "Executive Function",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float32),
    "input_shape":(1,)
}
language_cognition = {
    "name":"Language Cognition",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float32),
    "input_shape":(1,)
}
visuo_spatial = {
    "name": "Visuo-spatial Score",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float32),
    "input_shape":(1,)
}
executive_function_2 = {
    "name": "Executive Function 2",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float32),
    "input_shape":(1,)
}

age = {
    "name":"Age",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float16),
    "input_shape":(1,)
}
sex = {
    "name":"Sex",
    "tostring" : lambda x: ["Female","Male"][x[0]],
    "signature":tf.TensorSpec(shape=(2,), dtype=tf.uint8),
    "input_shape":(2,)
}
race = {
    "name":"Race",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(6,), dtype=tf.uint8),
    "input_shape":(6,)
}
weight = {
    "name":"Weight",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float16),
    "input_shape":(1,)
},
weight_unit = {
    "name":"Weight Unit",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float16),
    "input_shape":(1,)
}
systolic_BP = {
    "name":"Systolic BP",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float16),
    "input_shape":(1,)
}
diastolic_BP = {
    "name":"Diastolic BP",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.float16),
    "input_shape":(1,)
}
group = {
    "name":"Group",
    "tostring" : lambda x: ["CN","MCI","AD"][x],
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
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
                        "map":lambda x : {'M':[1,0], 'Male':[1,0],'F':[0,1], 'Female':[0,1]}[x]
                    },
                    "Group":{
                        "name":"Group",
                        "column":"Group",
                        "map":lambda x : {'CN':0, 'SMC':1, 'EMCI':2, 'MCI':3, 'LMCI':4, 'AD':5, 'Dementia':5}[x]
                    },
                    "Series Description":{
                        "name":"Series Description",
                        "column":"Description"
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

    "images_path" : "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI/*/*/*/*",
    "image_format" : "dcm",
    "getImageID": lambda path : path.split("\\")[-1],
    "image_input_shape" : (54, 78, 78, 1), #(depth, width, height, channels)
    "image_transformations":[preprocessing.normalize_process],

    "features":[
        age,
        sex,
        race
    ],

    "classes" : ["CN","SMC","EMCI","MCI","LMCI","AD"],
    
}

ADNI1_set = {
    "name":"ADNI1_Complete 1Yr 1.5T",

    "link":ADNI1_links,
    
    "features":[
        memory_score,
        age
    ],

    "images_path" : "data/ADNI1_Complete 1Yr 1.5T/ADNI/*/*/*/*/*.nii",
    "getImageID": lambda path : path.split("\\")[-2],
    "image_format" : "nii",
    "image_input_shape" : (128, 128, 90, 1),
    "image_transformations":[preprocessing.resize_process((128, 128, 90, 1)),preprocessing.normalize_process],

    "classes" : ["CN","MCI","AD"]
}

ADNI1_preprocessed_set = {
    "name":"ADNI1_Complete 1Yr 1.5T (preprocessed)",

    "link":ADNI1_links,
    "features":[
        memory_score,
        executive_function,
        executive_function_2,
        visuo_spatial,
        language_cognition
    ],

    "images_path" : "data/preprocessed/256_256_180/*.nii",
    "getImageID": lambda path : path.split("\\")[-1].replace(".nii",""),
    "image_format" : "nii",
    "image_input_shape" : (256, 256, 180, 1),
    "image_transformations":[],

    "classes" : ["CN","MCI","AD"],
}

#dataset_props = ADNI1_preprocessed_set

#image_shape = dataset_props["image_input_shape"]
#classes = dataset_props["classes"]
#num_classes = len(classes)
#
#images_directory = dataset_props["images_path"]

#ADNI_merge = "adni documentation/ADNIMERGE_13Apr2024.csv"
#vitals = "adni documentation/VITALS_16Apr2024.csv"
#demographic = "adni documentation/PTDEMOG_16Apr2024.csv"

batchSize = 1
epochs = 5
debug = True
#architecture = 'ResNet'


