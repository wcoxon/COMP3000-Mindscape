import tensorflow as tf
import preprocessing


vitals_reference = {
    "path":"adni documentation/VITALS_16Apr2024.csv",
    "features":{
        "Patient ID":{"column":"PTID"},
        "Visit Code":{"column":"VISCODE"},
        "Visit Code 2":{"column":"VISCODE2"},
        "Weight":{"column":"VSWEIGHT"},
        "Weight Unit":{"column":"VSWTUNIT"},
        "Height":{"column":"VSHEIGHT"},
        "Height Unit":{"column":"VSHTUNIT"},
        "Systolic BP":{"column":"VSBPSYS"},
        "Diastolic BP":{"column":"VSBPDIA"},
        "Pulse":{"column":"VSPULSE"}
    },
    "keys":["Patient ID", "Visit Code"]
}
demographic_reference = {
    "path": "adni documentation/PTDEMOG_16Apr2024.csv",
    "features":{
        "Patient ID":{"column":"PTID"},
        "Sex":{
            "column":"PTGENDER",
            "map":lambda x:{"1":[1,0],"2":[0,1]}[x]
            #1=Male; 2=Female
        },
        "Handedness" : {"column":"PTHAND", "map":lambda x : int(x)}, # 1=Right; 2=Left	
        "Marital Status":{"column":"PTMARRY", "map":lambda x : int(x)}, # 1=Married; 2=Widowed; 3=Divorced; 4=Never married; 5=Unknown
        "Work History" : {"column":"PTWORKHS", "map":lambda x : int(x)}, # 1=Yes; 0=No
        "Education" : {"column":"PTEDUCAT", "map":lambda x : int(x)}, # 0..20
        "Ethnic Category" : {"column":"PTETHCAT"}, # 1=Hispanic or Latino; 2=Not Hispanic or Latino; 3=Unknown
        "Retired" : {"column":"PTNOTRT", "map":lambda x : int(x)}, # 1=yes; 0=no i think
        "Race":{
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
            # 1=American Indian or Alaskan Native; 2=Asian; 3=Native Hawaiian or Other Pacific Islander; 4=Black or African American; 5=White; 6=More than one race; 7=Unknown
        }
    },
    "keys":["Patient ID"]
}
psych_reference = {
    "path":"adni documentation/UWNPSYCHSUM_19Apr2024.csv",
    "features":{
        "Roster ID" : {"column":"RID"},
        "Visit Code" : {
            "column":"VISCODE",
            "map":lambda x : ("sc" if x=="bl" else x)
        },
        "Visit Code 2" : {
            "column":"VISCODE2",
            "map":lambda x : ("sc" if x=="bl" else x)
        },
        "Memory Score" : {"column":"ADNI_MEM"},
        "Executive Function" : {"column":"ADNI_EF"},
        "Executive Function 2" : {"column":"ADNI_EF2"},
        "Language Cognition" : {"column":"ADNI_LAN"},
        "Visuo-spatial Score" : {"column":"ADNI_VS"}
    },
    "keys":["Roster ID", "Visit Code"]
}


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
handedness = {
    "name":"Handedness",
    "tostring" : lambda x: ["Right","Left"][x-1],
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
}
marital_status = {
    "name":"Marital Status",
    "tostring" : lambda x: ["Married","Widowed","Divorced","Never Married","Unknown"][x-1],
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
}
work_history = {
    "name":"Work History",
    "tostring" : lambda x: ["No","Yes"][x],
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
}
education = {
    "name":"Education",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
}
ethnicity = {
    "name":"Ethnic Category",
    "tostring" : lambda x: str(x),
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
}
retired = {
    "name":"Retired",
    "tostring" : lambda x: ["No","Yes"][x],
    "signature":tf.TensorSpec(shape=(), dtype=tf.uint8),
    "input_shape":(1,)
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

features = {
    "Memory Score":memory_score,
    "Executive Function":executive_function,
    "Language Cognition":language_cognition,
    "Visuo-spatial Score":visuo_spatial,
    "Executive Function 2":executive_function_2,
    "Age":age,
    "Sex":sex,
    "Handedness":handedness,
    "Marital Status":marital_status,
    "Work History":work_history,
    "Education":education,
    "Ethnicity":ethnicity,
    "Retired":retired,
    "Race":race,
    "Weight":weight,
    "Weight Unit":weight_unit,
    "Systolic BP":systolic_BP,
    "Diastolic BP":diastolic_BP,
    "Group":group

}

ADNI3_set = {
    "name": "ADNI3 T1 AXIAL 54 DEPTH",

    "refs":[
        {
            "path": "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI3_T1_AXIAL_54_DEPTH_4_07_2024.csv",
            "features":{
                "Image ID": {"column":"Image Data ID"},
                "Patient ID":{"column":"Subject"},
                "Roster ID":{
                    "column":"Subject",
                    "map": lambda x : str(int(x.split("_")[-1]))
                },
                "Visit Code":{"column":"Visit"},
                "Age":{"column":"Age"},
                "Sex":{
                    "column":"Sex",
                    "map":lambda x : {'M':[1,0], 'Male':[1,0],'F':[0,1], 'Female':[0,1]}[x]
                },
                "Group":{
                    "column":"Group",
                    "map":lambda x : {'CN':0, 'SMC':1, 'EMCI':2, 'MCI':3, 'LMCI':4, 'AD':5, 'Dementia':5}[x]
                },
                "Series Description":{"column":"Description"}
            },
            "keys":["Image ID"]
        },
        vitals_reference,
        demographic_reference,
        psych_reference
    ],
    "features":[
        age,
        sex,
        handedness,
        marital_status,
        race
    ],

    "images_path" : "data/ADNI3 T1 AXIAL 54 DEPTH/ADNI/*/*/*/*",
    "image_format" : "dcm",
    "getImageID": lambda path : path.split("\\")[-1],
    "image_input_shape" : (54, 78, 78, 1),
    "read_image":preprocessing.loadDicom,
    "image_transformations":[preprocessing.normalize_process],

    "classes" : ["CN","SMC","EMCI","MCI","LMCI","AD"],
}

ADNI1_refs = [
    {
        "path": "data/ADNI1_Complete 1Yr 1.5T/ADNI1_Complete_1Yr_1.5T_4_14_2024.csv",
        "features":{
            "Image ID": {"column":"Image Data ID"},
            "Patient ID":{"column":"Subject"},
            "Roster ID":{
                "column":"Subject",
                "map": lambda x : str(int(x.split("_")[-1]))
            },
            "Visit Code":{"column":"Visit"},
            "Age":{"column":"Age"},
            "Sex":{
                "column":"Sex",
                "map": lambda x : {'M':[1,0], 'Male':[1,0], 'F':[0,1], 'Female':[0,1]}[x]
            },
            "Group":{
                "column":"Group",
                "map": lambda x : {"CN":0,"MCI":1, "AD":2, "Dementia":2}[x]
            }
        },
        "keys":["Image ID"]
    },
    psych_reference,
    vitals_reference,
    demographic_reference
]

ADNI1_set = {
    "name" : "ADNI1_Complete 1Yr 1.5T",

    "refs" : ADNI1_refs,
    "features" : [memory_score, age],

    "images_path" : "data/ADNI1_Complete 1Yr 1.5T/ADNI/*/*/*/*/*.nii",
    "getImageID" : lambda path : path.split("\\")[-2],
    "image_format" : "nii",
    "image_input_shape" : (128, 128, 90, 1),
    "read_image" : preprocessing.loadNii,
    "image_transformations" : [
        preprocessing.resize_process((128, 128, 90, 1)), 
        preprocessing.normalize_process
    ],

    "classes" : ["CN","MCI","AD"]
}

ADNI1_preprocessed_set = {
    "name":"ADNI1_Complete 1Yr 1.5T (preprocessed)",

    "refs":ADNI1_refs,
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
    "read_image":preprocessing.loadNii,
    "image_transformations":[],

    "classes" : ["CN","MCI","AD"],
}


batchSize = 1
epochSize = 1
epochs = 5

report_state = 0
data_select_state = 1
feature_select_state = 2
data_browse_state = 3
architecture_select_state = 4
train_state = 5