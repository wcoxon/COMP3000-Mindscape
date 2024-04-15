
# ADNI/[collection name]/ADNI/[subjectID]/[series description]/[scan date]/[image id]/[images]

ADNI3_set = ["ADNI3 T1 AXIAL 54 DEPTH","ADNI3_T1_AXIAL_54_DEPTH_4_07_2024"]

ADNI_merge = "adni documentation\\ADNIMERGE_13Apr2024.csv"

setpaths = ADNI3_set

directory = "data/%s/ADNI"%setpaths[0]
imageCSVPath ="data/%s/%s.csv"%(setpaths[0], setpaths[1])

sex_map = {
    'M':0, # male
    'Male':0,
    'F':1, # female
    'Female':1
}

label_map = {
    'CN':0, # cognitively normal
    'SMC':1, # subjective memory complaints
    'EMCI':2, # early mild cognitive impairment
    'MCI':3, # mild cognitive impairment
    'LMCI':4, # late mild cognitive impairment
    'AD':5, 'Dementia':5 # alzheimers dementia
}
num_classes = len(set(label_map.values()))

image_shape = (
    54, # depth
    78, # width
    78, # height
    1    # channels
)

viscode_map = {
    "bl":"bl", 
    "sc":"bl",
    "init":"bl",
    "y1":"m12",
    "y2":"m24",
    "y3":"m36",
    "y4":"m48",
    "y5":"m60",
    "y6":"m72",
    "y7":"m84",
    "y8":"m96",
    "y9":"m108",
    "y10":"m120",
}


batchSize = 2