import os
from pathlib import Path

# -----------------------
# -- Global variable : --
# -----------------------

# 01 : Pre-processing
# RANDOM_SEED = 42 # As foretold in the book of truth, 42 is the answer to everything

# 02 : Modeling
N_EPOCH = 10
MODEL_DIC = {
    'VGG16' : { # 0
        'Size' : '550MB',
        'Speed' : '2h10min',
        'In_features' : 4096,
        'Accuracy' : 'Low'
    },
    'ResNet18' : { # 1
        'Size' : '44MB',
        'Speed' : '18m',
        'In_features' : 512,
        'Accuracy' : 'High'
    },
    'DenseNet121' : { # 2
        'Size' : '30MB',
        'Speed' : '1h',
        'In_features' : 1024,
        'Accuracy' : 'High'
    },
    'ResNet50' : { # 3
        'Size' : '98MB',
        'Speed' : '57m',
        'In_features' : 2048,
        'Accuracy' : 'High'
    }
}
SELECTED_MODEL = list(MODEL_DIC.keys())[1]
LEN_TRAIN = 4795
LEN_VAL = 1199
LEN_TEN = 5794

# 03 : Evaluation

# ---------------------------
# -- Directory managment : --
# ---------------------------

ROOT_DIR = Path.cwd().parent
OUT_DIR = Path("../")
DATA_DIR = "datasets/"
MODELS_DIR = "models/"
OUTPUTS_DIR = "outputs/"

DATA_FOLDER = [
    OUT_DIR / "datasets" / "processed" / "train" / "waterbird",
    OUT_DIR / "datasets" / "processed" / "train" / "landbird",
    OUT_DIR / "datasets" / "processed" / "val" / "waterbird",
    OUT_DIR / "datasets" / "processed" / "val" / "landbird",
    OUT_DIR / "datasets" / "processed" / "test" / "waterbird",
    OUT_DIR / "datasets" / "processed" / "test" / "landbird"
]

RAW_DATA_PATH = os.path.join(OUT_DIR, DATA_DIR, "raw/")
PROCESSED_DATA_PATH = os.path.join(OUT_DIR, DATA_DIR, "processed/")

METRICS_PATH = os.path.join(OUTPUTS_DIR, "classification_reports/")
CONF_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrices/")
CSV_PATH = os.path.join(OUTPUTS_DIR, "CSVs/")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "summary_plots/")
GRAD_CAM_PATH = os.path.join(OUTPUTS_DIR, "grad_cam/")
