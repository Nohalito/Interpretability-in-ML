# Interpretability-in-ML


## 1°/ Pre-requisite

Repository :

```
git clone https://github.com/Nohalito/Interpretability-in-ML.git

cd Interpretability-in-ML

code .
```

Environment setting on Window :

- Download Python 3.11

```
py -3.11 -m venv venv

cd venv/Scripts && . activate && cd ../..
```

Enjoy all of our notebook

## 2°/ Structure :

- Our most important outputs for the report understanding are located at `notebooks/03_Evaluation_Noa.ipynb` (Grad_CAM multi picture analysis) and at `outputs/confusion_matrices/ResNet18.png`

```
Interpretability-in-ML
├── .gitignore
├── README.md                                   # The will of D. ocumentation
├── config.py                                   # Global variable & path setting
├── repo_tree.ipynb
├── requirements.txt                            # Dependencies
├── datasets
│   ├── processed                               # Image folders
│   │   ├── test
│   │   │   ├── landbird
│   │   │   └── waterbird
│   │   ├── train
│   │   │   ├── landbird
│   │   │   └── waterbird
│   │   └── val
│   │       ├── landbird
│   │       └── waterbird
│   └── raw                                     # Raw parquet file
│       ├── test-00000-of-00001.parquet
│       ├── train-00000-of-00001.parquet
│       └── validation-00000-of-00001.parquet
├── models                                      # Our trained model
│   └── lr3e-5_ResNet18_cpu.pth
├── notebooks                                   # All notebooks used
│   ├── 01_Pre-processing_Noa.ipynb             # Pre-processing raw data to processed folder
│   ├── 02_Modeling_Noa.ipynb                   # Model training
│   └── 03_Evaluation.ipynb                     # Grad-CAM visualization
├── outputs                                     # some COMPLEMENTARY outputs
│   ├── CSVs
│   │   └── ResNet18.csv
│   ├── classification_reports
│   │   └── ResNet18.json
│   ├── confusion_matrices
│   ├── grad_cam
│   └── summary_plots
└── src                                         # Custom libraries
    ├── grad_cam.py
    ├── networks.py
    ├── plot_utils.py
    └── utils.py
```

## References :

- Database : <a href = "https://huggingface.co/datasets/grodino/waterbirds">Hugging face</a>
- Code architecture and Grad cam inspiration : <a href = "https://github.com/priyavrat-misra/xrays-and-gradcam?tab=readme-ov-file">Priyavrat Misra</a>
