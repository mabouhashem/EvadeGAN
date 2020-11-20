# EvadeGAN #
EvadeGAN is a GAN-based framework that can be trained to generate adversarial examples against ML models.

EvadeGAN has been developed to target malware classifiers with binary feature space. The model used as a test case is an SVM classifier trained on the DREBIN dataset. 

EvadeGAN was developed as part of a Master's project at King's Department of Informatics, titled:

```
=========================================================================================================
│ "Using Generative Adversarial Networks to Create Evasive Feature Vectors for Malware Classification"  │ 
│                                                                                                       │
│ By: Mohamed Abouhashem                        ||            Supervisor: Professor Lorenzo Cavallaro   │
=========================================================================================================
```

**Thesis:** https://github.com/mabouhashem/EvadeGAN/blob/master/Thesis.pdf

**Five-minute presentation:** https://youtu.be/adf4uOlnMt8

## EvadeGAN Architecture

![EvadeGAN_Architecture](https://xwrzea.am.files.1drv.com/y4mUzS0T-RYduypuSQKfx1fN-lERRjoZMV9de4OTGrMCJxoamJ0DXIYOZ1ecaQ_SjeCnva3dX5SuORmNoIPVCoZHw9H4-0PejGfPkvj0VtXPesah44gnmO9zHgwrMFBaTkQHz0NK8oEKBtWSvcDJPpfPKUFJyiGXbUJXmyqfM95TaPoyb9pol41NjimVMUVaiyIZ5aIkyuQHh5g9TWFHB7wVg/EvadeGAN_Architecture.png)
  
### D Loss:
![DLoss](https://latex.codecogs.com/svg.latex?%5Clarge%20%5Cmathcal%7BL%7D_%7BD%7D%20%3D%20-%20%5Cmathbb%7BE%7D_%7Bx%7Cf%28x%29%3D1%7D%5B%5Clog%20D%28x%29%5D%20-%20%5Cmathbb%7BE%7D_%7Bx%7Cf%28x%29%3D0%7D%5B%5Clog%20%281-D%28x%29%29%5D)  
  
### G Loss:
![GLoss](https://latex.codecogs.com/svg.latex?%5Cdpi%7B300%7D%20%5Clarge%20%5Cmathcal%7BL%7D_%7BG%7D%20%3D%20%5Cunderbrace%7B%5Cmathbb%7BE%7D_%7Bx%5E%5Cprime%7Cx%5E%5Cprime%20%3D%20x&plus;%5Cdelta%7D%5B%5Clog%20D%28x%5E%5Cprime%29%5D%7D_%5Ctext%7B%5Cemph%7BEvasion%20loss%7D%7D%20&plus;%20%5Cunderbrace%7B%5Calpha%20%5C%7C%20%5Cdelta%20%5C%7C_1%7D_%5Ctext%7B%5Cemph%7BInduce%20sparsity%7D%7D%20&plus;%20%5Cunderbrace%7B%5Cbeta%20%5Cmax%20%280%2C%20%5C%7Cx%5E%5Cprime%20-%20x%5C%7C_1%20-%20K%29%7D_%5Ctext%7B%5Cemph%7BEnforce%20an%20upper%20bound%7D%7D)    
  
## EvadeGAN Modes
EvadeGAN can operate in three different modes (based on the inputs to the generator) to generate either:\
**A) Sample-Dependent Perturbations** (in case of **EvadeGANx** and **EvadeGANxz**), OR  
**B) Sample-Independent (Universal)** Perturbations (in case of **EvadeGANz**)  
  
**The input-output configuration of the generator in each mode** is shown in this figure.  

![EvadeGAN_Modes](https://pnhxsw.am.files.1drv.com/y4mZfqQ-GOUQivMTvSqrbiO34e--2yam_Hkwr6diDyjQWig2yKhezwxlqT_NXy-DIKG8hOT9M2rEjrh9aqis4zxdGkU9MftWovw2sPEN2MsGkq6lJATQ9B839lz558KwNAiINNgzTQ_99ZCQsIXgnRMGTOc8aOgjHTTJAqZbmuU1MNW6AJg6SVr1xfS0fvCI7ohKCE7zG2aSixTb5Tmo6taIw/GeneratorModes.png)

A **sample run** of training and evaluating each mode (**EvadeGANx**, **EvadeGANxz**, and **EvadeGANz**) is provided as a **Jupyter notebook** as shown in the repo structure below. A separate notebook demonstrates several aspects about the used dataset and the target classifier.  

**Note**: If there is an issue with viewing the notebooks on Github, you could view them through these links:  
**EvadeGANx:** https://nbviewer.jupyter.org/github/mabouhashem/EvadeGAN/blob/master/test_EvadeGANx.ipynb  
**EvadeGANxz:** https://nbviewer.jupyter.org/github/mabouhashem/EvadeGAN/blob/master/test_EvadeGANxz.ipynb  
**EvadeGANz:** https://nbviewer.jupyter.org/github/mabouhashem/EvadeGAN/blob/master/test_EvadeGANz.ipynb  

### A Peak into EvadeGAN Learning: ###

![EvadeGAN_Learning](https://xwspyq.am.files.1drv.com/y4mtI75AknfHCJuh0iBTiL8MXSRC93xD36Y0yjKrrw7qPYcgK7esC2OiswCQpRzLqGecxpcbS5CB0RNJOgocB0x-2u9AaTWtHc1jErxUVaFIu609ArWXkvdHAg9DokzptCK49SzIY35EvhH9whLU6I80L1_uTsUOoI64_pzt3UBnQXqqueHKY7N43v0_LkqYFh_q5ZJ7PS4AhlazjDGh922mw/EvadeGAN_Learning.png)  

### Performance of EvadeGAN during 100 epochs of training: ###

![EvadeGAN_Learning](https://xwtyaw.am.files.1drv.com/y4mPD2KdEY-30D1CXTqQCtolwAkHtNxJzMZd1eggoU6XeGpCxJkcXcz9GySFAgMhIV4zy-FejUnb1hcgFDKSBIPu2jUuz5kMPjkjqNokWNXUUtkg8Ot9WFRdoS1tT40MbOiiMW7ubZHuCK9J2wAC6f5DT5egW-m6aO5HGrjLjLWYuPXdII3J-9tqeUpccayyen0K8_MBg_r6mTIlUHuLaOycg/EvadeGANxz_Training_100_Epochs.png)  

 

## Repo Structure. ##
This repository is structured as follows:
```
├── data/       # A directory for all used & generated data
│   ├── dataset/            # A directory for the original dataset (json) or pre-pickled shelves.
│   ├── GAN/                # A directory for the weights & models of EvadeGAN, with subdirectories for each mode.
│   ├── models/             # A directory for trained SVM classifiers (target models)
│   └── plots/              # A directory for plots
│   
├── src/        # A directory for all source code files
│   ├── attack.py                   # The main attack module, where the EvadeGAN class and other utility functions are defined. 
│   ├── classifier.py               # This module defines functions for creating, training, and evaluating the target SVM classifier. 
│   ├── data.py                     # This module defines the Data class which handles the dataset (reading, shelving, splitting, and feature selection).
│   ├── features.py                 # This module defines functions for feature analysis.
│   ├── globals.py                  # This module defines a few global variables & directories.
│   └── utilities.py                # A module with utility functions
│   
├── test_Dataset.ipynb      # A notebook to demonstrate reading the dataset, training & evaluating the classifier, and performing basic feature analysis.
├── test_EvadeGANx.ipynb    # A notebook to demonstrate the training and evaluation of the EvadeGANx mode
├── test_EvadeGANxz.ipynb   # A notebook to demonstrate the training and evaluation of the EvadeGANxz mode
├── test_EvadeGANz.ipynb    # A notebook to demonstrate the training and evaluation of the EvadeGANz mode
│
├── Thesis.pdf 
│   
└── README.md   # You are here
```  
  

## Running the code ##
For convenience, the code to run the different parts of the project is included in the Jupyter notebooks listed above.  
Support for command-line arguments to be added soon.

**Note:**  
Provided with the code are the following:  
1. A preprocessed shelf of the dataset (./data/dataset/).  
In case this is not available, the code will try to read the original json files of the dataset from the same directory (which are not included due to their large size).  
  
2. The trained classifier that was used as a target model in the experiments (./data/models/).  
In case this is not available, the code will train a new classifier based on the given hyperparameters, and save the trained classifier in the above directory.   
   
Other directories are there for the outputs of running the code.  

     
## Dependencies ##
The following are the main dependencies for the code to work:
```
python 3.6.9
scikit-learn 0.23.1
keras 2.4.3
tensorflow 2.2.0
numpy 1.18.5
scipy 1.4.1
pandas 1.0.5
matplotlib 3.2.2
seaborn 0.10.1
joblib 0.16.0
json 2.0.9
```