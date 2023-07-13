# XGraphCDS

XGraphCDS: an explainable deep learning model for predicting drug sensitivity from gene pathways and chemical structures


## Overview

XGraphCDS integrates gene expression pathway information from cancer samples and chemical structural knowledge of drugs to predict the sensitivity of anticancer drugs in both _in vitro_ and _in vivo_ environments.

- `Data/` contains the necessary dataset files;
- `Models/` contains the several trained weights of XGraphCDS;
- `utils/` contains the necessary processing subroutines;
- `main_reg.py` function for XGraphCDS regression model (train or test);
- `main_class.py` function for XGraphCDS binary model (GDSC or TCGA, train or test);
- `visualize.ipynb` Example of XGraphCDS interpretable analysis;

## Requirements

- Please install the environment using anaconda3;  
  conda create -n XGraphCDS python=3.8.13
- Install the necessary packages.  
  conda install -c rdkit rdkit==2022.03.2  
  pip install fitlog   
  pip install torch (1.12.1)
  
  pip install torchaudio (0.12.1)
  
  pip install torchvision (0.13.1)
  
  pip install dgl-cuda11.3 (0.9.1)
  
  pip install dgllife (0.3.0)

## Usage

### Step1: Data Processing

You should download the source files from [here](https://drive.google.com/drive/folders/1Ztiq_yYrhfMXSSUrYrP_QuhrugfYLF8P?usp=sharing) and save all the files in the `Data/` directory.
The data needs to be uncompressed first: `cd ./XGraphCDS/Data`,
 and then enter the following commands: `tar -zxvf reg.tar.gz -C ./`, `tar -zxvf class.tar.gz -C ./`, `tar -zxvf tcga.tar.gz -C ./`

### Step2: Model Training/Testing

- You can run `python main_reg.py --mode train` to train XGraphCDS on regression task or run `python main_reg.py --mode test` to test trained model.


- You can run `python main_class.py --task gdsc --mode train` to train XGraphCDS on class task of GDSC data (in vitro) or run `python main_class.py --task gdsc --mode test` to test trained model.


- You can run `python main_class.py --task tcga --mode train` to train XGraphCDS on class task of TCGA data (in vivo) or run `python main_class.py --task tcga --mode test` to test trained model.

Notes: Each model has the option of requiring pre-training or not.

### Step3: Access to Explanations

we include our notebook **visualize.ipynb** to demonstrate how to visualize the explainations.


## Cite

If you use this code (or parts thereof), please use the following BibTeX entry:

` `