# The DL mammography project

This repository stores the code used for the article "Deep learning predicts
interval and screen-detected cancer from screening mammograms: a
case-case-control study".

## Prerequisites

 * Linux Kernel 4.4+
 * Python 3.6+, with recent versions of the following python packages:
    - cv2
    - fire
    - matplotlib
    - numpy
    - pandas
    - pydicom
    - scipy
    - seaborn
    - pytorch
    
## Running

The pipeline consists of three steps, which are separated into three folders.

    ./s1-data_preprocessing
    ./s2-cnn_model
    ./s3-GAN

Inside each folder, there is a Python script which serves as the entry point.
These are:

    ./s1-data_preprocessing/preprocess.py
    ./s2-cnn_model/run.py
    ./s3-GAN/run.py
    
Running the entry point script without any command line arguments will prompt a
list of required/optional arguments. These script assumes that raw files were
put into the relative path of `./raw_data` and their outputs are put into the relative
path of `./out_data`, both are relative to the entry point's directory.
