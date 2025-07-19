# Faithful, Interpretable Chest X-ray Diagnosis with Anti-Aliased B-cos Networks

This is the GitHub for the corresponding master thesis surrounding B-cos networks.g
It includes two subdirectories for the Pneumonia and Multi-Label Dataset that include all necessary information for reproduction in training scripts.

-----

# Environment Setup
To set up the necessary environment for this thesis, follow these steps:
1. Install Anaconda or Miniconda
2. Download this repository and open a terminal in the project folder for creating the corresponding environment
3. Run the following commands in the .cmd line:
   conda env create -f environment.yml
   conda activate thesis-medical

------------------------

# Technical Details for using the Code Base
- Parts of the code require data from the pneumonia and multi-label datasets that are available on kaggle:
   - Pneumonia Dataset: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
   - Multi-Label Dataset: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
- many parts of the thesis require adjustments for the directories as they are absolute or relative paths to e.g. the images in the respective datasets or where the models should be saved to. Simply adjust these paths to your directories and the code runs accordingly.
- LayerCAM and EPG files are showing the execution of primarily one model but is applicable on all models - to verify the results of other models simply change the directory to the model.
