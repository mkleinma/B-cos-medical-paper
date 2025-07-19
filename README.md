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
   conda activate bcos-medical

------------------------

# Datasets
- Parts of the code require data from the pneumonia and multi-label datasets that are available on kaggle:
   - Pneumonia Dataset: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
   - Multi-Label Dataset: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
- Ensure to run the preprocessing Multi-Classification\training\preprocessing_and_splits (adjust paths accordingly) to obtain the 224x224 png files for the multi-label dataset. For the pneumonia dataset, all necessary preprocessing is provided within the training folder in the training splits and grouped_data.csv.
------------------------

# Structure of the Code Base
- Each dataset has it's respective directory 'Pneumonia' and 'Multi-Label Dataset'
- Both consist of corresponding folders: evaluation and training. Evaluation consists of all tools to derive quantitative and qualitative results while training consists the corresponding training scripts with the optimal hyperparameters to allow the reproduction of results. Preprocessing is also present in the training folder under   Multi-Classification\training\preprocessing_and_splits for the multi-label dataset
- After the preprocessing process, it is only necessary to change the corresponding paths of:
  1) the datasets and their corresponding data files 
  2) the models to evaluate
------------------------
# Datasets
- Parts of the code require data from the pneumonia and multi-label datasets that are available on kaggle:
   - Pneumonia Dataset: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
   - Multi-Label Dataset: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
- many parts of the thesis require adjustments for the directories as they are absolute or relative paths to e.g. the images in the respective datasets or where the models should be saved to. Simply adjust these paths to your directories and the code runs accordingly.
- LayerCAM and EPG files are showing the execution of primarily one model but is applicable on all models - to verify the results of other models simply change the directory to the model.
