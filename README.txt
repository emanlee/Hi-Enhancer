Hi-Enhancer: a two-stage framework for prediction and localization of enhancers based on Blending-KAN and Stacking-Auto models
===========================================================


INSTALLATION
-------------
https://www.youtube.com/playlist?list=PLWQiu8dLtUx4kmp3XkNk0SUU4cIjkiOdD


Hi-Enhancer relies on the following environments and tools:
------------
+ Linux/Mac OS
+ Anaconda3 


Configure the environment
# create and activate virtual python environment [User can use anaconda3 or other tools to create a virutual environment. Here we use anaconda3. ]
$ conda create -n hienv python=3.10
$ conda activate hienv

$ pip install autogluon==1.0.0    [This step is very slow]
$ pip install tensorflow  

# download files from https://github.com/emanlee/Hi-Enhancer
# and put them into /home/aimin/hi-enhancer/

$ python3 -m pip install -r requirements.txt

# [Skip this step] Installation of DNABERT-2：The modified DNABERT_2 feature extraction code is saved in the ./Stacking-Auto/DNABERT_2 folder. The original project address is https://github.com/MAGICS-LAB/DNABERT_2

# [Skip this step] Installation of KAN：The KAN network code is saved in the ./Blending-KAN/efficient_kan folder. The original project address is https://github.com/Blealtan/efficient-kan

Usage
Phase 1: Enhancer Region Prediction
cd Blending-KAN

#Data Preparation:
This study utilized epigenetic signal data from HCT116 cells. In the code, we use the term "chip" to represent various signals, with the following correspondences:
Chip1-H3k27ac
Chip2-H3k4me3
Chip3-H3k9ac
Chip4-H3K4me1
Chip5-DNAseq
The data can be obtained from https://sourceforge.net/projects/hi-enhancer/files/Blending_kan_data/. After downloading, it should be saved to the current folder. Users can replace the data or select appropriate signal combinations according to actual requirements.


#
#--The --chips parameter is used so that the user can select the appropriate combination of signals to be used according to actual requirements. In the command, --chips can be followed by one or more signal names separated by spaces.

1. Train Base Classifier (Optional)
Users can also use their own data to train the model, requiring the data to be in the same format as the example samples.
This process takes a long time, so it is recommended to use the trained base classifiers directly.

Example: Take the combination of chip1 and chip3 as an example. [about 1.2 hours]
 (example samples are chip1_train_7-3.csv, chip3_train_7-3.csv)
$ python Layer1_signal.py --chips chip1 chip3

2.#Train the meta-classifier, if not run step 1, it is recommended to use the trained base classifiers directly (our trained model is available from https://sourceforge.net/projects/hi-enhancer/files/phase1_models/, download it, unzip, and put it in the current folder.)
(Example samples are chip1_test_7-3.csv, chip3_test_7-3.csv)
 Example: to use the combination of chip1 and chip3 as an example
$ python Layer2_signal.py --chips chip1 chip3

3.Use the trained model to predict new samples
 # To use chip1, the chip3 for example, the example samples are chip1_predict.csv, chip3_predict.csv, the results are saved in predictions.csv after the prediction.
$ python predict_signal.py --chips chip1 chip3

$ more predictions.csv  
Sample_ID,Predicted_Class,Predicted_Probability
0,1,0.929342
1,1,1.2520199
2,1,0.929342
3,1,0.929342
4,1,0.929342
5,0,0.24864486
6,0,-0.9822619
7,1,1.4764237
8,1,1.4764237
9,1,0.929342

1  enhancer,  0  non-enhancer, 

Phase 2:
$ cd Stacking-Auto

 Stacking-Auto model code (we recommend using our trained Stacking-Auto model, using a re-trained model requires modifying
 the model paths in the predict.py code)
1.Retrain the model code (optional)
#train.csv and test.csv after extracting features for the baseline dataset.
#Train the model code [about 1.2 hours]
$ python train.py
#Test the model on an independent test set code
$ python test.py

2.Directly use the trained model to realize the enhancer boundary localization (our trained model can be obtained from https://sourceforge.net/projects/hi-enhancer/files/phase2_models/output_folder.zip, we recommend to use it directly, download and unzip it)

#Extract DNA sequences：For the samples predicted to be positive classes in the first stage, extract their sequences using a sliding window to split the enhancer region into 200 bp sub-sequences and extract the features using DNABERT-2.

Download zhihan1996.zip from https://sourceforge.net/projects/hi-enhancer/files/phase2_models/zhihan1996.zip, put it into the Blending-KAN folder, unzip it. 

$python 200bp_feature.py 

#Predict and locate using Stacking-Auto model
$python predict.py

$python location.py

  
==============
Dr. Aimin Li, Associate Professor
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, China

liaiminmail AT gmail.com
aimin.li AT xaut.edu.com

Mr. Haotian Zhou, Master Student
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, China

2350837044@qq.com 

