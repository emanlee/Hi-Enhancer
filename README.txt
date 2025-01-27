Hi-Enhancer: a two-stage framework for prediction and localization of enhancers based on Blending-KAN and Stacking-Auto models
===========================================================


INSTALLATION
-------------
Hi-Enhancer is a two-stage framework for predicting and localizing genetic enhancers (enhancers). It accurately identifies the presence of enhancers and their boundaries by flexibly combining multiple epigenetic signals. The framework is particularly suitable for situations where not all epigenetic signals are available, providing an efficient and flexible solution.

Hi-Enhancer relies on the following environments and tools:
------------
+AutoGluon
+DNABERT-2
+Kolmogorov-Arnold Networks (KAN)

Configure the environment
# create and activate virtual python environment
$ conda create -n Hi python=3.10
$ conda activate Hi

$ pip install autogluon==1.0.0
$ pip install tensorflow
$ python3 -m pip install -r requirements.txt

# Installation of DNABERT-2：The modified DNABERT_2 feature extraction code is saved in the ./Stacking-Auto/DNABERT_2 folder. The original project address is:https://github.com/MAGICS-LAB/DNABERT_2

#Installation of KAN：The KAN network code is saved in the ./Blending-KAN/efficient_kan folder. The original project address is：https://github.com/Blealtan/efficient-kan.git

Usage
Phase 1: Enhancer Region Prediction
cd Blending-KAN

#Data Preparation:
This study utilized epigenetic signal data from HCT116 cells. In the code, we use the term "chip" to represent various signals, with the following correspondences:Chip1-H3k27ac
Chip2-H3k4me3
Chip3-H3k9ac
Chip4-H3K4me1
Chip5-DNAseq
The data can be obtained from https://sourceforge.net/projects/hi-enhancer/. After downloading, it should be saved to the current folder. Users can replace the data or select appropriate signal combinations according to actual requirements.
#运行Blending-KAN模型

#--The # --chips parameter is used so that the user can select the appropriate combination of signals to be used according to actual requirements. In the command, --chips can be followed by one or more signal names separated by spaces.

1. Train Base Classifier (Optional)
Users can also use their own data to train the model, requiring the data to be in the same format as the example samples.
This process takes a long time, so it is recommended to use the trained base classifiers directly.

Example: Take the combination of chip1 and chip3 as an example.
 (example samples are chip1_train_7-3.csv, chip3_train_7-3.csv)
$ python Layer1_signal.py --chips chip1 chip3

2.#Train the meta-classifier, if not run 1, it is recommended to use the trained base classifiers directly (our trained model is available from https://sourceforge.net/projects/hi-enhancer/, download it and put it in the current folder.)
(Example samples are chip1_test_7-3.csv, chip3_test_7-3.csv)
 Example: to use the combination of chip1 and chip3 as an example
$ python Layer2_signal.py --chips chip1 chip3

3.Use the trained model to predict new samples
 # To use chip1, the chip3 for example, the example samples are chip1_predict.csv, chip3_predict.csv, the results are saved in predictions.csv after the prediction.
$ python predict_signal.py --chips chip1 chip3




Phase 2:
$ cd Stacking-Auto

 Stacking-Auto model code (we recommend using our trained Stacking-Auto model, using a re-trained model requires modifying
 the model paths in the predict.py code)
1.Retrain the model code (optional)
#train.csv and test.csv after extracting features for the baseline dataset.
#Train the model code
$ python train.py
#Test the model on an independent test set code
$ python test.py

2.Directly use the trained model to realize the enhancer boundary localization (our trained model can be obtained from https://sourceforge.net/projects/hi-enhancer/, we recommend to use it directly)
#Extract DNA sequences：For the samples predicted to be positive classes in the first stage, extract their sequences using a sliding window to split the enhancer region into 200 bp sub-sequences and extract the features using DNABERT-2.

$python 200bp_feature.py 

#Predict and locate using Stacking-Auto model
$python predict.py
$python location.py

 Results
Blending-KAN model：
 99.69% accuracy when using five signals.
 Accuracy ≥ 93.72% in cross-cell lineage prediction.
 In Gaussian noise, the accuracy is still up to 98.74%.
Stacking-Auto model:
 80.50% accuracy, better than 17 existing methods.
==============
Dr. Aimin Li, Associate Professor
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, P.R China

Mr. Haotian Zhou, Master Student
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, P.R China

liaiminmail AT gmail.com
emanlee815 AT 163.com

2350837044@qq.com 

