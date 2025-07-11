Hi-Enhancer: a two-stage framework for prediction and localization of enhancers based on Blending-KAN and Stacking-Auto models
===========================================================


INSTALLATION
-------------
https://www.youtube.com/playlist?list=PLWQiu8dLtUx4kmp3XkNk0SUU4cIjkiOdD
[Without voice]

Hi-Enhancer relies on the following environments and tools:
------------
+ Linux/Mac OS
+ Anaconda3 (or Miniconda3 for lighter resource requirements) 
Please note that while Hi-Enhancer has been tested with Anaconda3, it is also compatible with Miniconda3 or other virtual environment tools. Users can choose the most suitable environment manager based on their system resources and preferences.

GPU usage instructions
To take full advantage of the Hi-Enhancer framework for enhancer prediction and localization, we recommend using a GPU to accelerate the model training and testing process. In our tests, the complete training of the model took more than ten hours using an NVIDIA 1080 GPU. This long training process can put a strain on computational resources, so we strongly recommend using a GPU for model training.
In addition, even if you choose to use our pre-trained models directly, a GPU is recommended to ensure faster processing speeds and greater efficiency in model inference and application.

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


===========================================================
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


If you use your own data, you should use the following code to extract ChIP-seq/DNase-seq signal features from BigWig files.
$python extract_signalvalues_by_reions.py \
    -b path/to/chipseq.bigWig \
    -r path/to/regions.bed \
    -o path/to/output_features.csv \
    -w 4000 \
    -n 400

Example（Take DNase-seq for extraction of HCT116 as an example）:
$python extract_signalvalues_by_reions.py     -b HCT116.DNase-seq.bigWig     -r HCT116.positive.bed     -o output_features.csv     -w 4000     -n 400


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
#When users use their own data for prediction, they need to aggregate the selected signals by each 10 bp (base pair) interval and calculate the average value of each interval. After the aggregation process, the length of the signal data for each sample should be 400. After that, each signal data is converted to a CSV file separately and column headings are added for each column. The file naming format is chipX_predict.csv, where X is the signal number (as described in the #Data Preparation section). Specific file formats can be found in the example samples chip1_predict.csv and chip3_predict.csv.
#The user is required to also input the sample's bed file (i.e., the bed file used in the Data Preparation section) to contain the sample's chromosome information, start position, and end position.
# To use chip1, the chip3 for example, the example samples are chip1_predict.csv, chip3_predict.csv, the results are saved in predictions_with_probabilities.csv after the prediction.
$python predict_signal.py --chips chip1,chip3 --bed_file path/to/your_bed_file.bed

# Predicted results contain chromosome information, start position and end position.The second column in predictions_with_probabilities.csv: 1 means enhancer,  0 means non-enhancer.
$ head predictions_with_probabilities.csv  
Sample_ID	Predicted_Class	Probability_Class_0	Probability_Class_1	Chromosome	Start_Position	End_Position
0	1	0.40294233	0.5970577	chr1	1248320	1252321
1	1	0.12454599	0.875454	chr1	1503611	1507612
2	1	0.25905165	0.7409483	chr1	2178568	2182569
3	1	0.40294233	0.5970577	chr1	2546909	2550910
4	1	0.40294233	0.5970577	chr1	5884020	5888021
5	0	0.96190983	0.03809011	chr1	5915842	5919843
6	0	0.97383547	0.02616448	chr1	6197802	6201803
7	1	0.12684299	0.873157	chr1	6599207	6603208
8	1	0.12684299	0.873157	chr1	8196013	8200014

The second column: 1  means enhancer,  0 means non-enhancer


4.Extract the bed file predicted to be a positive class and extract its sequences.
#Users should select the genome reference file according to their actual situation and download it to the current folder.
#Let's take hg38.fa as an example, assuming that hg38.fa has already been downloaded.
$python extract_DNA_sequences.py \
    --input_csv predictions_with_probabilities.csv \
    --output_bed positive_samples.bed \
    --output_fasta positive_samples.fasta \
    --genome_fasta hg38.fa\
    --output_csv output_sequences.csv
#output_sequences.csv is the extracted enhancer region, which can be renamed and used directly in Phase 2 to localize enhancers.


===========================================================
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

  
===========================================================
Dr. Aimin Li, Associate Professor
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, China

liaiminmail AT gmail.com
aimin.li AT xaut.edu.cn


Mr. Haotian Zhou, Master Student
School of Computer Science and Engineering,
Xi'an University of Technology,
5 South Jinhua Road,
Xi'an, Shaanxi 710048, China

2350837044 AT qq.com 

# updated on June 7, 2025