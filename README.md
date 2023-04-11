
# DNS Censorship Detection with OONI

## Dataset
The preprocessed data for OONI is publicly available at [this link](https://drive.google.com/drive/folders/1ebmj1liu2i1bBxFh59vAdtTSsq1c7kzZ?usp=sharing)


## Introduction
This prroject try to use ML to detect censorship. This part of the process focus on using the measurements data from [OONI](https://ooni.org/post/mining-ooni-data), a platform that measure censorship by relying on volunteers located at differrent ventage points to run the probes. The 2 countrries that we investigate censorship for this prrroject is China (CN) and United States (US) during the period from 2021-07-01 to 2022-02-09 (inclusive). 

The project is a pipeline that is divided into different stages, organized into different folders. In order to modify the code, please modify the code in each folder, especially under the "TO-DO" and run the files according to the orrder of the stages.

Otherwise, you can simply run the default settings that we use for the paper by running the Bash file "runAllFiles.sh"


## The Pipeline is organized into the following Stages:

### 1. Download OONI
(download_OONI folder): This program downloads the OONI datasets into your specified folder

a) download_OONI.py - 
This file downloads the OONI data of the country you want to specify for the time range you specify 

b) remove_unnecessary.py - 
This file just remove random stuff that is generated during downloading process





### 2. Preprocess and add GFWatch label
(preprocess folder)

a) extract2csv.py - 
This file converts the JSON files into dataframes, extract relevant features and convert them into a dataframe that is ready for ML.

b) add_GFWatchlabel.py - 
This file is used specifically for China(CN). For each probe, this file will check the domain name as well as the time when the probe is generated and check it against the GFWatch blocking rules to see whether it is censored according to the GFWatch. After that, it will add a column to the dataframe to report about this.

c) preprocess_data.py - 
This file reformat, encode each feature according to the right format, process them based on whether they are numerical or categorical.

### 3. Splitting the data into training, validation and testing set
(split_data folder)


### 4. Get the best model for each classifier and training and validating scenario
(get_best_model folder)

a) get_best_model.py -
This file is used to get and store the best model for each combination of classifier, training and testing scenario. Please modify the name of classifier, training and testing scenario  and seed where the data was generated to get the desired best performing model. ```Note that the script supports grid parameter search to yield the optimal model, users are encouraged to update the parameter search setting accordingly in the script as they see fit```.

### 5. Get output from the stored model
a) get_output.py - 
This file is used to load the stored best performing model, and then run the model on the test data and save the prediction

### 6. Analyse the output obtained and analyze the feature importance of the model
a) analyze_results - 
This file allow you to specify the filename of the file you want to analyze how good the prediction is. It will automaticallyy produce statistics that report how well the predictions are
b) get_feature_importance.py - 
This file allow you to specify the name of the model you want to analyze, and automatically process the output, aggregate accordingly to generate a dataframe of features and their importance









