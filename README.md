
# DNS Censorship Detection with OONI

## Introduction
This prroject try to use ML to detect censorship. This part of the process focus on using the measurements data frrom [OONI](https://ooni.org/post/mining-ooni-data), a platform that measure censorship by relying on volunteers located at differrent ventage points to run the probes. The 2 countrries that we investigate censorship for this prrroject is China (CN) and United States (US) during the period from 2021-07-01 to 2022-02-09 (inclusive). 

The project is a pipeline that is divided into different stages, organized into different folders. In order to modify the code, please modify the code in each folder, especially under the "TO-DO" and run the files according to the orrder of the stages.

Otherwise, you can simply run the default settings that we use for the paper by running the Bash file "runAllFiles.sh"


## The Pipeline is organized into the following Stages:

### 1. Download OONI
(download_OONI folder): This program downloads the OONI datasets into your specified folder

a) download_OONI.py
This file downloads the OONI data of the country you want to specify for the time range you specify 

b) remove_unnecessary.py
This file just remove random stuff that is generated during downloading process





2. ### Program 2, folder: preprocess
This program converts the JSON files into dataframes, extract relevant features and convert them into a dataframe that is ready for ML.
#### extract2csv.py
This file takes the OONI data downloaded for the country and the time range you specify, extract the relevant features and convert them to a dataframe for each day
TODO:
- Specify the country code 
- Specify the date range that you want to use
#### add_GFWatchlabel.py
This file is used specifically for China(CN). For each probe, this file will check the domain name as well as the time when the probe is generated and check it against the GFWatch blocking rules to see whether it is censored according to the GFWatch. After that, it will add a column to the dataframe to report about this.
TODO:
- Specify the country code 
- Specify the date range that you want to use
- Specify the name of the files containing the probes and the name of the files where you want to store the new dataframe
#### preprocess_data.py
This file reformat each feature according to the right format, process them based on whether they are numerical or categorical.

3. ### Program 3, file: run_ML.py
What test are included in this file?
- In this file, you can run 3 kind of tests
- For the first test, you are using the whole set of data split them into train, validation and test set
- For the second test, you are trying to see whether using data from 1 month can predict censorship well for another month. The purpose of this is to check whether the freshness of the data for training will affect the performance of the censorship prediction
- For the third test, you are trying to see whether using data up and until certain dates can predict the censorship after that well. The purpose of this is to check whether the amount of the previous data used for training will affect the performance of the censorship prediction

Some important functions:
- run_supervised: this function is used to run supervised ML models. Note that this function will use a function, named get_accuracy to get tp,fp,tn,fn 
- run_unsupervised: this function is used to run unsupervised ML models. Note that this function will use another function, named get_accuracy_unsupervised to get tp,fp,tn,fn








