# DNS Censorship Detection with Censored Planet

## Introduction

The data used in this project is taken from [Censored Planet Satellite](https://data.censoredplanet.org).
We ingested probes ranging from 2021-06-30 to 2022-02-09 (inclusive). The data format corresponds to 
[Satellite-v2.2](https://docs.censoredplanet.org/dns.html#satellite-v2-2) which has since been deprecated.
However, the original data is still available online from the [raw data](https://data.censoredplanet.org/raw) page.

## How to Run

In each of the programs below, find the "TODOs", and change the appropriate variables as required. Then run the Bash
file "runAllFiles.sh". Programs 1, 2, 3, and 4 need to be run in sequence, as do items 5 and 6 (both sequences can be
run in parallel). Program 7 cannot run until programs 1 through 6 have completed. Programs 8 through 13 train and
evaluate different machine learning models, and thus can be run in any order or concurrently. 

Note also that programs 3, 7, and 8 process only one country at a time. These programs will need to be
re-run with different countries (including the US and China) in order to execute the full suite of tests in 
programs 10 through 13.

Because of the significant length and memory use of these programs, we recommend using a
20+ CPU machine with approximately 40 GB of RAM. No GPUs are required.

## Code Description

1. ### cp_crawler.py

This program downloads the Censored Planet datasets into your preferred folder.

2. ### create_raw_dataframes.py
This program converts the JSON files into dataframes (no information is discarded). Note that each original JSON file
divided into approximately 180 dataframes in order to reduce memory use.

3. ### create_aggregate_dataframe.py
This program aggregates the dataframes from the previous step into one dataframe. The program removes irrelevant columns 
and only keeps probes taken from the vantage points listed in Vantage_Point_CSV_List_V2.csv 
(see Vantage Point Selection). Note that this program only aggregates the dataframes for one country at a time
(default US), so the country name and country code must be changed in order to create aggregate dataframes for 
other countries.

4. ### add_GFWatch_column.py

This program appends an additional column to the China aggregate dataframe that serves as the ground truth label set for
model training and inference. These labels are taken from the GFWatch censorship monitoring service and are
stored in the file gfwatch_censored_domains.csv (the file requires significant pre-processing in order for it to be
matched with the Censored Planet dataset). Note that invalid records are removed in this step.

5. ### get_as_count_for_each_ip.py

This program outputs a table containing the most common autonomous system number (ASN) for each domain query at each
unique probe date. Note that
the input set for this program consists of all probes in the CP_Downloads folder, including those from other countries.
This is necessary because we will assume that the most common ASN globally is the "true" ASN of the host for that
domain; as a result, 
we need a large sample set that consists of probes taken from countries we know to be free of censorship.

6. ### aggregate_as_count.py

This program merges all the probe ASN max count daily datasets into one table. It also labels a domain-ASN record as
"clean" (usable) or "dirty" (not usable), depending on whether it means certain criteria (i.e. > 50%
of ASNs are the same for a given domain). See "Filtering and Labelling" for more detail.

7. ### convert_to_ml_sets_divide_dates.py

This program converts the aggregate data frames into an ML-ready format 
(the core methods are located in convert_to_ml_helper_methods.py). It then divides the dataset into unique training,
validation, and testing sets, and then further subdivides them into clean (sanitized) and unclean (mixed) sets. Each
month in the test period (7 in total) also has its own unique dataset.
In the process, the program removes invalid records (except for China records, whose invalid records were removed in 
_addGFWatch_column.py_). Note that this program only creates datasets for one country at a time
(default US), so the country name and country code must be changed in order to create datasets for 
other countries.

8. ### ml_harness_timeseries_OCSVM_SGD.py

This program trains, validates, and evaluates the One Class Support Vector Machine Single Gradient Descent
(OCSVM_SGD) model on the provided datasets. The program trains and tests the models on all possible combinations
of consecutive months, providing insight into the performance of machine learning models trained with data obtained
from different time points. It also outputs the ROC and Precision-Recall curves.

9. ### ml_harness_timeseries_XGBOOST.py
Same functionality as above, except for the XGBOOST model.

10. ### ml_harness_OCSVM_SGD.py
This program trains, validates, and evaluates the One Class Support Vector Machine Single Gradient Descent
(OCSVM_SGD) model on the provided datasets. Unlike the timeseries program, this script is meant for optimizing the
parameter set. This is accomplished by allowing multiple models to be trained and evaluated concurrently.

11. ### ml_harness_XGBOOST.py
Same functionality as above, except for the XGBOOST model.

12. ### ml_harness_OCSVM_OG.py
Same functionality as above, except for the vanilla One Class Support Vector Machine model.

13. ### ml_harness_IFOREST.py
Same functionality as above, except for the Isolation Forest model.

## Vantage Point (VP) Selection

Since Censored Planet takes measurements from 200+ vantage points in many countries, we needed to narrow this selection
to 15 VPs in order to make the dataset size manageable. We chose 15 VPs that were most representative of network traffic
in each country, with the number of VPs for each ISP ASN taken in rough proportion
to the proportion of the population served by that ISP. For some countries, such as Iran, VPs from major ISPs are
inaccessible to Censored Planet, so in this case we were forced to use VPs belonging to smaller ASNs.

## Filtering and Labelling

Many Censored Planet probes contain measurement errors that could mislead our ML models. As a result, we must remove
"invalid" probes from our dataset. These probes include those that failed certain control tests
or could not be replicated consistently.

In order to train certain unsupervised models, we also need to sanitize the dataset by removing all probes that _could_ 
be censored, not just those we definitively know to be censored. For example, all returned IPs whose ASN does not match
the majority ASN for that domain-date combination are deemed "unclean". Members of the "clean" set can serve as ground
truth negative data for training an anomaly detection model, while "unclean" data can be used for testing.

## Feature Selection

The file "Dataset_Feature_Description" describes the dataset used for
the machine learning models. The features are roughly analagous to the features present in the 
[Censored Planet Satellite-v2.2 dataset](https://docs.censoredplanet.org/dns.html#satellite-v2-2).

