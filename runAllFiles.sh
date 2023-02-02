#!/bin/bash

## stage 1: download OONI

python ./download_OONI/download_OONI.py #do that for both CN and US


## stage 2: preprocess data

python ./Preprocess/extract2csv.py #do that for both CN and US

python ./Preprocess/add_GFWatchlabel.py #do that for CN only

python ./Prerprocess/preprocess_data.py

## stage 3: split data
python ./split_data/split_data.py

## stage 4: get best model
python ./get_best_model/get_best_model.py

## stage 5: get output
python ./get_ouput/get_output.py

## stage6: analyze results
python ./analyze_results/analyze_results.py
python ./analyze_results/get_feature_importance.py
