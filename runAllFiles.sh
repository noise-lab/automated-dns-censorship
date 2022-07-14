#!/bin/bash

python CP_crawler.py

python create_raw_dataframes.py

python create_aggregate_dataframe.py # Need to change country name and country code, default is United States (US)

python add_GFWatch_column.py # For China data only

python get_as_count_for_each_ip.py

python aggregate_as_count.py

python convert_to_ml_sets_divide_dates.py # Need to change country name and country code, default is United States (US)

python ml_harness_timeseries_OCSVM_SGD.py # Execute this to obtain timeseries experiments on one parameter set for OCSVM_SGD

# python ml_harness_timeseries_XGBOOST.py # Execute this to obtain timeseries experiments on one parameter set for XGBOOST

# python ml_harness_OCSVM_SGD.py # Execute this to run parallel OCSVM_SGD experiments with different parameter sets

# python ml_harness_XGBOOST.py # Execute this to run parallel XGBOOST experiments with different parameter sets

# python ml_harness_OCSVM_OG.py # Execute this to run parallel OCSVM_OG experiments with different parameter sets

# python ml_harness_IFOREST.py # Execute this to run parallel IFOREST experiments with different parameter sets