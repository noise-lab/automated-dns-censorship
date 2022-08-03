#!/bin/bash

## program 1

python download_OONI.py

python remove_unnecessary.py

## program 2

python extract2csv.py

python add_GFWatchlabel.py

python preprocess_data.py

## program 3
run_ML.py
