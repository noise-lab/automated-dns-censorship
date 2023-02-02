import os
import datetime
import argparse
import pandas as pd
####### install this oonidata package #######
os.system("pip install oonidata")
############    TODO    ########
start_date = "2021-02-23" # modify the start date
end_date = "2022-02-24" # modify the end date
country = "CN" # modify the country
folder = "CN"  # modify the folde you want to store the data
################################


cmd = "oonidata sync --start-day {0} --end-day {1} --probe-cc {2} --test-name web_connectivity --output-dir {3}".format(start_date, end_date,country,folder)
os.system(cmd)
