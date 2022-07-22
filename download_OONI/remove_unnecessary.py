
import os
import datetime
import argparse
import pandas as pd
import glob
folder = "/data/censorship/OONI/"
def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date
dates = generate_dates('2021-06-20','2022-02-09')
for date in dates:
    for filename in glob.glob(folder+date+"/*/*/*.tar.gz"):
        cmd = "rm "+ filename
        os.system(cmd)
