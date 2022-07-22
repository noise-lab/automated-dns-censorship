import os
import datetime
import argparse
import pandas as pd

def generate_hours():
    return_list = []
    for i in range(24):
        str_num = str(i)
        if len(str_num)==1:
            return_list.append("0"+str_num)
        else:
            return_list.append(str_num)
    return return_list
def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date

hours = generate_hours()
countries = ['TM']
dates = generate_dates('2017-02-11','2021-06-19')


os.chdir('/data/censorship/OONI/T/')
for date in dates:
    if not os.path.isdir("{}/raw_data/{}".format(os.getcwd(), date)):
        cmd = "mkdir {0};".format(date.replace('-',''))
        os.system(cmd)

        print(">>>> crawling data of date: {}".format(date))
    for hour in hours:
        for country in countries:

            cmd = " /home/tranv/.local/bin/aws --no-sign-request s3 sync s3://ooni-data-eu-fra/raw/{0}/{2}/{3}/webconnectivity/ ./{1}/{2}/{3}".format(date.replace('-',''), date,hour,country)
            os.system(cmd)
