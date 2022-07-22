import json
import csv
import pandas as pd
import glob
import os
import gzip
import re
import math
import multiprocessing as mp




##### Functions ####
def process_date(time):
    converted_datetime = pd.to_datetime(arg=time, format='%Y/%m/%d')
    return converted_datetime

def translate_to_regex(rule):
    return_str = rule
    if return_str[0]=="*":
        return_str = ".*"+return_str[1:]
        
    if return_str[-1]=="*":
        return_str = return_str[:-1]+".*"
    return return_str

def match_rule(domain, rule):
    
    if re.match(rule,domain):
        return True
    return False

def find_match_rule(domain):
    for dic_key in dic_censored.keys():
        if dic_key in domain:
            return dic_key
    return ""

def check_domain_block(domain):
    return domain in censored_domains.keys()

def check_range_time(start_time, range_times):
    for i in range(len(range_times)):
        range_time = range_times[i]
        start = process_date(range_time[0])
        end = process_date(range_time[1])
        status = (start_time.date()>= start) and (start_time.date() <= end)
        if status:
            return "Confirmed"
    return "Possible"
    

def label_gwatch(df):
#     censored = list(gfwatch["base_censored_domain"])
    
    for index, row in df.iterrows():
        time_ranges=[]
        domain = row["input"]
        if check_domain_block(domain):
            time_ranges = censored_domains[domain]
        else:
            matched_domain = find_match_rule(domain)
            if matched_domain !="":
                blocking_rule = dic_censored[matched_domain]["blocking rule"]
                
                if match_rule(domain, blocking_rule):
                    time_ranges= dic_censored[matched_domain]["blocking time"]
        if len(time_ranges) >0:
                row["dns_blocking_truth"] = check_range_time(row["measurement_start_time"],time_ranges)
                print("censored !")
            
        else:
                print(" Not matched pattern!")
                print(domain)
                row["dns_blocking_truth"]=None
    return df

def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date
def process_time(time):
    converted_datetime = pd.to_datetime(arg=time, format='%Y-%m-%d %H:%M:%S')

    return converted_datetime

def process_asn(asn):
    return asn[2:]
def get_domainname(name):
    new_name = name.split("//")[-1]
#     if new_name[:4]=="www.":
#         new_name = new_name[4:]
    if new_name[-1]=="/":
        new_name = new_name[:-1]
    return new_name
# def match_block_signature(ip):
#     blocked = False
#     block_signature = [ "8.7.198.45","37.61.54.158", "46.82.174.68","78.16.49.15","93.46.8.89","159.106.121.75","203.98.7.65", "59.24.3.173","203.98.7.65", "243.185.187.39"]

#     ip_ = ip.split(":")[-1]
# #     print(type(ip_))
# #     print(ip_)
#     if ip.split(":")[-1] in block_signature:
#         blocked = True
#     return blocked

# def label_groundtruth(df):
    
# #     df['dns_blocking_truth'] = [match_block_signature(x) for x in df["test_keys_ipv4"]]
# #     block_level = []
#     for i in df['dns_blocking_truth']:
#         if i:
#             block_level.append("Country")
#         else:
#             block_level.append("")
#     df["Block_level"] = block_level     
#     return df

def process_dataframe(df):
    print("starting")
    df["input"] = [get_domainname(name) for name in df["input"]]
    for col in df.columns:
        if col=="probe_asn" or col=="resolver_asn":
            df[col]=process_asn(df[col])
        elif col == "measurement_start_time" or col == "test_start_time":
            df[col] = process_time(df[col])
#     df = label_groundtruth(df)

    df = label_gwatch(df)
    return df

def split_df(df, splits):
    split_size = math.ceil(df.shape[0]/splits)
    df_list = []
    for i in range(splits):
        if i!= splits-1:
            df_ = df.iloc[list(range(i*split_size,(i+1)*split_size))]
            df_list.append(df_)
        else:
            df_ = df.iloc[list(range(i*split_size,df.shape[0]))]
            df_list.append(df_)
            
    return df_list



#### loading GFWatch rule ####
filename = "/data/censorship/gfwatch_censored_domains.csv"
df = pd.read_csv(filename)
list_ = []
index = df.columns.tolist()
column_names = index[0].split("|")
for i in df[df.columns[0]]:
    list_.append(i.split("|"))
gfwatch = pd.DataFrame(list_, columns = column_names)

#### Processing rules by GFWatch
dic_censored = {}
censored_domains = {}
for index, row in gfwatch.iterrows():
    domain = row["censored_domain"]
    based_censored_domain = row["base_censored_domain"]
    blocking_rule = row["blocking_rules"]
    start = row["first_seen"]
    end = row["last_seen"]
    blocking_rule = translate_to_regex(blocking_rule)
    
    if domain not in censored_domains.keys():
        censored_domains[domain]=[]
    sub_dic = censored_domains[domain]
    sub_dic.append((start, end))
          
    if based_censored_domain not in dic_censored.keys():
        dic_censored[based_censored_domain] = {}
        sub_dic=dic_censored[based_censored_domain]
        sub_dic["blocking rule"]=blocking_rule
        sub_dic["blocking time"]=[]
        
    block_time = dic_censored[based_censored_domain]["blocking time"]
    block_time.append((start, end))

        
    
dates = generate_dates('2021-06-20','2022-02-09')
print(dates)
country ='CN'
for date in dates:
    print(date)
    filename = '/data/censorship/OONI/' + date +'/'+country+"/combined.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df =process_dataframe(df)
      
        df.to_csv('/data/censorship/OONI/' + date +'/'+country+"/groundtruth_combined.csv")
