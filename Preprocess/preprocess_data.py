import json
import csv
import pandas as pd
import gzip
import glob
import os
import datetime
import random
import copy
import pickle
from sklearn.model_selection import train_test_split
import numpy as np


main_cols = ['input', 'measurement_start_time', 'probe_asn', 'probe_cc', 'probe_ip','probe_network_name','resolver_asn', "report_id","resolver_asn",'resolver_ip', 'resolver_network_name', "solftware_name",
'test_name', 'test_runtime', 'test_start_time']

special = {'test_keys':["dns_experiment_failure","dns_consistency","control_failure","http_experiment_failure","body_length_match","body_proportion","status_code_match","headers_match","title_match","accessible","blocking","x_status"]}

special_special = {'test_keys':{"queries":{"answers":["asn","as_org_name", "ipv4"]}}}

df_columns = main_cols.copy()
df_columns = df_columns+special['test_keys']
df_columns = df_columns+ ["test_keys_" + i for i in special_special['test_keys']["queries"]["answers"]]

def process_string(string):
    return [int(i) for i in list(string)]
def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date
def replace_nan(df, column):
    new_labels = []
    for val in df[column]:
        if pd.isna(val):
            new_labels.append("")
        else:
            new_labels.append(val)
    df[column]=new_labels
    return df
def get_domainname(name):
    new_name = name.split("//")[-1]
#     if new_name[:4]=="www.":
#         new_name = new_name[4:]
    if "/" in new_name:
        new_name = new_name.split("/")[0]
    return new_name

def relabel(df, col_name, base_cat):
    new_label = []
    for i in df[col_name]:
        if i==base_cat:
            new_label.append(0)
        else:
            new_label.append(1)
    df[col_name]=new_label
    return df
def replace_nan(df, column):
    new_labels = []
    for val in df[column]:
        if pd.isna(val):
            new_labels.append("")
        else:
            new_labels.append(val)
    df[column]=new_labels
    return df







###### Preprocess the data
def add_zeros(string, string_length):
    padd_num = string_length - len(string)
    return "0"*padd_num+string
def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date

def findNaN(df, column):
    col = list(df[column])
    nan_rows = []
    for i in range(len(col)):
        if pd.isna(col[i]):
            nan_rows.append(i)
    return nan_rows
def convert_measurement_starttime(time):
    converted_datetime = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S')
    benchmark = datetime.datetime(2021,7,1)
    difference = converted_datetime-benchmark
    return difference.total_seconds()


def relabel_category(df, column):
    df[column] = [str(val) for val in df[column]]
    df = replace_nan(df,column)
    unique_labels = list(df[column].unique())
    ## sorting the labels
    unique_labels.sort()
    dict_label = {}

    for i in range(len(unique_labels)):
        dict_label[unique_labels[i]]=i
    df[column]=[dict_label[val] for val in df[column]]
    
    labels_df = pd.DataFrame()
    labels_df["Old label"]=unique_labels
    labels_df["New label"]=[i for i in range(len(unique_labels))]
    ## return the relabeled dataframee as well as the new label
    return df, labels_df

def replace_nan(df, column):
    new_labels = []
    for val in df[column]:
        if pd.isna(val):
            new_labels.append("")
        else:
            new_labels.append(val)
    df[column]=new_labels
    return df



##### Loading data


US = pd.read_csv("./data/V1/US_v1.csv") 
US["Index"] = [i for i in range(US.shape[0])]
print(US.columns)



dates = generate_dates('2021-07-01','2022-02-09')
country = "CN"
ls = []
count = 0
for date in dates:
    filename = "/data/censorship/OONI/"+date+"/"+country+"/groundtruth_combined_new.csv"
    if os.path.exists(filename):
        count +=1
        df = pd.read_csv(filename)
        ls.append(df)
CN = pd.concat(ls)
CN["Index"]=[i for i in range(CN.shape[0])]
CN = replace_nan(CN,"blocking")
CN.to_csv("./data/CN_v1.csv")



CN["measurement_start_time"] = [convert_measurement_starttime(time) for time in 
CN["test_start_time"] = [convert_measurement_starttime(time) for time in CN["test_start_time"]]




######## Sanitizing the dataset #######

#### removing rows that do not pass control test
US = replace_nan(US,"control_failure")
US = US[US["control_failure"]==""]

CN = replace_nan(CN,"control_failure")
CN = CN[CN["control_failure"]==""]

### removing the samples where probe_asn cannot be determined
CN = replace_nan(CN,"probe_asn")
CN = CN[CN["probe_asn"]!=""]

### removing the samples where probe_asn cannot be determined
US = replace_nan(US,"probe_asn")
US = US[US["probe_asn"]!=""]

### removing the samples where resolver_asn cannot be determined
US= replace_nan(US,"resolver_asn")
US = US[US["resolver_asn"]!=""]
US = US[US["resolver_asn"]!="AS0"]

CN = replace_nan(CN,"resolver_asn")
CN = CN[CN["resolver_asn"]!="AS0"]
CN = CN[CN["resolver_asn"]!=""]

### removing the samples where test_keys cannot be determined
US = replace_nan(US,"test_keys_asn")
US = US[US["test_keys_asn"]!=""]
US = US[US["test_keys_asn"]!="AS0"]

CN = replace_nan(CN,"test_keys_asn")
CN = CN[CN["test_keys_asn"]!="AS0"]
CN = CN[CN["test_keys_asn"]!=""]


### removing the samples where body_proportion cannot be determined
US = replace_nan(US,"body_proportion")
US = US[US["body_proportion"]!=""]

CN = replace_nan(CN,"body_proportion")
CN = CN[CN["body_proportion"]!=""]



CN = replace_nan(CN,"blocking")
CN = replace_nan(CN,"GFWatchblocking_truth_new")
CN = pd.concat([CN[CN["blocking"]=='False'],CN[CN["blocking"]=='dns']])
CN = pd.concat([CN[CN["GFWatchblocking_truth_new"]==''],CN[CN["GFWatchblocking_truth_new"]=='Confirmed']])


GFWatchblocking_truth = []
for i in US['dns_consistency']:
    if i=="consistent":
        GFWatchblocking_truth.append(np.nan)
    else:
        GFWatchblocking_truth.append("Possible")
US["GFWatchblocking_truth_new"]= GFWatchblocking_truth
US = replace_nan(US,"blocking")
US = replace_nan(US,"GFWatchblocking_truth_new")
US = US[US["blocking"]=='False']
US = US[US["GFWatchblocking_truth_new"]=='']


US.to_csv("./data/US_v2_sanitized.csv")
CN.to_csv("./data/CN_v2_sanitized.csv")


US_sample = US.sample(frac=0.025, replace=True, random_state=1)
US_sample.to_csv("./data/US_v2_sanitized_sample.csv")

columns_selected_test = ["Index","input","Domain",'measurement_start_time',"control_failure",
       'probe_asn','probe_network_name',
       'resolver_asn','resolver_network_name','test_runtime',
       'test_start_time', 'dns_experiment_failure', 'dns_consistency',
        'http_experiment_failure', 'body_length_match',
       'body_proportion', 'status_code_match', 'headers_match', 'title_match',
        'blocking',  'test_keys_asn',
       'test_keys_as_org_name', "GFWatchblocking_truth_new","test_keys_ipv4" ]
CN = CN[columns_selected_test]
US_sample["Domain"] = [get_domainname(i) for i in US_sample["input"]]
US_sample = US_sample[columns_selected_test]



######### Processing the dataframes

combined_USCN = pd.concat([CN,US_sample])
http_failure = []
for error in combined_USCN["http_experiment_failure"]:
    if not pd.isna(error):
        if "unknown_failure" in error:
            http_failure.append(error.split(":")[-1].strip())
        else:
            http_failure.append(error)
    else:
        http_failure.append("")

combined_USCN["http_experiment_failure"] = http_failure

dns_failure = []
for error in combined_USCN["dns_experiment_failure"]:
    if not pd.isna(error):
        if "unknown_failure" in error:
            dns_failure.append(error.split(":")[-1].strip())
        else:
            dns_failure.append(error)
    else:
        dns_failure.append("")

combined_USCN["dns_experiment_failure"] = dns_failure

CN = combined_USCN.iloc[:CN.shape[0],:]
US = combined_USCN.iloc[CN.shape[0]:,:]
US.to_csv("US_v3_sampled.csv")
CN.to_csv("CN_v3.csv")

relabels_dict = {}
columns_need_relabeling = ["probe_network_name","probe_asn","resolver_asn","resolver_network_name","status_code_match","headers_match",
                          "title_match","body_length_match","test_keys_asn","test_keys_as_org_name","dns_consistency", "dns_experiment_failure","http_experiment_failure"]

for col in columns_need_relabeling:
    combined_USCN, labels_combined = relabel_category(combined_USCN,col)
    relabels_dict[col]=labels_combined
for k in relabels_dict.keys():
    df = relabels_dict[k]
    df.to_csv("./data/Labels/"+k+".csv")

    
continuous_variables = ['test_runtime','measurement_start_time','test_start_time','body_proportion']
labels = ["blocking","GFWatchblocking_truth_new","input","Domain","Index"]
cat_features = ['probe_asn','probe_network_name','resolver_network_name','resolver_asn','dns_experiment_failure','dns_consistency','http_experiment_failure','body_length_match','status_code_match','headers_match','title_match','test_keys_asn', 'test_keys_as_org_name']



df = pd.DataFrame()
for feature in cat_features:
    print(feature)
#     print(combined_USCN[feature])
    uniques_val = len(combined_USCN[feature].unique())
    print(uniques_val)
    if uniques_val>2:
        cols_num = len("{0:b}".format(uniques_val))
        values = ["{0:b}".format(val) for val in combined_USCN[feature]]### this is the values
        values = [add_zeros(val, cols_num) for val in values]## this is the value after padding zeros

        
        columns = [feature + str(i) for i in range(cols_num)]
        print(columns)
#         vals_split = [list(string) for string in values]  

        for i in range(cols_num):
            df[columns[i]] = [int(val[i]) for val in vals_split]
    else:
        df[feature]= list(combined_USCN[feature])

for col in continuous_variables:
    df[col] = list(combined_USCN[col])
for col in labels:
    df[col] = list(combined_USCN[col])
    
CN_encoded = df.iloc[:CN.shape[0],:]
US_encoded = df.iloc[CN.shape[0]:,:]
CN_encoded["blocking"]=[str(item) for item in CN_encoded["blocking"]]
CN_encoded = replace_nan(CN_encoded,"GFWatchblocking_truth_new")
CN_encoded = relabel(CN_encoded, "blocking","False")
CN_encoded = relabel(CN_encoded, "GFWatchblocking_truth_new","")
US_encoded["blocking"]=[str(item) for item in US_encoded["blocking"]]
US_encoded = replace_nan(US_encoded,"GFWatchblocking_truth_new")
US_encoded = relabel(US_encoded, "blocking","False")
US_encoded = relabel(US_encoded, "GFWatchblocking_truth_new","")
US_encoded.to_csv("US_v4_encoded.csv")
CN_encoded.to_csv("CN_v4_encoded.csv")






