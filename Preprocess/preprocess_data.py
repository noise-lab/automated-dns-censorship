import json
import csv
import pandas as pd
import glob
import os
import datetime
import random
import copy
import pickle
from sklearn.model_selection import train_test_split
# from pyod.models.iforest import IForest
from joblib import dump, load

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
  
    benchmark = datetime.datetime(2021,6,20)
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
    df[column]=new_l
    
dates = generate_dates('2021-07-01','2022-02-09')
columns_selected_test = ["input",'measurement_start_time',
       'probe_asn','probe_network_name',
       'resolver_asn','resolver_network_name','test_runtime',
       'test_start_time', 'dns_experiment_failure', 'dns_consistency',
       'control_failure', 'http_experiment_failure', 'body_length_match',
       'body_proportion', 'status_code_match', 'headers_match', 'title_match',
       'accessible', 'blocking', 'x_status', 'test_keys_asn',
       'test_keys_as_org_name', 'test_keys_ipv4',"GFWatchblocking_truth"]
country = "CN"
ls=[]
for date in dates:
    print(date)
    filename = "/data/censorship/OONI/"+date+"/"+country+"/groundtruth_combined.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df=df[columns_selected_test]
        df["measurement_start_time"] = [convert_measurement_starttime(time) for time in df["measurement_start_time"]]
        df["test_start_time"] = [convert_measurement_starttime(time) for time in df["test_start_time"]]
        
        ls.append(df)
China_combined = pd.concat(ls)
dates = generate_dates('2021-07-01','2022-02-09')

columns_selected_test = ["input",'measurement_start_time',
       'probe_asn','probe_network_name',
       'resolver_asn','resolver_network_name','test_runtime',
       'test_start_time', 'dns_experiment_failure', 'dns_consistency',
       'control_failure', 'http_experiment_failure', 'body_length_match',
       'body_proportion', 'status_code_match', 'headers_match', 'title_match',
       'accessible', 'blocking', 'x_status', 'test_keys_asn',
       'test_keys_as_org_name', 'test_keys_ipv4']
country = "US"
ls=[]
for date in dates:
    print(date)
    filename = "/data/censorship/OONI/"+date+"/"+country+"/combined.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df=df[columns_selected_test]
        df["measurement_start_time"] = [convert_measurement_starttime(time) for time in df["measurement_start_time"]]
        df["test_start_time"] = [convert_measurement_starttime(time) for time in df["test_start_time"]]
        ls.append(df)
US_combined = pd.concat(ls)

benchmark = datetime.datetime(2021,6,20)
upper_benchmark = datetime.datetime(2021,7,1)
difference = upper_benchmark-benchmark
difference_in_s = difference.total_seconds()
# ### making the data start from 2021, July,1
China_combined=China_combined[China_combined["measurement_start_time"]>difference_in_s]
print(China_combined.shape)
China.combined("raw_CN.csv")
import numpy as np
### Adding groundtruth for the US
GFWatchblocking_truth = []
for i in US_combined['dns_consistency']:
    if i=="consistent":
        GFWatchblocking_truth.append(np.nan)
    else:
        GFWatchblocking_truth.append("Possible")


US_combined["GFWatchblocking_truth"]= GFWatchblocking_truth
##### Process US and CN data together

#### removing rows that do not pass control test
US_combined = replace_nan(US_combined,"control_failure")
US_combined = US_combined[US_combined["control_failure"]==""]

China_combined = replace_nan(China_combined,"control_failure")
China_combined = China_combined[China_combined["control_failure"]==""]
### removing the samples where probe_asn cannot be determined
China_combined = replace_nan(China_combined,"probe_asn")
China_combined = China_combined[China_combined["probe_asn"]!=""]

### removing the samples where probe_asn cannot be determined
US_combined = replace_nan(US_combined,"probe_asn")
US_combined = US_combined[US_combined["probe_asn"]!=""]
### removing the samples where resolver_asn cannot be determined
# US_combined = replace_nan(US_combined,"resolver_asn")
US_combined = US_combined[US_combined["resolver_asn"]!=""]
US_combined = US_combined[US_combined["resolver_asn"]!="AS0"]

China_combined = replace_nan(China_combined,"resolver_asn")
China_combined = China_combined[China_combined["resolver_asn"]!="AS0"]
China_combined = China_combined[China_combined["resolver_asn"]!=""]
### removing the samples where test_keys cannot be determined

US_combined = replace_nan(US_combined,"test_keys_asn")
US_combined = US_combined[US_combined["test_keys_asn"]!=""]
US_combined = US_combined[US_combined["test_keys_asn"]!="AS0"]

China_combined = replace_nan(China_combined,"test_keys_asn")
China_combined = China_combined[China_combined["test_keys_asn"]!="AS0"]
China_combined = China_combined[China_combined["test_keys_asn"]!=""]
### Getting the size of the US and China data

china_size = China_combined.shape[0]
us_size = US_combined.shape[0]

US_combined = replace_nan(US_combined,"body_proportion")
China_combined = replace_nan(China_combined,"body_proportion")
US_combined = US_combined[US_combined["body_proportion"]!= ""]
China_combined = China_combined[China_combined["body_proportion"]!= ""]
print(china_size)
print(us_size)
combined_uschina = pd.concat([China_combined,US_combined])
http_failure = []
for error in combined_uschina["http_experiment_failure"]:
    if not pd.isna(error):
        if "unknown_failure" in error:
            http_failure.append(error.split(":")[-1].strip())
        else:
            http_failure.append(error)
    else:
        http_failure.append("")

combined_uschina["http_experiment_failure"] = http_failure
# combined, labels_combined = relabel_category(combined,"http_experiment_failure")
# for name in labels_combined["Old label"]:
#     print(name)
dns_failure = []
for error in combined_uschina["dns_experiment_failure"]:
    if not pd.isna(error):
        if "unknown_failure" in error:
            dns_failure.append(error.split(":")[-1].strip())
        else:
            dns_failure.append(error)
    else:
        dns_failure.append("")

combined_uschina["dns_experiment_failure"] = dns_failure
# combined, labels_combined = relabel_category(combined,"dns_experiment_failure")
# for name in labels_combined["Old label"]:
#     print(name)
China_combined = combined_uschina.iloc[:china_size,:]
US_combined = combined_uschina.iloc[china_size:,:]
US_combined.to_csv("US_temp.csv")
China_combined.to_csv("CN_temp.csv")
combined_uschina = pd.concat([China_combined,US_combined])
relabels_dict = {}
columns_need_relabeling = ["probe_network_name","probe_asn","resolver_asn","resolver_network_name","status_code_match","status_code_match","headers_match",
                          "title_match","body_length_match","test_keys_asn","test_keys_as_org_name","dns_consistency", "dns_experiment_failure","http_experiment_failure"]

for col in columns_need_relabeling:
    print(col)
    combined_uschina, labels_combined = relabel_category(combined_uschina,col)
    relabels_dict[col]=labels_combined

    

ML_runs_modelsChina_combined = combined_uschina.iloc[:china_size,:]
US_combined = combined_uschina.iloc[china_size:,:]
US_combined.to_csv("./US_temp.csv")
China_combined.to_csv("./CN_temp.csv")
for k in relabels_dict.keys():
    df = relabels_dict[k]
    df.to_csv("../Re_labeling/"+k+".csv")
##### Applying one-hot-encoding
def process_string(string):
    return [int(i) for i in list(string)]

continuous_variables = ['test_runtime','measurement_start_time','test_start_time','body_proportion']
labels = ["blocking","GFWatchblocking_truth"]
cat_features = ['probe_asn','probe_network_name','resolver_network_name','resolver_asn','dns_experiment_failure','dns_consistency','http_experiment_failure','body_length_match','status_code_match','headers_match','title_match','test_keys_asn', 'test_keys_as_org_name']
# df = pd.DataFrame()
# for feature in cat_features:
#     print(feature)
#     uniques_val = len(combined_uschina[feature].unique())
#     if uniques_val>2:
#         cols_num = len("{0:b}".format(len(combined_uschina[feature].unique())))
#         values = ["{0:b}".format(val) for val in combined_uschina[feature]]
#         values = [add_zeros(val, cols_num) for val in values]

        
#         columns = [feature + str(i) for i in range(cols_num)]
#         vals_split = [process_string(string) for string in values]       
#         for i in range(cols_num):
#             df[columns[i]] = [val[i] for val in vals_split]
#     else:
#         df[feature]=df_combined[feature]
# print(df.head())

# df_China_combined = df.iloc[:china_size,:]
# df_US_combined = df.iloc[china_size:,:]
# df_US_combined.to_csv("US_temp_encoded.csv")
# df_China_combined.to_csv("CN_temp_encoded.csv")
US_combined = pd.read_csv("US_temp_encoded.csv")
CN_combined = pd.read_csv("CN_temp_encoded.csv")
US = pd.read_csv("US_temp.csv")
CN = pd.read_csv("CN_temp.csv")
print(CN.columns)
# print(US_combined.shape)
# print(US_combined.head())
labels = ["blocking","GFWatchblocking_truth","input"]
continuous_variables = ['test_runtime','measurement_start_time','test_start_time','body_proportion']

print(US.columns)
for feature in labels:
#     df[feature]=df_combined[feature]
    
    US_combined[feature] = US[feature]
for feature in continuous_variables:
#     df[feature]=df_combined[feature]
    US_combined[feature] = US[feature] 
print(US_combined.columns)
# US_combined["input"] = US["input"]

US_combined.to_csv("US_temp_encoded.csv")
print(US_combined)
# # print(df.columns)
for feature in labels:
#     df[feature]=df_combined[feature]
    
    CN_combined[feature] = CN[feature]
for feature in continuous_variables:
#     df[feature]=df_combined[feature]
    CN_combined[feature] = CN[feature] 
CN_combined["input"] = CN["input"]
CN_combined.to_csv("CN_temp_encoded.csv")
# print(df.columns)
CN_combined = pd.read_csv("CN_temp_encoded.csv")
print(CN_combined.head())
CN_combined = replace_nan(CN_combined,"blocking")
CN_combined = replace_nan(CN_combined,"GFWatchblocking_truth")
CN_combined = pd.concat([CN_combined[CN_combined["blocking"]=='False'],CN_combined[CN_combined["blocking"]=='dns']])

CN_combined = pd.concat([CN_combined[CN_combined["GFWatchblocking_truth"]==''],CN_combined[CN_combined["GFWatchblocking_truth"]=='Confirmed']])

CN_combined.to_csv("CN_temp_encoded_sanitized.csv")
CN_combined = pd.read_csv("CN_temp_encoded_sanitized.csv")
US = pd.read_csv("US_temp_encoded.csv")
US = replace_nan(US,"blocking")
US = replace_nan(US,"GFWatchblocking_truth")
# US = relabel(US, "blocking","False")
# US = relabel(US, "GFWatchblocking_truth","")
print(US["blocking"].unique())
US = pd.concat([US[US["blocking"]=='False'],US[US["blocking"]=='dns']])
print(US["blocking"].unique())
US_sample = US.sample(frac=0.025, replace=True, random_state=1)
print(US_sample.shape)
US_sample = relabel(US_sample, "GFWatchblocking_truth","")
US_sample = relabel(US_sample, "blocking","False")
US_sample.to_csv("US_temp_encoded_sanitized_sampled.csv")
