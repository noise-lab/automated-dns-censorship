import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
import datetime
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
import os
import glob
import shap
fn = "./train_validate_test_data/V2/X_CNclean_train.csv"
df = pd.read_csv(fn)
for col in df.columns:
    print(col)
import csv
title_match = ['title_match0', 'title_match1']
headers_match = ['headers_match0','headers_match1']
status_code_match = ['status_code_match0', 'status_code_match1']               
body_length_match = ['body_length_match0', 'body_length_match1']                  
http_experiment_failure = ['http_experiment_failure0', 'http_experiment_failure1',
       'http_experiment_failure2', 'http_experiment_failure3',
       'http_experiment_failure4', 'http_experiment_failure5',
       ] 
probe_asn = ["probe_asn0","probe_asn1","probe_asn2","probe_asn3","probe_asn4","probe_asn5","probe_asn6","probe_asn7","probe_asn8","probe_asn9"]
probe_network_name = ["probe_network_name0","probe_network_name1","probe_network_name2","probe_network_name3","probe_network_name4","probe_network_name5","probe_network_name6","probe_network_name7","probe_network_name8"]
resolver_network_name = ["resolver_network_name0","resolver_network_name1","resolver_network_name2","resolver_network_name3","resolver_network_name4","resolver_network_name5","resolver_network_name6","resolver_network_name7","resolver_network_name8"  ]
resolver_asn = ["resolver_asn0","resolver_asn1","resolver_asn2","resolver_asn3","resolver_asn4","resolver_asn5","resolver_asn6","resolver_asn7","resolver_asn8"]
test_keys_asn = ["test_keys_asn0","test_keys_asn1","test_keys_asn2","test_keys_asn3","test_keys_asn4","test_keys_asn5","test_keys_asn6","test_keys_asn7","test_keys_asn8","test_keys_asn9"]
test_keys_as_org_name = ["test_keys_as_org_name0","test_keys_as_org_name1","test_keys_as_org_name2","test_keys_as_org_name3","test_keys_as_org_name4","test_keys_as_org_name5","test_keys_as_org_name6","test_keys_as_org_name7","test_keys_as_org_name8","test_keys_as_org_name9"]
non_features = ["dns_experiment_failure","dns_consistency",'test_runtime', 'measurement_start_time','test_start_time', 'body_proportion']

list_aggregate = [test_keys_as_org_name,test_keys_asn,resolver_asn,resolver_network_name,probe_network_name,
probe_asn,http_experiment_failure,body_length_match, status_code_match,headers_match, title_match ]


def write_dict(dict_,filename):
    # open file for writing, "w" is writing
    w = csv.writer(open(filename, "w"))

    # loop over dictionary keys and values
    for key, val in dict_.items():

        # write every key and value to file
        w.writerow([key, val])
def aggregate(dict_):
    count = 0
    features = dict_["Features"]
    importance = dict_["Importance"]
    new_dict = {}
    for i in range(len(features)):
        new_dict[features[i]]=importance[i]
    another_dict = {}
    
    for li in list_aggregate:
        scores =0
        for item in li:
            count+=1
            scores+=new_dict[item]
        name = li[0][:-1]
        another_dict[name]=scores
    for name in non_features:
        another_dict[name]=new_dict[name]
    return another_dict

def get_importance(shap_values, feature_names):

    importance_dict = {}
    
    for index in range(0, len(shap_values[0])):
        mean_absolute_value = np.mean(np.abs(shap_values[:][index]))
        importance_dict[feature_names[index]] = mean_absolute_value
    sorted_features_list = sorted(importance_dict, key=importance_dict.__getitem__, reverse=True)
    sorted_num_list = sorted(importance_dict.values(), reverse=True)
    sorted_num_list = np.array(sorted_num_list)/sum(sorted_num_list)  # normalize weights 0 to 1
    printout_dict = {"Features": sorted_features_list, "Importance": sorted_num_list}
    aggregate_dict = aggregate(printout_dict)
    return aggregate_dict
fn = "./train_validate_test_data/V2/X_CNclean_train.csv"
df = pd.read_csv(fn)
drops = ""


columns_to_drop2 = ["Unnamed: 0", "GFWatchblocking_truth_new","input","Domain","Index","blocking"]

if "m" in drops:
    drops_list = drops_list+["measurement_start_time"]
if "h" in drops:
    drops_list = drops_list + ['http_experiment_failure0', 'http_experiment_failure1',
   'http_experiment_failure2', 'http_experiment_failure3',
   'http_experiment_failure4', 'http_experiment_failure5',
   'http_experiment_failure6']
if "t" in drops:
    drops_list = drops_list + ["test_start_time"]


X_val = df.drop(columns = columns_to_drop2)

feature_names = list(X_val.columns)
for col in feature_names:
    print(col)
X_val = X_val.to_numpy()



#############  for IF #############
# model = IsolationForest(random_state=0, max_features = 30, contamination=0.001, n_estimators = 20)
fn = "./train_validate_test_data/V2/X_CNclean_train.csv"
df = pd.read_csv(fn)
columns_to_drop2 = ["Unnamed: 0", "GFWatchblocking_truth_new","input","Domain","Index","blocking"]
sample = df.drop(columns=columns_to_drop2)
sample_ = np.array(sample)

model_name = "./models/IF/CNclean_train_GF_val.sav"
model = pickle.load(open(model_name, 'rb'))
# model.fit(sample_)
print(model)
exp = shap.TreeExplainer(model)
shap_values = exp.shap_values(sample)
print(shap_values)
print(len(shap_values))

dict_name = model_name.split("/")[-1].split(".")[0]
importances_dict = get_importance(shap_values,feature_names)
print(importances_dict)
# write_dict(importances_dict,"./Results/XGB/"+dict_name+"_feature_impt.csv")

# importances_dict = get_importance(shap_values,feature_names)

##############  for XGB ###########
model = XGBClassifier(max_depth=4, n_estimators=15)
model.fit(X_val,y_)
exp = shap.TreeExplainer(model)
shap_values = exp.shap_values(X_val)
importance_dict = get_importance(shap_values,feature_names)


sorted_features_list = sorted(importance_dict, key=importance_dict.__getitem__, reverse=True)
sorted_num_list = sorted(importance_dict.values(), reverse=True)
sorted_num_list = np.array(sorted_num_list)/sum(sorted_num_list)  # normalize weights 0 to 1
printout_dict = {"Features": sorted_features_list, "Importance": sorted_num_list}
write_dict(printout_dict,"./models/XGB/feature_imp_GFCN.csv")
