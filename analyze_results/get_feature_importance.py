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
import csv

#os.system("pip3 install shap")

fn = "../data/train_validate_test/Seed0/X_CNclean_train.csv"
df = pd.read_csv(fn)


title_match = [col for col in df.columns if "title_match" in col]
headers_match = [col for col in df.columns if "headers_match" in col]
status_code_match = [col for col in df.columns if "status_code_match" in col]
body_length_match = [col for col in df.columns if "body_length_match" in col]        
http_experiment_failure = [col for col in df.columns if "http_experiment_failure" in col]        
probe_asn = [col for col in df.columns if "probe_asn" in col]        
probe_network_name = [col for col in df.columns if "probe_network_name" in col]             
resolver_network_name = [col for col in df.columns if "resolver_network_name" in col]        
resolver_asn = [col for col in df.columns if "resolver_asn" in col]        
test_keys_asn = [col for col in df.columns if "test_keys_asn" in col]        
test_keys_as_org_name = [col for col in df.columns if "test_keys_as_org_name" in col]                          
             
                        
###### features that was not encoded, dont need to aggregated using shap ######
non_features = ["dns_experiment_failure","dns_consistency",'test_runtime', 'measurement_start_time','test_start_time', 'body_proportion']
####### features that were encoded, need to aggregate ###############
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
    
    for index in range(0, shap_values.shape[1]):
     
        mean_absolute_value = np.mean(np.abs(shap_values[:,index]))
        importance_dict[feature_names[index]] = mean_absolute_value
    print(importance_dict)
    sorted_features_list = sorted(importance_dict, key=importance_dict.__getitem__, reverse=True)
    sorted_num_list = sorted(importance_dict.values(), reverse=True)
    sorted_num_list = np.array(sorted_num_list)/sum(sorted_num_list)  # normalize weights 0 to 1
    printout_dict = {"Features": sorted_features_list, "Importance": sorted_num_list}
    aggregate_dict = aggregate(printout_dict)
    return aggregate_dict

df = pd.read_csv(fn)
drops = ""


columns_to_drop2 = ["Unnamed: 0", "GFWatchblocking_truth_new","input","Domain","Index","blocking"]

# #########################   TODO    #########################
# ############ Modigy this to see which model you want to see the feature importance
classifier = "XGB"
Seed = 0
################################################################



model_folder = "../Best_Models/"+classifier+"/Seed"+str(Seed)+"/"
validation_data_folder = "../data/train_validate_test/Seed"+str(Seed)+"/"

for fn in glob.glob(model_folder+"*"):
    y_column_name = "blocking"
    validate_data = fn.split("/")[-1].split("_")[2]
    if validate_data == "GFCN":
        y_column_name = "GFWatchblocking_truth_new"
        
  
    full_fn = validation_data_folder+"X_"+validate_data+"_validate.csv"
    df =  pd.read_csv(full_fn)
    X_val = df.drop(columns = columns_to_drop2)
    y_val = df[y_column_name]
    feature_names = list(X_val.columns)

    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
    
    model = pickle.load(open(fn, 'rb'))
    if classifier == "IF" or classifier=="OCSVM":
        model.fit(X_val)
        
    else:

        model.fit(X_val,y_val)
    exp = shap.TreeExplainer(model)
    shap_values = exp.shap_values(X_val)
 


    importances_dict = get_importance(shap_values,feature_names)

    feature_folder = "../Best_Models/"+classifier+"/Seed"+str(Seed)+"/"+ fn.split("/")[-1].split(".")[0]+"_feature_impt.csv"
    
    
    write_dict(importances_dict,feature_folder)

