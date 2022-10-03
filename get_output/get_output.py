import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
import datetime
from xgboost import XGBClassifier
import pickle
import glob
from sklearn.model_selection import StratifiedKFold

def get_accuracy_unsupervised(predictions, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i]==1:
            if y_test[i]==0:
                tn+=1
            else:
                fn+=1
        else:
            if y_test[i]==1:
                tp +=1
            else:
                fp+=1        
    return tp,fp,tn,fn
def get_accuracy(predictions, y_test):

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i]==1:
            if y_test[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if y_test[i]==0:
                tn +=1
            else:
                fn+=1            
    return tp,fp,tn,fn
def replace_nan(df, column):
    new_labels = []
    for val in df[column]:
        if pd.isna(val):
            new_labels.append("")
        else:
            new_labels.append(val)
    df[column]=new_labels
    return df

def relabel(df, col_name, base_cat):
    new_label = []
    for i in df[col_name]:
        if i==base_cat:
            new_label.append(0)
        else:
            new_label.append(1)
    df[col_name]=new_label
    return df

def accuracy(lst):
    accuracy =[]
    for item in lst:
        tp = item[0]
        fp = item[1]
        tn = item[2]
        fn = item[3]
        accuracy.append((tp+tn)/(tp+tn+fp+fn))
    return accuracy
def get_unsupervised_output(output):
    converted_output = []
    for i in list(output):
        if i==1:
            converted_output.append(0)
        else:
            converted_output.append(1)
    return converted_output

#######################change this line of code to see where the test data comes from #########
folder_test = "./train_validate_test_data/"
sub_folder = "621/Seed0/"
folder_test = folder_test+sub_folder

######################################################



X_USclean_test = pd.read_csv(folder_test+"X_USclean_test.csv")
y_USclean_test = X_USclean_test["blocking"]
X_CNclean_test = pd.read_csv(folder_test+"X_CNclean_test.csv")
y_CNclean_test = X_CNclean_test["blocking"]
X_GFCN_test = pd.read_csv(folder_test+"X_GFCN_test.csv")
y_GFCN_test = X_GFCN_test["GFWatchblocking_truth_new"]
X_OONICN_test = pd.read_csv(folder_test+"X_OONICN_test.csv")
y_OONICN_test = X_OONICN_test["blocking"]


if "Unnamed: 0" in X_USclean_test.columns:
    X_USclean_test = X_USclean_test.drop(columns = ["Unnamed: 0"])

if "Unnamed: 0.1" in X_USclean_test.columns:
    X_USclean_test = X_USclean_test.drop(columns = ["Unnamed: 0.1"])
    
if "Unnamed: 0" in X_CNclean_test.columns:
    X_CNclean_test = X_CNclean_test.drop(columns = ["Unnamed: 0"])

if "Unnamed: 0.1" in X_CNclean_test.columns:
    X_CNclean_test = X_CNclean_test.drop(columns = ["Unnamed: 0.1"])
    
    
if "Unnamed: 0" in X_GFCN_test.columns:
    X_GFCN_test = X_GFCN_test.drop(columns = ["Unnamed: 0"])

if "Unnamed: 0.1" in X_GFCN_test.columns:
    X_GFCN_test = X_GFCN_test.drop(columns = ["Unnamed: 0.1"])

if "Unnamed: 0" in X_OONICN_test.columns:
    X_OONICN_test = X_OONICN_test.drop(columns = ["Unnamed: 0"])

if "Unnamed: 0.1" in X_OONICN_test.columns:
    X_OONICN_test = X_OONICN_test.drop(columns = ["Unnamed: 0.1"])



######################### This is for temporal running ############

# folder_test = "./train_validate_test_data/V2/Temporal/"
# month_8 = pd.read_csv(folder_test+"CN_month_8.csv")
# month_9 = pd.read_csv(folder_test+"CN_month_9.csv")
# month_10 = pd.read_csv(folder_test+"CN_month_10.csv")
# month_11 = pd.read_csv(folder_test+"CN_month_11.csv")
# month_12 = pd.read_csv(folder_test+"CN_month_12.csv")
# month_1 = pd.read_csv(folder_test+"CN_month_1.csv")


# month = 7
# run_type = "single"
# folder_models = "./models/"+classifier+"/Temporal/"+str(month)+"/"

################################################################################
    
    





############################## change this line of code for the model ##################
classifier = "XGB"

drops=['blocking',"Domain","Index",
       'GFWatchblocking_truth_new', 'input']
folder_models = "./models/"+classifier+"/" + sub_folder
filename = "OONICN_train_OONICN_val"
model_fn = folder_models+filename+".sav"
#############################################################

model = pickle.load(open(model_fn, 'rb'))

for col in X_USclean_test.columns:
    print(col)

X_USclean_test_ = np.array(X_USclean_test.drop(columns = drops))
X_CNclean_test_ = np.array(X_CNclean_test.drop(columns = drops))
X_GFCN_test_ = np.array(X_GFCN_test.drop(columns = drops))
X_OONICN_test_ = np.array(X_OONICN_test.drop(columns = drops))



X_USclean_results = pd.DataFrame()
X_CNclean_results = pd.DataFrame()
X_GFCN_results = pd.DataFrame()
X_OONICN_results = pd.DataFrame()

if classifier =="IF" or classifier =="OCSVM":
    y_USclean = get_unsupervised_output(model.predict(X_USclean_test_))
    y_CNclean = get_unsupervised_output(model.predict(X_CNclean_test_))
    y_GFCN = get_unsupervised_output(model.predict(X_GFCN_test_))
    y_OONICN = get_unsupervised_output(model.predict(X_OONICN_test_))

    
    
else:
    y_USclean = model.predict(X_USclean_test_)
    y_CNclean = model.predict(X_CNclean_test_)
    y_GFCN = model.predict(X_GFCN_test_)
    y_OONICN = model.predict(X_OONICN_test_)
    
    
X_USclean_results["Predict"] = y_USclean
X_CNclean_results["Predict"] = y_CNclean
X_GFCN_results["Predict"] = y_GFCN
X_OONICN_results["Predict"] = y_OONICN




X_USclean_results.to_csv("./Results/"+classifier+"/"+ sub_folder + filename+"results_USclean_test.csv")
X_CNclean_results.to_csv("./Results/"+classifier+ "/"+ sub_folder + filename+"results_CNclean_test.csv")
X_GFCN_results.to_csv("./Results/"+classifier+"/" + "/"+ sub_folder+filename+"results_GFCN_test.csv")
X_OONICN_results.to_csv("./Results/"+classifier+ "/"+ sub_folder +filename+"results_OONICN_test.csv")






