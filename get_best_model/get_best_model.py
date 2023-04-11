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
from sklearn.svm import OneClassSVM
from multiprocessing import Pool, Process
import threading
import random

seed = 777

folder_model = "./models/"
VERBOSITY = 6
rng = np.random.default_rng(seed)

#### Train clean data, validate using clean China data


#### Train clean data, validate using clean China data

def get_accuracy_unsupervised(predictions, y_test): # -1 is abnormal, 1 is normal
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


def get_model_IF(contaminations):
    params={"max_features":[15,30,50],"n_estimators":[100,300,700]}
 
    for j in params["max_features"]:
        for t in params["n_estimators"]:

            model = IsolationForest(random_state=0, max_features = j, contamination=contaminations, n_estimators = t)
            models.append(model)


def get_model(model_name):
    models = []
    #TO Change: Set grid search parameters.
    if model_name == "OCSVM":
        ###### OCSVM #####
#         params={ "max_iter":[10,20,40]}
        models = [] 
#         for j in params["max_iter"]:
        model = OneClassSVM(kernel = "rbf",degree = 3,gamma='scale', max_iter = -1)
        models.append(model)

#         for j in params["max_iter"]:
#             model = OneClassSVM(gamma='auto', max_iter = j)
#             models.append(model)
    elif model_name == "IF":

        params={"max_features":[15,30,50],"n_estimators":[30,50,100,300], "contamination":[0.001,0.01,0.1,0.15] }
        for n in params["contamination"]:
            for j in params["max_features"]:
                for t in params["n_estimators"]:

                    model = IsolationForest(random_state=0, max_features = j, contamination=n, n_estimators = t)
                    models.append(model)
    elif model_name=="XGB":
        params={"max_depth":[15,30,45], "n_estimators":[15,50,100] }
        models = [] 
        for n in params["max_depth"]:
            for j in params["n_estimators"]:

                model = XGBClassifier(max_depth=n, n_estimators=j)
                models.append(model)
    return models
def accuracy(lst):
    accuracy =[]
    for item in lst:
        tp = item[0]
        fp = item[1]
        tn = item[2]
        fn = item[3]
        accuracy.append((tp+tn)/(tp+tn+fp+fn))
    return accuracy




def run_unsupervised(models,X_train,X_validation,y_validation):
    results = []
    index = 0
    for model in models:
        print(index)
        index+=1
        model.fit(X_train)
        predictions = model.predict(X_validation)

        tp,fp,tn,fn = get_accuracy_unsupervised(list(predictions), list(y_validation))
        results.append([tp,fp,tn,fn])
    for i in results:
        print(i)
    accuracy_score = accuracy(results)
    print(accuracy_score)
    
    

    test_result = []
    sorted_acc_index = np.argsort(accuracy_score)
    best_model = models[sorted_acc_index[-1]]
    print(best_model)

    return best_model
    
    
    
    
    
def run_supervised(models,X_train,y_train,X_validation,y_validation):
    results = []
    index = 0
    for model in models:
        print(index)
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        tp,fp,tn,fn = get_accuracy(np.array(predictions), y_validation)
        results.append([tp,fp,tn,fn])
        index+=1
    for i in results:
        print(i)
    accuracy_score = accuracy(results)
    print(accuracy_score)


    sorted_acc_index = np.argsort(accuracy_score)
    best_model = models[sorted_acc_index[-1]]
    print(best_model)

    return best_model
############################### change this line of code to change the folder the data is gotten from ##
f1 = "./train_validate_test_data/"
sub_folder = "621/Seed2/"
f1 = f1+sub_folder
###############################################################################

X_USclean_train = pd.read_csv(f1+"X_USclean_train.csv")
y_USclean_train = X_USclean_train["blocking"]
X_USclean_validate = pd.read_csv(f1+"X_USclean_validate.csv")
y_USclean_validate = X_USclean_validate["blocking"]



X_CNclean_train = pd.read_csv(f1+"X_CNclean_train.csv")
y_CNclean_train = X_CNclean_train["blocking"]
X_CNclean_validate = pd.read_csv(f1+"X_CNclean_validate.csv")
y_CNclean_validate = X_CNclean_validate["blocking"]


X_GFCN_train = pd.read_csv(f1+"X_GFCN_train.csv")
y_GFCN_train = X_GFCN_train["GFWatchblocking_truth_new"]
X_GFCN_validate = pd.read_csv(f1+"X_GFCN_validate.csv")
y_GFCN_validate = X_GFCN_validate["GFWatchblocking_truth_new"]



X_OONICN_train = pd.read_csv(f1+"X_OONICN_train.csv")
y_OONICN_train = X_OONICN_train["blocking"]
X_OONICN_validate = pd.read_csv(f1+"X_OONICN_validate.csv")
y_OONICN_validate = X_OONICN_validate["blocking"]




columns_to_drop1 = ["Unnamed: 0", "GFWatchblocking_truth","input","test_keys_ipv4","blocking"]

columns_to_drop2 = ["Unnamed: 0", "GFWatchblocking_truth_new","input","Domain","Index","blocking"]





X_USclean_train = X_USclean_train.drop(columns = ['Unnamed: 0.1'])
X_USclean_validate = X_USclean_validate.drop(columns = ['Unnamed: 0.1'])


X_USclean_train = X_USclean_train.drop(columns = columns_to_drop2)
X_USclean_validate = X_USclean_validate.drop(columns = columns_to_drop2)


X_CNclean_train = X_CNclean_train.drop(columns = columns_to_drop2)
X_CNclean_validate = X_CNclean_validate.drop(columns = columns_to_drop2)

X_GFCN_train = X_GFCN_train.drop(columns = columns_to_drop2)
X_GFCN_validate = X_GFCN_validate.drop(columns = columns_to_drop2)


X_OONICN_train = X_OONICN_train.drop(columns = columns_to_drop2)
X_OONICN_validate = X_OONICN_validate.drop(columns = columns_to_drop2)

################## change this line of code to determine which features to drop ###############
 

drops = ""
##########################################################


drops_list = [] 
if "m" in drops:
    drops_list = drops_list+["measurement_start_time"]
if "h" in drops:
    drops_list = drops_list + ['http_experiment_failure0', 'http_experiment_failure1',
   'http_experiment_failure2', 'http_experiment_failure3',
   'http_experiment_failure4', 'http_experiment_failure5',
   'http_experiment_failure6']
if "t" in drops:
    drops_list = drops_list + ["test_start_time"]
    
    
X_USclean_train = X_USclean_train.drop(columns = drops_list)
X_CNclean_train = X_CNclean_train.drop(columns = drops_list)
X_GFCN_train = X_GFCN_train.drop(columns = drops_list)
X_OONICN_train = X_OONICN_train.drop(columns = drops_list)

X_CNclean_validate = X_CNclean_validate.drop(columns = drops_list)
X_USclean_validate = X_USclean_validate.drop(columns = drops_list)
X_GFCN_validate = X_GFCN_validate.drop(columns = drops_list)
X_OONICN_validate = X_OONICN_validate.drop(columns = drops_list)


X_USclean_train = np.array(X_USclean_train)
y_USclean_train= np.array(y_USclean_train)


X_CNclean_train = np.array(X_CNclean_train)
y_CNclean_train= np.array(y_CNclean_train)

X_GFCN_train = np.array(X_GFCN_train)
y_GFCN_train= np.array(y_GFCN_train)

X_OONICN_train = np.array(X_OONICN_train)
y_OONICN_train= np.array(y_OONICN_train)

X_CNclean_validate = np.array(X_CNclean_validate)
y_CNclean_validate = np.array(y_CNclean_validate)

X_USclean_validate = np.array(X_USclean_validate)
y_USclean_validate = np.array(y_USclean_validate)

X_GFCN_validate = np.array(X_GFCN_validate)
y_GFCN_validate= np.array(y_GFCN_validate)




X_OONICN_validate = np.array(X_OONICN_validate)
y_OONICN_validate= np.array(y_OONICN_validate)



####################### change this line of code to get the models you want ########
model_name = "XGB"
file_name = "GFCN_train_GFCN_val"+drops
folder_models = "./models/"+model_name+"/"+sub_folder

models = get_model(model_name)

if model_name == "IF" or model_name == "OCSVM":
    best_model = run_unsupervised(models,X_GFCNclean_train,X_GFCN_validate,y_GFCN_validate)
else:
    
    best_model = run_supervised(models, X_GFCN_train, y_GFCN_train, X_GFCN_validate,y_GFCN_validate)
######################################################################



pickle.dump(best_model, open(folder_models+file_name+".sav", 'wb'))
