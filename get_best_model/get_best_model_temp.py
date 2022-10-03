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

seed = 777

folder_model = "./models/"
VERBOSITY = 6
rng = np.random.default_rng(seed)

#### Train clean data, validate using clean China data

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

    
def runSKFold(splits, X, y):

    runs = []
    skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
    for train, test in skf.split(X, y):
#         print(train)
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        arr = [X_train, X_test, y_train, y_test]
        runs.append(arr)
    return runs
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

def get_model(model_name):
    models = []
    if model_name == "OCSVM":
#         ###### OCSVM #####
#         params={ "max_iter":[20,40,60,80]}
#         models = [] 

#         for j in params["max_iter"]:
        model = OneClassSVM(kernel = "rbf", degree = 3,gamma='scale', max_iter = -1)
        models.append(model)
    elif model_name == "IF":
        ###### IF #######
# Model parameters
#         params = {"max_features":[15],"n_estimators":[100], "contamination":[0.001] }
        params={"max_features":[15,30,50],"n_estimators":[100,300,700], "contamination":[0.001,0.0025,0.005,0.007] }
        for n in params["contamination"]:
            for j in params["max_features"]:
                for t in params["n_estimators"]:

                    model = IsolationForest(random_state=0, max_features = j, contamination=n, n_estimators = t)
                    models.append(model)
    elif model_name=="XGB":
        params={"max_depth":[5,10,15], "n_estimators":[15,30] }

        # Model parameters
#         params={"max_depth":[15,30,45], "n_estimators":[15,30,50,75,100,200] }


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

def get_unsupervised_output(output):
    converted_output = []
    for i in list(output):
        if i==1:
            converted_output.append(0)
        else:
            converted_output.append(1)
    return converted_output
    
    
    


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

def run_unsupervised_model(args):#, X_train, X_validation,y_validation, results,prediction_val):
    print("running one model")
    index = args[0]
    model = args[1]
    X_train = args[2]
    X_validate = args[3]
    y_validate = args[4]
    model.fit(X_train)
    predictions = model.predict(X_validate)
    tp,fp,tn,fn = get_accuracy_unsupervised(list(predictions), list(y_validate))
    return [index,model,[tp,fp,tn,fn]]


def run_unsupervised(models,X_train,X_validation,y_validation):

    new_args = [(i,models[i],X_train,X_validation,y_validation) for i in range(len(models))]
    
    with Pool(16) as p:
        results = p.map(run_unsupervised_model,new_args)
    return results

def run_supervised_model(args):
    print("running supervised model")
    index = args[0]
    model = args[1]
    X_train = args[2]
    y_train = args[3]
    X_validate = args[4]
    y_validate = args[5]
    model.fit(X_train, y_train)
    predictions = model.predict(X_validate)
    tp,fp,tn,fn = get_accuracy(np.array(predictions), y_validate)
    return [index,model,[tp,fp,tn,fn]]
   
def run_supervised(models,X_train,y_train,X_validation,y_validation):
    new_args = [(i,models[i],X_train,y_train,X_validation,y_validation) for i in range(len(models))]
    with Pool(16) as p:
        results = p.map(run_supervised_model,new_args)
    return results



##### TRAIN: clean CHINA, VALIDATE: GF-CHINA
##### TEST: GF-CHINA, CLEAN:US, OONI-CHINA


def run_experiments(month, model_name,run_type,sup_unsup,validation_set,drops, seed): ### run_type can be single single or accumulative
    folder = "./train_validate_test_data/V2/Temporal/CN_"
    
    
################### change this line of code to determine when is the start accumulative month#####
    start_month = 7
###############################################
    if run_type == "single":
        CN_temp = pd.read_csv(folder+"month_" + str(month)+".csv")
    else:
        ls = []
        for i in range(start_month,month+1):
            temp = pd.read_csv(folder+"month_" + str(i)+".csv")
            print(folder+"month_" + str(i)+".csv")
            ls.append(temp)
        CN_temp = pd.concat(ls)
    if 'Unnamed: 0' in CN_temp.columns:
        CN_temp = CN_temp.drop(columns = ['Unnamed: 0'],axis=1)
    clean_CN = CN_temp[CN_temp["blocking"]==0]
    clean_CN = clean_CN[clean_CN["GFWatchblocking_truth_new"]==0]

    OONI_CN = CN_temp[CN_temp["blocking"]==1]
    GF_CN = CN_temp[CN_temp["GFWatchblocking_truth_new"] == 1]
    X_CNclean_train, X_CNclean_rest, y_CNclean_train, y_CNclean_rest = train_test_split(clean_CN,clean_CN["blocking"] , test_size=0.33, random_state = seed)
    X_GFCN_train, X_GFCN_rest, y_GFCN_train, y_GFCN_rest = train_test_split(GF_CN,GF_CN["GFWatchblocking_truth_new"] , test_size=0.33, random_state = seed)
    X_OONICN_train, X_OONICN_rest, y_OONICN_train, y_OONICN_rest = train_test_split(OONI_CN,OONI_CN["blocking"] , test_size=0.33, random_state = seed)
    columns_to_drop = ['blocking',
           'GFWatchblocking_truth_new', 'input', 'Index',"Domain"]
  
    if validation_set =="GF":
        X_validate_ = pd.concat([X_CNclean_rest,X_GFCN_rest])
        y_validate_ = pd.concat([y_CNclean_rest,y_GFCN_rest])
        X_train_ = pd.concat([X_CNclean_train,X_GFCN_train])
        y_train_ = pd.concat([y_CNclean_train,y_GFCN_train])
    else:
        X_validate_ = pd.concat([X_CNclean_rest,X_OONICN_rest])
        y_validate_ = pd.concat([y_CNclean_rest,y_OONICN_rest])
        X_train_ = pd.concat([X_CNclean_train,X_OONICN_train])
        y_train_ = pd.concat([y_CNclean_train,y_OONICN_train])
        
    if sup_unsup == "unsup":
        X_train_ = X_CNclean_train
        y_train_ = y_CNclean_train
        
    drops_list = columns_to_drop
    if "m" in drops:
        drops_list = drops_list+["measurement_start_time"]
    if "h" in drops:
        drops_list = drops_list + ['http_experiment_failure0', 'http_experiment_failure1',
       'http_experiment_failure2', 'http_experiment_failure3',
       'http_experiment_failure4', 'http_experiment_failure5',
       'http_experiment_failure6']
    if "t" in drops:
        drops_list = drops_list + ["test_start_time"]
    X_train = np.array(X_train_.drop(columns = drops_list))
    y_train= np.array(y_train_)
    
    print(len(X_train_.drop(columns = drops_list).columns))

    X_validate= np.array(X_validate_.drop(columns = drops_list))
    y_validate= np.array(y_validate_)


    models = get_model(model_name)
    folder_models = "./models/"+model_name+"/Temporal/"+str(month)+"/Seed"+str(seed)+"/"

    file_name =  validation_set +"_"+drops+run_type
    if sup_unsup == "unsup":
        results = run_unsupervised(models, X_train,X_validate, y_validate)
    else:
        results = run_supervised(models, X_train, y_train, X_validate, y_validate)
        
        
    trained_models = [i[1] for i in results]
    actual_results = [i[2] for i in results]
    
    accuracy_score = accuracy(actual_results)
    print(accuracy_score)
    sorted_acc_index = np.argsort(accuracy_score)
    best_index = results[sorted_acc_index[-1]][0]
    best_model = trained_models[best_index]
    print(best_model)
    
    if run_type == "acc":
        pickle.dump(best_model, open(folder_models+file_name+"_trainmonth_"+ str(start_month)+".sav", 'wb'))
        print(folder_models+file_name+"_trainmonth_"+ str(start_month)+".sav")
    else:
        pickle.dump(best_model, open(folder_models+file_name+".sav", 'wb'))
        print(folder_models+file_name+".sav")
        


    

# run_experiments(11,"IF","acc","unsup","GF","",0)
# run_experiments(11,"IF","acc","unsup","GF","",1)
# run_experiments(11,"IF","acc","unsup","GF","",2)
run_experiments(12,"IF","acc","unsup","OONI","",2)
# run_experiments(12,"IF","acc","unsup","GF","",1)
# run_experiments(12,"IF","acc","unsup","GF","",2)




