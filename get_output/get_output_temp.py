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

standard_drop=['blocking','GFWatchblocking_truth_new', 'input', 'Index',"Domain","Unnamed: 0"]



def run_test(train_month, test_month, model_name,   seed):
    folder_model = "./models/"+model_name+"/Temporal/"+str(train_month)+"/"+seed+"/"
    print(folder_model)
    df = pd.DataFrame()
    results = []
    results_df = pd.DataFrame()
    for filename in glob.glob(folder_model + "*"+".sav"):
        name = filename.split("/")[-1].split(".")[0]
        print(filename)
        model = pickle.load(open(filename, 'rb'))
        data_val = filename.split("_")[1]
#         drop_columns = filename.split(".")[-2].split("_")[-1]
        drops = standard_drop

        folder = "./train_validate_test_data/V2/Temporal/CN_"
        CN_temp = pd.read_csv(folder+"month_" + str(test_month)+".csv")

        CN_temp_X = CN_temp.drop(columns = drops)
        y_CN_temp = CN_temp["GFWatchblocking_truth_new"]
        

        X_test = np.array(CN_temp_X)
        if model_name == "XGB":
            result = model.predict(X_test)
        else:
            
            result = get_unsupervised_output(model.predict(X_test))
        tp,fp,tn,fn = get_accuracy(np.array(result), np.array(y_CN_temp))
        df[name] = result
        results.append([tp,fp,tn,fn])
        
    accuracy_score = accuracy(results)
    print(accuracy_score)
    print(df)

    print("./Results/"+model_name+"/Temporal/test"+str(train_month)+"/"+seed+"/test_month"+str(test_month)+"_predictions.csv")
    df.to_csv("./Results/"+model_name+"/Temporal/"+str(train_month)+"/"+seed+"/test_month"+str(test_month)+"_predictions.csv")
    
    
    

model = "IF"

# acc = "single"
tm=12
seed = "Seed2"

run_test(tm, 1, model,  seed)

