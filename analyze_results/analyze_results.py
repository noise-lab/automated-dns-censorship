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
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


def compare_results(predict, actual):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predict)):
        p = predict[i]
        a = actual[i]
        if p==0:
            if a ==0:
                tn +=1
            else:
                fn+=1
        else:
            if a==1:
                tp +=1
            else:
                fp+=1
    return [tp,fp,tn,fn]
def get_statistic(tp,fp,tn,fn):
    tpr = 0
    fpr = 0
    tnr = 0
    fnr = 0
    if tp+fn>0:
        tpr = tp/(tp+fn)
        fnr = fn/(tp+fn)
    if fp+tn>0:
        fpr = fp/(fp+tn)
        tnr = tn/(fp+tn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = 0
    if tp+fp>0:
        precision = tp/(tp+fp)
    f1 = 0
    if tp+fp+fn >0:
        f1 = tp/(tp+(fp+fn)/2)
    return tpr,fpr,tnr,fnr,accuracy,precision,f1

def get_unsupervised_output(output):
    converted_output = []
    for i in output:
        if i==-1:
            converted_output.append(1)
        else:
            converted_output.append(0)
    return converted_output
################### for non temporal #############
result_classifier = "IF"
result_filename_end = "OONICN_test.csv"
result_folder = "./Results/"+result_classifier + "/"
# sub_folder="811/Seed2/"


data_filename = "./train_validate_test_data/V2/"+"X_"+result_filename_end

# data_filename = "./train_validate_test_data/"+sub_folder +"X_"+result_filename_end

data = pd.read_csv(data_filename)
# print(data.columns)
data = data.drop(columns = ["Unnamed: 0"])

result_df = pd.read_csv(result_folder+"OONIresults_OONICN_test.csv")
# print(result_df.columns)
result_df = result_df.drop(columns = ["Unnamed: 0"])


data_result = pd.concat([data, result_df], axis=1)
print(data_result)

################  this is for temporal ###############
### concating the results and the data**************

data_folder = "./train_validate_test_data/V2/Temporal/"
################## change this line of code ################
train_month_end = 12
seed = "/Seed0"

result_folder = "./Results/XGB/Temporal/"+str(train_month_end) +seed +"/"
###############################################################
for fn in glob.glob(result_folder+"*"):
    print(fn)
    filename_end = fn.split("/")[-1]
    test_month = int(filename_end.split("test_month")[-1].split("_")[0])
    data = pd.read_csv(data_folder+"CN_month_"+str(test_month)+".csv")
    data= data.drop(columns = ["Unnamed: 0"])
    result = pd.read_csv(fn)
    result= result.drop(columns = ["Unnamed: 0"])
    data_result = pd.concat([data,result],axis=1)
    print(result_folder + "data_result"+filename_end)
    data_result.to_csv(result_folder + "data_result"+filename_end)
    
folder = "./Results/IF/Temporal/"
ls = []

TP = []
FP = []
TN = []
FN = []
TPR = []
FPR = []
TNR = []
FNR = []
ACC = []
Precision = []
F1 = []
starttrain = []
endtrain = []
testmonth = []

name = "OONI"

for fn in glob.glob(folder+"*/*/"+name+"_aggregate.csv"):
    df = pd.read_csv(fn)
#     print(df.columns)
    df = df.drop(columns = ["Unnamed: 0"])
    ls.append(df)
aggregate_all = pd.concat(ls)
for start_train in aggregate_all["Start train"].unique():
    df1 = aggregate_all[aggregate_all["Start train"]==start_train]
    for end_train in df1["End train"].unique():
        df2 = df1[df1["End train"]==end_train]
        for test_month in df2["Test month"].unique():
            df3 = df2[df2["Test month"]==test_month]
            tp = sum(df3[name+" TP"])/df3.shape[0]
            fp = sum(df3[name+" FP"])/df3.shape[0]
            tn = sum(df3[name+" TN"])/df3.shape[0]
            fn = sum(df3[name+" FN"])/df3.shape[0]
            
            tpr = sum(df3[name+" TPR"])/df3.shape[0]
            fpr = sum(df3[name+" FPR"])/df3.shape[0]
            tnr = sum(df3[name+" TNR"])/df3.shape[0]
            fnr = sum(df3[name+" FNR"])/df3.shape[0]
            
            acc = sum(df3[name+" ACC"])/df3.shape[0]
            precision = sum(df3[name+" Precision"])/df3.shape[0]
            f1 = sum(df3[name+" F1"])/df3.shape[0]
            st = list(df3["Start train"])[0]
            et = list(df3["End train"])[0]
            tm = list(df3["Test month"])[0]
            
            TP.append(tp)
            FP.append(fp)
            TN.append(tn)
            FN.append(fn)
            TPR.append(tpr)
            FPR.append(fpr)
            TNR.append(tnr)
            FNR.append(fnr)
            ACC.append(acc)
            Precision.append(precision)
            F1.append(f1)
            starttrain.append(st)
            endtrain.append(et)
            testmonth.append(tm)
df = pd.DataFrame()
df[name+ ' TP'] = TP
df[name+ ' FP'] = FP
df[name+ ' TN'] = TN
df[name+ ' FN'] = FN
df[name+ ' TPR'] = TPR
df[name+ ' FPR'] = FPR
df[name+ ' TNR'] = TNR
df[name+ ' FNR'] = FNR

df[name+ ' ACC'] = ACC
df[name+ ' Precisioin'] = Precision 
df[name+ ' F1'] = F1
df["Start train"] = starttrain
df["End train"] = endtrain
df["Test month"] = testmonth
df

df.to_csv(folder + name+"summary.csv")
