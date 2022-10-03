import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
import datetime
from xgboost import XGBClassifier
import pickle

from sklearn.svm import OneClassSVM
from multiprocessing import Pool, Process
import threading


seed = 2

folder_model = "./models/"

rng = np.random.default_rng(seed)

CN = pd.read_csv("./data/CN_v4_encoded.csv")
US = pd.read_csv("./data/US_v4_encoded.csv")

CN = CN.drop(columns = ['Unnamed: 0'])
US = US.drop(columns = ['Unnamed: 0'])

clean_CN = CN[CN["blocking"]==0]
clean_CN = clean_CN[clean_CN["GFWatchblocking_truth_new"]==0]
OONI_CN = CN[CN["blocking"]==1]
GF_CN = CN[CN["GFWatchblocking_truth_new"] == 1]
clean_US = US[US["blocking"]==0]
clean_US = clean_US[clean_US["GFWatchblocking_truth_new"]==0]







# X_USclean_train, X_USclean_rest, y_USclean_train, y_USclean_rest = train_test_split(clean_US,clean_US["blocking"] , test_size=0.2, random_state =seed)
# X_USclean_validate, X_USclean_test, y_USclean_validate, y_USclean_test = train_test_split(X_USclean_rest,y_USclean_rest, test_size=0.5, random_state =seed)

# X_CNclean_train, X_CNclean_rest, y_CNclean_train, y_CNclean_rest = train_test_split(clean_CN,clean_CN["blocking"] , test_size=0.2, random_state =seed)
# X_CNclean_validate, X_CNclean_test, y_CNclean_validate, y_CNclean_test = train_test_split(X_CNclean_rest,y_CNclean_rest , test_size=0.5, random_state =seed)


# X_GFCN_train, X_GFCN_rest, y_GFCN_train, y_GFCN_rest = train_test_split(GF_CN,GF_CN["GFWatchblocking_truth_new"] , test_size=0.2, random_state =seed)
# X_GFCN_validate, X_GFCN_test, y_GFCN_validate, y_GFCN_test = train_test_split(X_GFCN_rest,y_GFCN_rest , test_size=0.5, random_state =seed)

# X_OONICN_train, X_OONICN_rest, y_OONICN_train, y_OONICN_rest = train_test_split(OONI_CN,OONI_CN["blocking"] , test_size=0.2, random_state =seed)
# X_OONICN_validate, X_OONICN_test, y_OONICN_validate, y_OONICN_test = train_test_split(X_OONICN_rest,y_OONICN_rest , test_size=0.5, random_state =seed)




X_GFCN_train = pd.concat([X_GFCN_train,X_CNclean_train])
X_OONICN_train = pd.concat([X_OONICN_train,X_CNclean_train])
X_GFCN_validate = pd.concat([X_GFCN_validate,X_CNclean_validate])
X_OONICN_validate = pd.concat([X_OONICN_validate,X_CNclean_validate])
X_GFCN_test = pd.concat([X_GFCN_test,X_CNclean_test])
X_OONICN_test = pd.concat([X_OONICN_test,X_CNclean_test])

################################## change this line of code to change which folder is the data saved to

folder = "./train_validate_test_data/811/Seed"+str(seed)+"/"

############################################################################################
X_USclean_train.to_csv(folder+"X_USclean_train.csv")
X_USclean_validate.to_csv(folder+'X_USclean_validate.csv')
X_USclean_test.to_csv(folder+"X_USclean_test.csv")


X_CNclean_train.to_csv(folder+"X_CNclean_train.csv")
X_CNclean_validate.to_csv(folder+"X_CNclean_validate.csv")
X_CNclean_test.to_csv(folder+"X_CNclean_test.csv")

X_GFCN_train.to_csv(folder+"X_GFCN_train.csv")
X_GFCN_validate.to_csv(folder+"X_GFCN_validate.csv")
X_GFCN_test.to_csv(folder+"X_GFCN_test.csv")

X_OONICN_train.to_csv(folder+"X_OONICN_train.csv")
X_OONICN_validate.to_csv(folder+"X_OONICN_validate.csv")
X_OONICN_test.to_csv(folder+"X_OONICN_test.csv")


# def split_by_month(data,name):
#     new_benchmark = datetime.datetime(2021,7,1)
#     upper_benchmark_7 = (datetime.datetime(2021,7,31) - new_benchmark).total_seconds()
#     upper_benchmark_8 = (datetime.datetime(2021,8,31)- new_benchmark).total_seconds()
#     upper_benchmark_9 = (datetime.datetime(2021,9,30)- new_benchmark).total_seconds()
#     upper_benchmark_10 = (datetime.datetime(2021,10,31)- new_benchmark).total_seconds()
#     upper_benchmark_11 = (datetime.datetime(2021,11,30)- new_benchmark).total_seconds()
#     upper_benchmark_12 = (datetime.datetime(2021,12,31)- new_benchmark).total_seconds()
#     upper_benchmark_1 = (datetime.datetime(2022,1,31)- new_benchmark).total_seconds()

#     index_7=[]
#     index_8=[]
#     index_9=[]
#     index_10=[]
#     index_11=[]
#     index_12=[]
#     index_1=[]

#     index = 0
#     for row in data.iterrows():
#         time  = row[1]["measurement_start_time"]
#         if time> upper_benchmark_1:
#             index_1.append(index)
#         elif time> upper_benchmark_12:
#             index_12.append(index)
#         elif time> upper_benchmark_11:
#             index_11.append(index)
#         elif time> upper_benchmark_10:
#             index_10.append(index)
#         elif time> upper_benchmark_9:
#             index_9.append(index)
#         elif time> upper_benchmark_8:
#             index_8.append(index)
#         elif time> upper_benchmark_7:
#             index_7.append(index)
#         index +=1
#     month_7=data.iloc[index_7]
#     month_8=data.iloc[index_8]
#     month_9=data.iloc[index_9]
#     month_10=data.iloc[index_10]
#     month_11=data.iloc[index_11]
#     month_12=data.iloc[index_12]
#     month_1=data.iloc[index_1]

#     folder = "./train_validate_test_data/Temporal/"+name
#     month_7.to_csv(folder + "_month_7.csv")
#     month_8.to_csv(folder + "_month_8.csv")
#     month_9.to_csv(folder + "_month_9.csv")
#     month_10.to_csv(folder + "_month_10.csv")
#     month_11.to_csv(folder + "_month_11.csv")
#     month_12.to_csv(folder + "_month_12.csv")
#     month_1.to_csv(folder + "_month_1.csv")

# split_by_month(CN,"CN")
# split_by_month(US,"US")







