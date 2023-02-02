import json
import csv
import pandas as pd
import gzip
import glob
import os


#### extracting from json files into dataframes #####


############     these are the featues that we will extract from JSON files b###### 
main_cols = ['input', 'measurement_start_time', 'probe_asn', 'probe_cc', 'probe_ip','probe_network_name','resolver_asn', "report_id","resolver_asn",'resolver_ip', 'resolver_network_name', "solftware_name",
'test_name', 'test_runtime', 'test_start_time']

special = {'test_keys':["dns_experiment_failure","dns_consistency","control_failure","http_experiment_failure","body_length_match","body_proportion","status_code_match","headers_match","title_match","accessible","blocking","x_status"]}

special_special = {'test_keys':{"queries":{"answers":["asn","as_org_name", "ipv4"]}}}

df_columns = main_cols.copy()
df_columns = df_columns+special['test_keys']
df_columns = df_columns+ ["test_keys_" + i for i in special_special['test_keys']["queries"]["answers"]]
# df = pd.DataFrame(columns=df_columns)
def generate_dates(start_date, end_date):
    lst =  pd.date_range(start_date, end_date, freq='D')
    
    list_date = []
    for i in range(len(lst)):
        list_date.append(lst[i].date().strftime("%Y-%m-%d"))
    return list_date

def convert_to_csv(filename):
    with gzip.open(filename, 'rb') as f:
        file_content = f.readlines() 
    thelist =[]
    for i in range(len(file_content)):
        index = 0
        lst =["" for i in range(len(df_columns))]
        a = file_content[i].decode('UTF-8')
        dic = json.loads(a)

        for col in main_cols:
            if col in dic.keys():
                lst[index] = dic[col]
            index+=1
        test_keys = dic["test_keys"]
        for col in special["test_keys"]:
            if col in test_keys.keys():
                lst[index] = test_keys[col]
            index+=1
        if "queries" in dic["test_keys"].keys():
            
            answers = dic["test_keys"]["queries"]#[0]["answers"][0]

            if dic["test_keys"]["queries"] is not None:
                for j in range(len(dic["test_keys"]["queries"])):
                    answer = dic["test_keys"]["queries"][j]["answers"]
                    if answer is not None:
                        for t in range(len(answer)):
                            sub_index=index


                            ans = answer[t]

                            for col in special_special["test_keys"]["queries"]["answers"]:
                                if col in ans.keys():
                                    lst[sub_index] = str(ans[col])+ "_"
                                sub_index+=1

        thelist.append(lst)
    df = pd.DataFrame(data = thelist, columns = df_columns)
    
    ### columns need to be processed again
    cols_need_processing = ['test_keys_asn', 'test_keys_as_org_name',
       'test_keys_ipv4']
    for col in cols_need_processing:
        list_ = list(df[col])
        list_=[i[:-1].split("_") for i in list_]
        df[col]=list_

    
    
    # folder = "/".join(filename.split("/")[:-1])
    # new_filename = filename.split("/")[-1].split(".")[0]
#     print(df)
#     df.to_csv(folder+"/"+new_filename+".csv")
    return df


def combine_data_by_day(folder):
 
    for filename in glob.glob(folder+"/*.jsonl.gz"):
        print(filename)
        df = convert_to_csv(filename)
        ls.append(df)
    if len(ls)>0:
        df = pd.concat(ls)
        print(df.shape)

        df.to_csv(folder+"/combined.csv")
        return df
    return None






#############   TODO #############
folder_OONI_data = "../CN/webconnectivity/" # Specify the folder where you want to convert the data, change the country name for a different country
dates = generate_dates('2021-02-23','2021-02-24') # Specify the date range you want to convert the data, this include the last date
##############################
ls = []
for date in dates:
    print(folder_OONI_data + date)
    df = combine_data_by_day(folder_OONI_data + date)
    print(df.shape)
    if not df is None:
        ls.append(df)
final_df = pd.concat(ls)
final_df.to_csv(folder_OONI_data +"combine_all_dates.csv")
    




