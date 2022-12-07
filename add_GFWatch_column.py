import pandas as pd
import re
from multiprocessing import Process

def process_date(time):
    converted_datetime = pd.to_datetime(arg=time, format='%Y/%m/%d')

    return converted_datetime

def process_time(time):
    converted_datetime = pd.to_datetime(arg=time, format='%Y-%m-%d %H:%M:%S')

    return converted_datetime
def translate_to_regex(rule):
    return_str = rule
    if return_str[0]=="*":
        return_str = ".*"+return_str[1:]
        
    if return_str[-1]=="*":
        return_str = return_str[:-1]+".*"
    return return_str
def match_rule(domain, rule):
    
    if re.match(rule,domain):
        return True
    return False

def find_match_rule(domain, dic_censored):
    for dic_key in dic_censored.keys():
        if dic_key in domain:
            return dic_key
    return ""

def check_domain_block(domain, censored_domains):
    return domain in censored_domains.keys()

def check_range_time(start_time, range_times):
    for i in range(len(range_times)):
        range_time = range_times[i]
        start = process_date(range_time[0])
        end = process_date(range_time[1])
        status = (start_time.date() >= start.date()) and (start_time.date() <= end.date())
        if status:
            return 1
    return 0

def get_domainname(name):
    new_name = name.split("//")[-1]
#     if new_name[:4]=="www.":
#         new_name = new_name[4:]
    if new_name[-1]=="/":
        new_name = new_name[:-1]
    return new_name

def process_dataframe(df, censored_domains, dic_censored, save_filename, process_num):

    dns_blocking = []

    for row_index in range(0, df.shape[0]):

        if row_index%1000 == 0:
            print(str(row_index), flush=True)

        time_ranges = []
        domain = get_domainname(df['test_url'].iloc[row_index])
        measurement_time = df['batch_datetime'].iloc[row_index]
        if check_domain_block(domain, censored_domains=censored_domains):
            time_ranges = censored_domains[domain]
        else:
            matched_domain = find_match_rule(domain, dic_censored=dic_censored)
            if matched_domain != "":
                blocking_rule = dic_censored[matched_domain]["blocking rule"]

                if match_rule(domain, blocking_rule):
                    time_ranges = dic_censored[matched_domain]["blocking time"]
        if len(time_ranges) > 0:
            dns_blocking.append(check_range_time(measurement_time, time_ranges))
        else:
            dns_blocking.append(0)

    df["GFWatch_Censored"] = dns_blocking

    # save partial df

    df.to_parquet(index=True, compression="gzip", engine='pyarrow', path=save_filename+str(process_num)+"_partial_GFWatch_df.gzip")

    return df

def remove_invalid_records(df):

    valid_df = df[(df.in_control_group == True) & (df.control_response_start_success == True) & (df.excluded_below_threshold == False)]

    return valid_df.copy()

def run_add_GFWatch_column(data_df, table_filename, save_filename):

    GFWatch_df = pd.read_csv(table_filename)
    list_ = []
    index = GFWatch_df.columns.tolist()
    column_names = index[0].split("|")
    for i in GFWatch_df[GFWatch_df.columns[0]]:
        list_.append(i.split("|"))
    gfwatch = pd.DataFrame(list_, columns=column_names)

    dic_censored = {}
    censored_domains = {}
    for index, row in gfwatch.iterrows():
        domain = row["censored_domain"]
        based_censored_domain = row["base_censored_domain"]
        blocking_rule = row["blocking_rules"]
        start = row["first_seen"]
        end = row["last_seen"]
        blocking_rule = translate_to_regex(blocking_rule)

        if domain not in censored_domains.keys():
            censored_domains[domain]=[]
        sub_dic = censored_domains[domain]
        sub_dic.append((start, end))

        if based_censored_domain not in dic_censored.keys():
            dic_censored[based_censored_domain] = {}
            sub_dic=dic_censored[based_censored_domain]
            sub_dic["blocking rule"]=blocking_rule
            sub_dic["blocking time"]=[]

        block_time = dic_censored[based_censored_domain]["blocking time"]
        block_time.append((start, end))

    # add GFWatch_Censored column to dataframe
    data_df['GFWatch_Censored'] = 0

    print("Finished constructing GF_Watch dictionary", flush=True)

    ps = list()

    cpus_minus_one = 19

    count_per_process = int(data_df.shape[0]/(cpus_minus_one))

    begin = 0
    end = count_per_process

    for i in range(0, cpus_minus_one+1):

        print("Begin process " +str(i))
        print("Begin line " +str(begin))
        print("End line " + str(end))

        partial_df = data_df.iloc[begin: end]

        partial_df = partial_df.copy()
        print("Partial DF length: " +str(partial_df.shape[0]))

        p = Process(target=process_dataframe,
                    args=(partial_df, censored_domains, dic_censored, save_filename, i))
        ps.append(p)
        p.start()

        begin = begin + count_per_process
        end = min(end + count_per_process, data_df.shape[0])

    for p in ps:
        p.join()

    print("Begin merging dataframe", flush=True)

    complete_dataframe_list = []

    for i in range(0, cpus_minus_one+1):

        partial_df = pd.read_parquet(path=save_filename + str(i) + "_partial_GFWatch_df.gzip", engine='pyarrow')
        complete_dataframe_list.append(partial_df)

    complete_df = pd.concat(complete_dataframe_list, ignore_index=True, axis=0)

    return complete_df

home_folder = r"/home/jambrown/"  # TODO change this name for your file structure

home_file_name = home_folder + r"CP_Analysis/"

country_code = "CN"
country_name = "China"

ml_ready_data_file_name = home_file_name +country_code+ "/ML_ready_dataframes/"
aggregate_file_name = home_file_name +country_code+ "/raw_dataframe.gzip"


GFWatch_table_filename = home_file_name+ r'gfwatch_censored_domains.csv'
save_gfwatch_file_name = home_file_name +country_code+ "/GFWatch_Combined_Dataset.gzip"
GFWatch_partial_dataframes_filename = home_file_name +country_code+ "/GFWatch_partial_dataframes/"

# If we have not already preprocessed the datasets

original_df = pd.read_parquet(path=aggregate_file_name, engine='pyarrow')

original_df_length = original_df.shape[0]

valid_df = remove_invalid_records(original_df)

valid_df_length = valid_df.shape[0]

print("Total number of probes: " +str(original_df_length))
print("Valid number of probes: " +str(valid_df_length))

complete_data_df = run_add_GFWatch_column(valid_df, GFWatch_table_filename, GFWatch_partial_dataframes_filename)

complete_data_df.to_parquet(index=True, compression="gzip", engine='pyarrow', path=save_gfwatch_file_name)

print("Finished merging dataframes. End of Program.")


