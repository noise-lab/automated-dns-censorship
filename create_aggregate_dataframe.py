import os
import pandas as pd

cur_country_name = "United States" # TODO enter country name here
cur_country_code = "US" # TODO enter country code here
home_folder = r"/home/jambrown/" # TODO change this name for your file structure

home_file_name = home_folder + r"CP_Analysis/"
cp_downloads_zipped_file_name = home_folder + r"CP_Downloads/"

# Create Country-specific file directory
country_file_name = home_file_name +cur_country_code+ r'/'
os.mkdir(country_file_name)

partial_aggregate_dataframes_file_name = country_file_name+ r"partial_aggregate_dataframes/"
os.mkdir(partial_aggregate_dataframes_file_name)

# Merge raw files into singular file for use
print("Begin merging raw dataframes into a singular dataframe.", flush=True)

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

print("Input Vantage Point Files")

vp_df = pd.read_csv("VantagePoint_CSV_List_V2.csv")

vp_df = vp_df[vp_df.COUNTRY_CODE == cur_country_code]

vp_list = vp_df['IP'].tolist()

print(vp_list)

print("Begin merging " +cur_country_name+ " dataframes.", flush=True)

# # Create dataframe with all the files
# # Start by reading an initial dataframe to obtain the column headers
# df_0 = pd.read_parquet(path='/home/jambrown/CP_Analysis/CP_Satellite-2022-02-09-12-00-01/raw_dataframes/0_raw_dataframe.gzip', engine='pyarrow')
# master_df = pd.DataFrame.from_records(data=df_0, nrows=1)
# master_df.drop(labels=master_df.index[0:], inplace=True)

# note that we assume that the file names in the CP_Download folder are exactly the same as those in CP_Analysis
# daily scan folders should probably be placed in a separate file so that there names can be processed properly
for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]
    print("Beginning to process file: " +str(cp_scan_only_name), flush=True)

    scan_base_file_name = home_file_name + cp_scan_only_name + r"/"
    other_docs_file_name = scan_base_file_name + r"other_docs/"
    raw_dataframes_file_name = scan_base_file_name + r'raw_dataframes/'

    splitfile_count = len(os.listdir(raw_dataframes_file_name))

    daily_dataframe_list = []

    # Concatenate dataframes and remove extra columns, selecting by country and vantage point
    for fileNumber in range(0, splitfile_count):

        partial_df = pd.read_parquet(path=raw_dataframes_file_name + str(fileNumber) + "_raw_dataframe.gzip", engine='pyarrow')

        columns_to_keep = [
            'test_url',
            'vantage_point',
            'batch_datetime',
            'average_matchrate',
            'untagged_controls',
            'untagged_response',
            'passed_liveness',
            'connect_error',
            'anomaly',
            'in_control_group',
            'excluded_below_threshold',
            'delta_time',
            'control_response_start_success',
            'control_response_end_success',
            'control_response_start_has_type_a',
            'control_response_start_rcode',
            'control_response_end_has_type_a',
            'control_response_end_rcode',
            'test_query_successful',
            'test_query_unsuccessful_attempts',
            'test_noresponse_1_has_type_a',
            'test_noresponse_1_rcode',
            'test_noresponse_2_has_type_a',
            'test_noresponse_2_rcode',
            'test_noresponse_3_has_type_a',
            'test_noresponse_3_rcode',
            'test_noresponse_4_has_type_a',
            'test_noresponse_4_rcode',
            'test_response_has_type_a',
            'test_response_rcode',
            'test_response_IP_count',
            'test_response_0_IP_match',
            'test_response_0_http_match',
            'test_response_0_cert_match',
            'test_response_0_asnum_match',
            'test_response_0_asname_match',
            'test_response_0_match_percentage',
            'test_response_0_asnum',
            'test_response_1_IP_match',
            'test_response_1_http_match',
            'test_response_1_cert_match',
            'test_response_1_asnum_match',
            'test_response_1_asname_match',
            'test_response_1_match_percentage',
            'test_response_1_asnum',
            'test_response_2_IP_match',
            'test_response_2_http_match',
            'test_response_2_cert_match',
            'test_response_2_asnum_match',
            'test_response_2_asname_match',
            'test_response_2_match_percentage',
            'test_response_2_asnum',
            'test_response_3_IP_match',
            'test_response_3_http_match',
            'test_response_3_cert_match',
            'test_response_3_asnum_match',
            'test_response_3_asname_match',
            'test_response_3_match_percentage',
            'test_response_3_asnum',
            'test_response_4_IP_match',
            'test_response_4_http_match',
            'test_response_4_cert_match',
            'test_response_4_asnum_match',
            'test_response_4_asname_match',
            'test_response_4_match_percentage',
            'test_response_4_asnum',
        ]

        partial_df = partial_df[partial_df.columns.intersection(columns_to_keep)]  # drop unwanted columns

        # Select only the vantage points we want

        partial_df = partial_df[partial_df.vantage_point.isin(vp_list)]

        daily_dataframe_list.append(partial_df)

        print("Partial df size: " +str(partial_df.shape[0]))

        print("Finished merging file number " + str(fileNumber) +" of " +str(splitfile_count), flush=True)

    daily_df = pd.concat(daily_dataframe_list, ignore_index=True, axis=0)

    daily_df.to_parquet(index=True, compression="gzip", engine='pyarrow',
                          path=partial_aggregate_dataframes_file_name + cp_scan_only_name + "_partial_aggregate_dataframe.gzip")

    del daily_df

    print("Finished merging the scan from day " +cp_scan_only_name)

# Concatenate dataframes and save

# Merge with master dataframe

master_dataframe_list = []

for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]

    partial_df = pd.read_parquet(path=partial_aggregate_dataframes_file_name + cp_scan_only_name + "_partial_aggregate_dataframe.gzip", engine='pyarrow')

    master_dataframe_list.append(partial_df)

master_df = pd.concat(master_dataframe_list, ignore_index=True, axis=0)

# Shuffle records
master_df = master_df.sample(frac=1)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

# Save file
master_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=country_file_name + "raw_dataframe.gzip")

print("End of merging " + cur_country_name + " dataframes.", flush=True)

print("End of raw dataframe merging section.", flush=True)