import pandas as pd
import os

home_folder = r"/home/jambrown/"  # TODO change this name for your file structure

cp_downloads_zipped_file_name = home_folder + r"CP_Downloads/"

home_file_name = home_folder + r"CP_Analysis/"

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True)  # Ensures most recent scans are processed first

# Get label headers
df_0 = pd.read_parquet(path=home_folder+'CP_Analysis/CP_Satellite-2022-02-09-12-00-01/other_docs/max_asn.gzip', engine='pyarrow')
master_df = pd.DataFrame.from_records(data=df_0, nrows=1)
master_df.drop(labels=master_df.index[0:], inplace=True)

for folder_name in list_of_zipped_cp_files:

    # Load dataframe
    cp_scan_only_name = folder_name.split('.')[0]
    scan_base_file_name = home_file_name + cp_scan_only_name + r"/"
    asn_count_file_name = scan_base_file_name + "other_docs/max_asn.gzip"
    partial_df = pd.read_parquet(path=asn_count_file_name, engine='pyarrow')

    # Merge with master dataframe
    master_df = pd.concat([master_df, partial_df], ignore_index=True)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

clean_records = []

for row_index in range(0, master_df.shape[0]):

    if master_df['most_common_asn_num'].iloc[row_index] == 0 or master_df['most_common_asn_count'].iloc[row_index] < 10000 or master_df['percent_of_asns'].iloc[row_index] < 0.5:

        clean_records.append("DIRTY")

    else:

        clean_records.append("CLEAN")

master_df.insert(loc=master_df.shape[1], column="clean_records", value=clean_records)

# Save file
# fastparquet must be used so that the categorical variables associated with integers (ex rcode) will also deserialize into categorical variables
master_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=home_file_name + "max_asn_aggregate.gzip")