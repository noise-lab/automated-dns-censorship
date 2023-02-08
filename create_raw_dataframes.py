#
# Transformed compressed Satellite data files into a machine-learning usable format
# The program assumes that all the tar files have already been downloaded using CP_crawler.py
# Before starting, please specify "cp_downloads_zipped_file_name" and "home_file_name"
# The following folders are created during execution:
# 1. Folder containing subfolders coresponding to each scan, with each containing
#    a. a folder holding blockpages.json, dns.pkt, resolvers.json
#    b. a folder holding unmodified dataframes (compressed) extracted from results.json
#    c. a folder for temporarily holding the unzipped tar file and segmented JSON files. This folder is deleted after the raw dataframes have been compiled.
# 2. Folder holding a singular dataframe of all probes across all scans that employ the selected vantage points, one for each country
# 3. Folder containing subfolders corresponding to each country
#    a. ML-ready dataframes (compressed) derived from the unmodified dataframes, one for each country (all scans combined in one file)
#    b. Anomaly Vector for that country's ML dataframe
#    c. Standard Scaler for that country's ML dataframe

import os.path
from dataframe_processing_helper_methods import *
import shutil
import tarfile

# Get list of files in folder
home_folder = r"/home/jambrown/" # TODO change this name for your file structure
cp_downloads_zipped_file_name = home_folder + r"CP_Downloads/"
home_file_name = home_folder + r"CP_Analysis/"

max_vantagepoint_count_per_country = 15

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

list_of_zipped_cp_files = list_of_zipped_cp_files

print("Begin creating raw dataframes from JSON file.", flush=True)

for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]

    print("Beginning to process file: " +str(cp_scan_only_name), flush=True)

    # Create file substructure
    scan_base_file_name = home_file_name + cp_scan_only_name + r"/"
    os.mkdir(scan_base_file_name)

    # Create new scan-specific folder and sub-directory
    raw_dataframes_file_name = scan_base_file_name + r'raw_dataframes/'
    os.mkdir(raw_dataframes_file_name)

    # Create subfolder containing the other docs extracted (blockpages.json, dns.pkt, resolvers.json, resolvers_raw.json), as well as a dictionary of vantage points
    other_docs_file_name = scan_base_file_name + r"other_docs/"
    os.mkdir(other_docs_file_name)

    # Create (temporary) subdirectory to hold split JSON files and other extracted files
    temp_file_name = scan_base_file_name + r'temp/'
    os.mkdir(temp_file_name)

    # Create directory within temp to hold the split files
    json_divide_file_name = temp_file_name + r'json_split_files/'
    os.mkdir(json_divide_file_name)

    # Create directory within temp to hold the extracted files
    unzipped_download_file_name = temp_file_name + r'unzipped_download/'
    os.mkdir(unzipped_download_file_name)

    # Extract files to a temporary folder
    print("Beginning extraction of file " +str(cp_scan_only_name), flush=True)
    tar = tarfile.open(cp_downloads_zipped_file_name + zipped_cp_file, "r:gz")
    tar.extractall(path=unzipped_download_file_name)
    tar.close()
    print("Finished extraction of file " + str(cp_scan_only_name), flush=True)

    # Move other items (blockpages.json, dns.pkt, resolvers.json, resolvers_raw.json) to new location
    shutil.move(unzipped_download_file_name +cp_scan_only_name +r"/blockpages.json", other_docs_file_name)
    shutil.move(unzipped_download_file_name +cp_scan_only_name +r"/dns.pkt", other_docs_file_name)
    shutil.move(unzipped_download_file_name +cp_scan_only_name +r"/resolvers.json", other_docs_file_name)

    # Begin splitting the results.json file into manageable parts

    og_JSON_download_filename = unzipped_download_file_name +cp_scan_only_name +r"/results.json"
    json_splitfile_output_filename_base = r'_JSON_Part.txt'
    records_per_file = 250000

    # Divide results.json into manageable parts
    DivideJSON(og_JSON_download_filename, \
               json_divide_file_name, \
               json_splitfile_output_filename_base, \
               records_per_file)

    # Create dataframe with all features
    print("Create raw dataframe", flush=True)
    batch_dt_input = format_datetime_from_file_name(cp_scan_only_name) # the starting time for this batch of probes as a string
    splitfile_count = len(os.listdir(json_divide_file_name))

    for fileNumber in range(0, splitfile_count):

        json_split_filename = json_divide_file_name + str(fileNumber) + json_splitfile_output_filename_base

        df = json_to_df(json_split_filename, batch_dt_input)

        # fastparquet must be used so that the categorical variables associated with integers (ex rcode) will also deserialize into categorical variables
        df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=(raw_dataframes_file_name + str(fileNumber) + "_raw_dataframe.gzip"))
        print("Parquet file size in bytes with gzip compression: " + str(os.path.getsize(raw_dataframes_file_name + str(fileNumber) + "_raw_dataframe.gzip")))
        print("Finished file " +str(fileNumber) +" of " +str(splitfile_count), flush=True)

    # Delete temp folder
    shutil.rmtree(temp_file_name)

print("Finished creating raw dataframes.", flush=True)