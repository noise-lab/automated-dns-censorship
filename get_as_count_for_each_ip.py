"""
 Export clean/non-censored domain_ASes based on majority vote
 """
from multiprocessing import Manager, Process
from dataframe_processing_helper_methods import format_datetime_from_file_name
import os
import pandas as pd
import numpy as np

cpus = 10  # change this param depend on how many CPUS you have/want to use

def asn_magic(domain_dict, domain):

    asn_dict = domain_dict[domain]

    total_asn_count = 0
    max_asn_num = 0
    max_asn_count = 0

    for asn_num in asn_dict.keys():

        asn_count = asn_dict[asn_num]
        total_asn_count = total_asn_count + asn_count

        if asn_count > max_asn_count:
            max_asn_count = asn_count
            max_asn_num = asn_num

    percent_of_asns = float(max_asn_count)/total_asn_count

    return max_asn_num, max_asn_count, percent_of_asns;

def worker(list_of_probe_files, home_folder):

    home_file_name = home_folder + r"CP_Analysis/"

    for probe_file in list_of_probe_files:

        cp_scan_only_name = probe_file.split('.')[0]
        scan_base_file_name = home_file_name + cp_scan_only_name + r"/"
        raw_dataframe_file_name = scan_base_file_name + "raw_dataframes/"

        print("Beginning to process file: " + str(cp_scan_only_name), flush=True)

        total_records = 0
        valid_records = 0
        clean_records = 0

        local_DOMAIN_ASes = {}
        local_DOMAIN_IPs = {}

        for file_name in os.listdir(raw_dataframe_file_name):

            df = pd.read_parquet(path=raw_dataframe_file_name+file_name, engine='pyarrow')

            total_records = total_records + df.shape[0]

            # drop invalid records

            valid_df = df[(df.in_control_group == True) & (df.control_response_start_success == True) & (df.excluded_below_threshold == False)]
            valid_records = valid_records + valid_df.shape[0]

            # drop non-clean records

            clean_df = valid_df[(valid_df.control_response_end_success == True) & (valid_df.anomaly == False) & (valid_df.connect_error == False) & (valid_df.test_query_successful == True)]
            clean_records = clean_records + clean_df.shape[0]

            test_url = clean_df['test_url']
            test_response_IP_count = clean_df['test_response_IP_count']

            for row_index in range(0, clean_df.shape[0]):

                domain = test_url.iloc[row_index]

                for ip_index in range(0, test_response_IP_count.iloc[row_index]):

                    ip = clean_df['test_response_' +str(ip_index)+ '_IP'].iloc[row_index]
                    asn = clean_df['test_response_' +str(ip_index)+ '_asnum'].iloc[row_index]

                    if domain not in local_DOMAIN_ASes:
                        local_DOMAIN_ASes[domain] = {asn: 1}
                    else:
                        if asn not in local_DOMAIN_ASes[domain]:
                            local_DOMAIN_ASes[domain][asn] = 1
                        else:
                            local_DOMAIN_ASes[domain][asn] = local_DOMAIN_ASes[domain][asn] + 1

                    # IP
                    if domain not in local_DOMAIN_IPs:
                        local_DOMAIN_IPs[domain] = {ip: 1}
                    else:
                        if ip not in local_DOMAIN_IPs[domain]:
                            local_DOMAIN_IPs[domain][ip] = 1
                        else:
                            local_DOMAIN_IPs[domain][ip] = local_DOMAIN_IPs[domain][ip] + 1

        # Print and store

        length = len(local_DOMAIN_ASes)

        domain_name = np.full(shape=length, dtype=object, fill_value="")
        max_asn_num = np.full(shape=length, dtype=np.int64(), fill_value=0)
        max_asn_count = np.full(shape=length, dtype=np.int64(), fill_value=0)
        percent_of_asns = np.full(shape=length, dtype=np.float64(), fill_value=0.0)
        unique_IPs_count = np.full(shape=length, dtype=np.int64(), fill_value=0)
        date = np.full(shape=length, dtype='datetime64[ns]', fill_value=format_datetime_from_file_name(cp_scan_only_name))

        index = 0
        for domain in local_DOMAIN_ASes.keys():

            domain_name[index] = domain
            max_asn_num[index], max_asn_count[index], percent_of_asns[index] = asn_magic(local_DOMAIN_ASes, domain)
            unique_IPs_count[index] = len(local_DOMAIN_IPs[domain].keys())
            index = index+1

        final_df_dict = {
            'domain_name': domain_name,
            'most_common_asn_num': max_asn_num,
            'most_common_asn_count': max_asn_count,
            'percent_of_asns': percent_of_asns,
            'unique_IPs_count': unique_IPs_count,
            'datetime': date,
        }

        final_df = pd.DataFrame(final_df_dict)

        # Save dataframe

        final_df.to_parquet(index=True, compression="gzip", engine='pyarrow', path=scan_base_file_name + "other_docs/max_asn.gzip")

        # Print and store records
        print("For file " +str(cp_scan_only_name)+ " Total Records: " +str(total_records))
        print("For file " +str(cp_scan_only_name)+ "Valid Records: " +str(valid_records))
        print("For file " +str(cp_scan_only_name)+ "Clean Records: " +str(clean_records))
        print("Ending file: " + str(cp_scan_only_name), flush=True)

    return

def main():

    print("Start Processing")

    home_folder = r"/home/jambrown/"  # TODO change this name for your file structure

    cp_downloads_zipped_file_name = home_folder + r"CP_Downloads/"

    list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
    list_of_zipped_cp_files.sort(reverse=True)  # Ensures most recent scans are processed first

    ps = list()

    cpus_minus_one = cpus-1

    probes_per_cpu = int(int(len(list_of_zipped_cp_files))/int(cpus_minus_one))
    cur_start_index = 0
    cur_end_index = probes_per_cpu # this is not inclusive

    for i in range(0, cpus+1):

        list_of_probe_files = list_of_zipped_cp_files[cur_start_index: cur_end_index]

        p = Process(target=worker, args=(list_of_probe_files, home_folder,))
        ps.append(p)
        p.start()

        cur_start_index = cur_start_index + probes_per_cpu
        cur_end_index = min(cur_end_index + probes_per_cpu, len(list_of_zipped_cp_files))

    for p in ps:
        p.join()

    print("Finished processing")

main()