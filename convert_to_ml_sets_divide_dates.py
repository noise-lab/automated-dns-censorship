"""
 Divide the aggregate dataframes into train, validation, and test sets.
 All invalid records are dropped and two copies of the validation and test sets are made
 One copy has both the unclean and clean (presumed uncensored) data, and the other only the clean data
 """

from convert_to_ml_helper_methods import *
from joblib import dump
import os

# remove invalid records
def remove_invalid_records(df):

    valid_df = df[(df.in_control_group == True) & (df.control_response_start_success == True) & (df.excluded_below_threshold == False)]

    return valid_df.copy()

def get_ASN_list(input_df, row):

    IP_count = input_df['test_response_IP_count'].iloc[row]

    asn_list = []

    for index in range(0, min(IP_count, 5)): # A maximum of five responses are stored

        asn_list.append(input_df['test_response_' +str(index)+ '_asnum'].iloc[row])

    return asn_list

def get_unclean_record_indices (input_df, AS_count_df, country_code):

    # Convert AS_count_df into dictionary

    domain_datetime_clean_dict = {}
    domain_datetime_asn_dict = {}

    for row in range(0, AS_count_df.shape[0]):

        datetime = AS_count_df['datetime'].iloc[row].strftime('%Y-%m-%d-%H-%M-%S')
        domain = AS_count_df['domain_name'].iloc[row]
        clean_records = AS_count_df['clean_records'].iloc[row]
        asn = AS_count_df['most_common_asn_num'].iloc[row]

        if domain not in domain_datetime_clean_dict.keys():

            domain_datetime_clean_dict[domain] = {datetime: clean_records}
            domain_datetime_asn_dict[domain] = {datetime: asn}

        else:

            domain_datetime_clean_dict[domain][datetime] = clean_records
            domain_datetime_asn_dict[domain][datetime] = asn

    indices_to_remove = []

    for row in list(input_df.index.values):

        datetime = input_df['batch_datetime'].iloc[row].strftime('%Y-%m-%d-%H-%M-%S')
        domain = input_df['test_url'].iloc[row]
        control_response_end_success = input_df['control_response_end_success'].iloc[row]
        anomaly = input_df['anomaly'].iloc[row]
        connect_error = input_df['connect_error'].iloc[row]
        test_query_successful = input_df['test_query_successful'].iloc[row]

        if country_code == "CN":
            gfwatch_censored = input_df['GFWatch_Censored'].iloc[row]

        remove_index = True

        asn_list = get_ASN_list(input_df, row)

        # Check if domain and datetime combination is in the ASN list
        if domain in domain_datetime_clean_dict.keys() and datetime in domain_datetime_clean_dict[domain].keys():

            # Check ASN list quality control
            if domain_datetime_clean_dict[domain][datetime] == "CLEAN":

                if control_response_end_success == True and anomaly == False and connect_error == False and test_query_successful == True:

                    if country_code != "CN" or gfwatch_censored == False:

                        if domain_datetime_asn_dict[domain][datetime] in asn_list:

                            remove_index = False # If there is a problem here, it is probably because the asn column is not recognized as numerical by Pandas

        if remove_index == True:

            indices_to_remove.append(row)

    print("Finished cleaning data")

    return indices_to_remove

# Create the training data from the original dataframe
def create_ML_ready_data(df, AS_count_df, save_filename):

    df = df.copy()

    df = df.reset_index(drop=True)  # Reset the index from 0 to length-1

    total_records_count = df.shape[0]

    dirty_indices = get_unclean_record_indices(df, AS_count_df, country_code)

    print("Clean indices determined")

    clean_indices = get_clean_indices(dirty_indices, total_records_count)

    records_removed_count = len(dirty_indices)

    mixed_training_index_list = np.arange(0, int(total_records_count * training_split_fraction))
    mixed_validation_index_list = np.arange(int(total_records_count * training_split_fraction) + 1,
                             int(total_records_count * (training_split_fraction + validation_split_fraction)))
    mixed_testing_index_list = np.arange(int(total_records_count * (training_split_fraction + validation_split_fraction)) + 1, total_records_count)

    mixed_index_dict = {"TRAINING": mixed_training_index_list,
                        "VALIDATION": mixed_validation_index_list,
                        "TESTING": mixed_testing_index_list}

    clean_training_index_list = intersection(mixed_training_index_list, clean_indices)
    clean_validation_index_list = intersection(mixed_validation_index_list, clean_indices)
    clean_testing_index_list = intersection(mixed_testing_index_list, clean_indices)

    clean_index_dict = {"TRAINING": clean_training_index_list,
                        "VALIDATION": clean_validation_index_list,
                        "TESTING": clean_testing_index_list}

    for month_year in [(7, 2021), (8, 2021), (9, 2021), (10, 2021), (11, 2021), (12, 2021), (1, 2022)]:
        cur_month = month_year[0]
        cur_year = month_year[1]

        monthly_df = df.loc[(df['batch_datetime'].dt.month == cur_month) & (df['batch_datetime'].dt.year == cur_year)]

        print("Month: " + str(cur_month) + " Year: " + str(cur_year))
        print("Data total probes: " + str(monthly_df.shape[0]), flush=True)

        date_folder_file_name = save_filename + str(cur_month) + "_" + str(cur_year) + "/"
        os.mkdir(date_folder_file_name)

        print("Month: " + str(cur_month) + " Year: " + str(cur_year))
        print("Data total probes: " + str(monthly_df.shape[0]), flush=True)

        # Save the truth columns for comparison
        for clean_moniker in ["Mixed", "Clean"]:

            if clean_moniker == "Mixed":

                index_dict = mixed_index_dict
                print("Mixed")

            elif clean_moniker == "Clean":

                index_dict = clean_index_dict
                print("Clean")

            for data_type in ["TRAINING", "VALIDATION", "TESTING"]:

                index_list = index_dict[data_type]

                subset_df = monthly_df.loc[intersection(index_list, list(monthly_df.index.values))] # Select only the rows we want

                print(data_type+ " data has length " +str(subset_df.shape[0]))

                subset_df['anomaly'].astype(int).to_csv(path_or_buf=date_folder_file_name+data_type+ "_" +clean_moniker+ "_targetFeature_anomaly.csv", \
                    index=False) # Converts True and False to 1 and 0

                if country_code == 'CN':

                    subset_df['GFWatch_Censored'].to_csv(path_or_buf=date_folder_file_name+data_type+ "_" +clean_moniker+ "_targetFeature_GFWatch_Censored.csv", \
                            index=False)

                elif country_code == "US":

                    subset_df['Presumed_Censored'].to_csv(path_or_buf=date_folder_file_name+data_type+ "_" +clean_moniker+ "_targetFeature_Presumed_Censored.csv", \
                        index=False)

                else:

                    pass # No presumption of censorship for other countries

    # Drop the columns
    df.drop(['anomaly'], axis=1)

    if country_code == 'CN':

        df.drop(['GFWatch_Censored'], axis=1)

    elif country_code == "US":

        df.drop(['Presumed_Censored'], axis=1)

    else:

        pass # No presumption of censorship for other countries

    ml_ready_df, ohenc, scaler = create_ML_features(df.copy())

    # Save one-hot-encoder and scaler

    dump(ohenc, save_filename+"one_hot_encoder")
    dump(scaler, save_filename+"scaler")

    print("Transformation to ML Dataframe Completed")

    for month_year in [(7, 2021), (8, 2021), (9, 2021), (10, 2021), (11, 2021), (12, 2021), (1, 2022)]:
        cur_month = month_year[0]
        cur_year = month_year[1]

        monthly_df = ml_ready_df.loc[(df['batch_datetime'].dt.month == cur_month) & (df['batch_datetime'].dt.year == cur_year)]

        print("Month: " + str(cur_month) + " Year: " + str(cur_year))
        print("Data total probes: " + str(monthly_df.shape[0]), flush=True)

        date_folder_file_name = save_filename + str(cur_month) + "_" + str(cur_year) + "/"

        print("Month: " + str(cur_month) + " Year: " + str(cur_year))
        print("Data total probes: " + str(monthly_df.shape[0]), flush=True)

        for clean_moniker in ["Mixed", "Clean"]:

            if clean_moniker == "Mixed":

                index_dict = mixed_index_dict
                print("Mixed")

            elif clean_moniker == "Clean":

                index_dict = clean_index_dict
                print("Clean")

            for data_type in ["TRAINING", "VALIDATION", "TESTING"]:

                index_list = index_dict[data_type]

                subset_ml_ready_df = monthly_df.loc[intersection(index_list, list(monthly_df.index.values))] # Select only the rows we want

                print(data_type+ " data has length " +str(subset_ml_ready_df.shape[0]))

                # Here we do not reset the index so we can later check that the dataset does not overlap

                subset_ml_ready_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=date_folder_file_name+data_type+ "_" +clean_moniker+ "_descriptiveFeatures_fullDataset.gzip")

    return total_records_count, records_removed_count

home_folder = r"/home/jambrown/"  # TODO change this name for your file structure
country_name = "United States" # TODO change country code as required
country_code = "US" # TODO update country code as required

home_file_name = home_folder +"CP_Analysis/"

training_split_fraction = 0.8
validation_split_fraction = 0.1
testing_split_fraction = 1 - training_split_fraction - validation_split_fraction

intermediary_file_name = home_file_name +country_code+ "/ML_ready_dataframes_V2/"
aggregate_file_name = home_file_name +country_code+ "/raw_dataframe.gzip"

AS_count_table_filename = home_file_name + "max_asn_aggregate.gzip"

AS_count_df = pd.read_parquet(path=AS_count_table_filename, engine='pyarrow')

# If we have not already preprocessed the datasets
if country_code != "CN":

    original_df = pd.read_parquet(path=aggregate_file_name, engine='pyarrow')

    original_df_length = original_df.shape[0]

    valid_df = remove_invalid_records(original_df)

    valid_df_length = valid_df.shape[0]

    print("Total number of probes: " +str(original_df_length))
    print("Valid number of probes: " +str(valid_df_length))

if country_code == "CN":

    save_gfwatch_file_name = home_file_name +country_code+ "/GFWatch_Combined_Dataset.gzip"
    original_with_newColumn_df = pd.read_parquet(path=save_gfwatch_file_name, engine='pyarrow')

elif country_code == "US":

    original_with_newColumn_df = valid_df.copy() # change the reference
    original_with_newColumn_df['Presumed_Censored'] = 0  # All US records are presumed uncensored

else:

    original_with_newColumn_df = valid_df.copy()  # change the reference and don't add the new column

print("Saving truth column complete!", flush=True)

row_count = original_with_newColumn_df.shape[0]

# Calculate indices for dividing data into unique training, validation, and testing components

original_with_newColumn_df = original_with_newColumn_df.sample(frac=1, ignore_index=True) # Shuffle the dataframe - this is the last time this is done in the pipeline

original_with_newColumn_df = original_with_newColumn_df.reset_index(drop=True) # Reset the index from 0 to n-1

# Create clean datasets
total_records_count, records_removed_count \
    = create_ML_ready_data(original_with_newColumn_df, AS_count_df, save_filename=intermediary_file_name)

print("Program Complete")

# # Create clean training dataset
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
#                           index_list=training_index_list, save_filename=ml_ready_data_file_name, data_type="TRAINING")
#
# print("Training data (clean) total probes: " +str(total_records_count), flush=True)
# print("Training data (clean) probes removed: " +str(records_removed_count), flush=True)
#
# # Create mixed training set
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
#                            index_list=training_index_list, save_filename=ml_ready_data_file_name, data_type="TRAINING")
#
# print("Training data (mixed) total probes: " +str(total_records_count), flush=True)
# print("Training data (mixed) probes removed: " +str(records_removed_count), flush=True)

# # Create clean validation dataset
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
#                            index_list=validation_index_list, save_filename=ml_ready_data_file_name, data_type="VALIDATION")
#
# print("Validation data (clean) total probes: " +str(total_records_count), flush=True)
# print("Validation data (clean) probes removed: " +str(records_removed_count), flush=True)
#
# # Create mixed validation set
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
#                            index_list=validation_index_list, save_filename=ml_ready_data_file_name, data_type="VALIDATION")
#
# print("Validation data (mixed) total probes: " +str(total_records_count), flush=True)
# print("Validation data (mixed) probes removed: " +str(records_removed_count), flush=True)
#
# # Create clean testing dataset
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
#                            index_list=testing_index_list, save_filename=ml_ready_data_file_name, data_type="TESTING")
#
# print("Testing data (clean) total probes: " +str(total_records_count), flush=True)
# print("Testing data (clean) probes removed: " +str(records_removed_count), flush=True)
#
# # Create mixed testing set
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
#                            index_list=testing_index_list, save_filename=ml_ready_data_file_name, data_type="TESTING")
#
# print("Testing data (mixed) total probes: " +str(total_records_count), flush=True)
# print("Testing data (mixed) probes removed: " +str(records_removed_count), flush=True)


