import pandas as pd
import numpy as np
import json
import re

# Divide original JSON from Censored Planet into documents of a certain number of lines
def DivideJSON(ogJSONFile, writer_filepathName, writer_docName, linesPerDoc):

    reader = open(ogJSONFile, 'r')

    try:
        writer = open(writer_filepathName + str(0) +writer_docName, 'w')
        writer.write("[\n")

        curDoc = 0
        lines = 1
        for line in reader:

            if lines <= linesPerDoc and (curDoc != 0 or lines != 1):
                writer.writelines(",\n")

            if lines > linesPerDoc:
                writer.write("\n]")
                writer.close()
                print("Just Completed Document: " + str(curDoc))
                curDoc = curDoc + 1
                writer = open(writer_filepathName + str(curDoc) +writer_docName, 'w')
                writer.write("[\n")
                lines = 1

            writer.writelines(line.rstrip())

            lines = lines + 1

        try:
            writer.write("\n]")
            writer.close()
            print("Just Completed Document: " + str(curDoc))

        except:

            print("Final writer cannot be closed")

    finally:
        reader.close()

# Converts the string into Pandas-readable datetime
def format_censoredplanet_datetime(old_string):

    stringlist = re.split(pattern='(\.[0-9]*)', string=old_string, maxsplit=1) # Remove the nanoseconds

    if len(stringlist) < 2: # Corner case if timestamp is exactly x seconds (no nanoseconds)

        stringlist = re.split(pattern='(\s-\d\d\d\d)', string=old_string, maxsplit=1) # Remove the nanoseconds

        new_string = stringlist[0] + ".000000001"

    else:

        new_string = stringlist[0] + stringlist[1]

    converted_datetime = pd.to_datetime(arg=new_string, format='%Y-%m-%d %H:%M:%S.%f')

    return converted_datetime

# Converts the file name (without the file extension) into Pandas-readable datetime
def format_datetime_from_file_name(old_string):

    new_string = re.split(pattern='CP_Satellite-', string=old_string)[1] # Remove the nanoseconds

    converted_datetime = pd.to_datetime(arg=new_string, format='%Y-%m-%d-%H-%M-%S')

    return converted_datetime

# This method converts every JSON split file into a pandas dataframe containing all low-level features.
def json_to_df(filename, batch_dt_input):

    with open(filename, "r") as file:

        data = json.loads(file.read()) # Convert json to dictionary

        line_count = len(data)

        print("Line Count: " +str(line_count))

        MAX_IPs = 30

        # Define arrays
        batch_datetime = np.full(shape=line_count, dtype='datetime64[ns]', fill_value=batch_dt_input)
        average_matchrate = np.full(shape=line_count, dtype=np.float64(), fill_value=0)
        untagged_controls = np.full(shape=line_count, dtype=bool, fill_value=False)
        untagged_response = np.full(shape=line_count, dtype=bool, fill_value=False)
        passed_liveness = np.full(shape=line_count, dtype=bool, fill_value=False)
        connect_error = np.full(shape=line_count, dtype=bool, fill_value=False)
        in_control_group = np.full(shape=line_count, dtype=bool, fill_value=False)
        anomaly = np.full(shape=line_count, dtype=bool, fill_value=False)
        excluded = np.full(shape=line_count, dtype=bool, fill_value=False)
        excluded_is_CDN = np.full(shape=line_count, dtype=bool, fill_value=False)
        excluded_below_threshold = np.full(shape=line_count, dtype=bool, fill_value=False)
        vantage_point = np.full(shape=line_count, dtype=object, fill_value="null")
        test_url = np.full(shape=line_count, dtype=object, fill_value="null")
        start_time = np.full(shape=line_count, dtype='datetime64[ns]', fill_value='NaT')
        end_time = np.full(shape=line_count, dtype='datetime64[ns]', fill_value='NaT')
        delta_time = np.full(shape=line_count, dtype=np.float64(), fill_value=0)
        country_name = np.full(shape=line_count, dtype=object, fill_value="null")
        country_code = np.full(shape=line_count, dtype=object, fill_value="null")

        control_response_start_success = np.full(shape=line_count, dtype=bool, fill_value=True)
        control_response_end_success = np.full(shape=line_count, dtype=bool, fill_value=True)
        control_response_start_url = np.full(shape=line_count, dtype=object, fill_value="a.root-servers.net")
        control_response_start_has_type_a = np.full(shape=line_count, dtype=bool, fill_value=True)
        control_response_start_error = np.full(shape=line_count, dtype=object, fill_value="null")
        control_response_start_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=0)
        control_response_end_url = np.full(shape=line_count, dtype=object, fill_value="a.root-servers.net")
        control_response_end_has_type_a = np.full(shape=line_count, dtype=bool, fill_value=True)
        control_response_end_error = np.full(shape=line_count, dtype=object, fill_value="null")
        control_response_end_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=0)

        test_query_successful = np.full(shape=line_count, dtype=bool, fill_value=False)
        test_query_unsuccessful_attempts = np.full(shape=line_count, dtype=np.int64(), fill_value=0)

        test_noresponse_1_url = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_1_has_type_a = np.full(shape=line_count, dtype=np.int64(), fill_value=2)
        test_noresponse_1_error = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_1_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)

        test_noresponse_2_url = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_2_has_type_a = np.full(shape=line_count, dtype=np.int64(), fill_value=2)
        test_noresponse_2_error = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_2_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)

        test_noresponse_3_url = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_3_has_type_a = np.full(shape=line_count, dtype=np.int64(), fill_value=2)
        test_noresponse_3_error = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_3_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)

        test_noresponse_4_url = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_4_has_type_a = np.full(shape=line_count, dtype=np.int64(), fill_value=2)
        test_noresponse_4_error = np.full(shape=line_count, dtype=object, fill_value="not_used")
        test_noresponse_4_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)

        test_response_url = np.full(shape=line_count, dtype=object, fill_value="never_succeeded")
        test_response_has_type_a = np.full(shape=line_count, dtype=np.int64(), fill_value=2)
        test_response_error = np.full(shape=line_count, dtype=object, fill_value="never_succeeded")
        test_response_rcode = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)
        test_response_IP_count = np.full(shape=line_count, dtype=np.int64(), fill_value=-2)

        features_IP_list = []

        for ip_index in range(0, MAX_IPs):

            response_feature_dict = {
                "test_response_" + str(ip_index) + "_IP": np.full(shape=line_count, dtype=object, fill_value="n"),
                "test_response_" + str(ip_index) + "_http": np.full(shape=line_count, dtype=object, fill_value="n"),
                "test_response_" + str(ip_index) + "_cert": np.full(shape=line_count, dtype=object, fill_value="n"),
                "test_response_" + str(ip_index) + "_asnum": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_asname": np.full(shape=line_count, dtype=object, fill_value="n"),
                "test_response_" + str(ip_index) + "_IP_match": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_http_match": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_cert_match": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_asnum_match": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_asname_match": np.full(shape=line_count, dtype=np.int64(), fill_value=-2),
                "test_response_" + str(ip_index) + "_match_percentage":np.full(shape=line_count, dtype=np.float64(), fill_value=-2),
            }

            features_IP_list.append(response_feature_dict)

        i = 0 # Record number
        for dt in data:

            # Collect metadata on control query and test query attempts
            numResponses = len(dt["response"])
            loc_test_url = dt["test_url"]
            loc_test_query_unsuccessful_attempts = 0
            test_query_responses = [] # List containing dictionary of fields for each IP address returned with a successful test query

            # Obtain values from "response" field
            for response_index in range(numResponses):

                if response_index == 0 and dt["response"][response_index]["url"] != loc_test_url: # Start Control Response
                    control_response_start_url[i] = dt["response"][response_index]["url"]
                    control_response_start_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                    control_response_start_error[i] = dt["response"][response_index]["error"]
                    control_response_start_rcode[i] = dt["response"][response_index]["rcode"]
                    control_response_start_success[i] = False

                elif response_index == numResponses-1 and dt["response"][response_index]["url"] != loc_test_url: # End control response
                    control_response_end_url[i] = dt["response"][response_index]["url"]
                    control_response_end_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                    control_response_end_error[i] = dt["response"][response_index]["error"]
                    control_response_end_rcode[i] = dt["response"][response_index]["rcode"]
                    control_response_end_success[i] = False

                else:

                    if dt["response"][response_index]["rcode"] == 0: # Successful test query
                        test_query_successful[i] = True
                        test_response_url[i] = dt["response"][response_index]["url"]
                        test_response_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                        test_response_error[i] = dt["response"][response_index]["error"]
                        test_response_rcode[i] = dt["response"][response_index]["rcode"]

                        loc_test_response_IP_count = len(dt["response"][response_index]["response"])
                        test_response_IP_count[i] = loc_test_response_IP_count

                        if loc_test_response_IP_count > 0:

                            ip_index = 0
                            for ip in dt["response"][response_index]["response"].keys():

                                if ip_index >= MAX_IPs:
                                    break # Ensures that max number of IP response columns is not exceeded

                                matched_list = dt["response"][response_index]["response"][ip]["matched"]

                                ip_match = 0
                                http_match = 0
                                cert_match = 0
                                asname_match = 0
                                asnum_match = 0

                                if matched_list == None: # Nothing matches
                                    pass

                                elif "no_tags" in matched_list:

                                    if "ip" in matched_list:
                                        ip_match = 1

                                else:

                                    if "ip" in matched_list:
                                        ip_match = 1

                                    if "http" in matched_list:
                                        http_match = 1

                                    if "cert" in matched_list:
                                        cert_match = 1

                                    if "asnum" in matched_list:
                                        asnum_match = 1

                                    if "asname" in matched_list:
                                        asname_match = 1

                                if matched_list == None or matched_list == ["no_tags"]:
                                    match_percentage = 0

                                else:
                                    match_percentage = dt["confidence"]["matches"][ip_index]

                                features_IP_list[ip_index]["test_response_" +str(ip_index)+"_IP"][i] = ip
                                features_IP_list[ip_index]["test_response_" +str(ip_index)+"_http"][i] = dt["response"][response_index]["response"][ip]["http"]
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_cert"][i] = dt["response"][response_index]["response"][ip]["cert"]
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_asnum"][i] = dt["response"][response_index]["response"][ip]["asnum"]
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_asname"][i] = dt["response"][response_index]["response"][ip]["asname"]
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_IP_match"][i] = ip_match
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_http_match"][i] = http_match
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_cert_match"][i] = cert_match
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_asnum_match"][i] = asnum_match
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_asname_match"][i] = asname_match
                                features_IP_list[ip_index]["test_response_" + str(ip_index) + "_match_percentage"][i] = match_percentage

                                test_query_responses.append(response_feature_dict)
                                ip_index = ip_index + 1

                    else: # Unsuccessful test query
                        loc_test_query_unsuccessful_attempts = loc_test_query_unsuccessful_attempts+1

                        if loc_test_query_unsuccessful_attempts == 1:
                            test_noresponse_1_url[i] = dt["response"][response_index]["url"]
                            test_noresponse_1_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                            test_noresponse_1_error[i] = dt["response"][response_index]["error"]
                            test_noresponse_1_rcode[i] = dt["response"][response_index]["rcode"]

                        elif loc_test_query_unsuccessful_attempts == 2:
                            test_noresponse_2_url[i] = dt["response"][response_index]["url"]
                            test_noresponse_2_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                            test_noresponse_2_error[i] = dt["response"][response_index]["error"]
                            test_noresponse_2_rcode[i] = dt["response"][response_index]["rcode"]

                        elif loc_test_query_unsuccessful_attempts == 3:
                            test_noresponse_3_url[i] = dt["response"][response_index]["url"]
                            test_noresponse_3_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                            test_noresponse_3_error[i] = dt["response"][response_index]["error"]
                            test_noresponse_3_rcode[i] = dt["response"][response_index]["rcode"]

                        elif loc_test_query_unsuccessful_attempts == 4:
                            test_noresponse_4_url[i] = dt["response"][response_index]["url"]
                            test_noresponse_4_has_type_a[i] = dt["response"][response_index]["has_type_a"]
                            test_noresponse_4_error[i] = dt["response"][response_index]["error"]
                            test_noresponse_4_rcode[i] = dt["response"][response_index]["rcode"]

            test_query_unsuccessful_attempts[i] = loc_test_query_unsuccessful_attempts

            # Fill data columns for every record

            # batch_datetime[i] = batch_dt_input not necessary because the batch datetime is the same for all records
            average_matchrate[i] = dt["confidence"]["average"]
            untagged_controls[i] = dt["confidence"]["untagged_controls"]
            untagged_response[i] = dt["confidence"]["untagged_response"]
            passed_liveness[i] = dt["passed_liveness"]
            connect_error[i] = dt["connect_error"]
            in_control_group[i] = dt["in_control_group"]
            anomaly[i] = dt["anomaly"]
            excluded[i] = dt["excluded"]
            excluded_is_CDN[i] = (r"u'is_CDN" in dt["exclude_reason"])
            excluded_below_threshold[i] = (r"u'domain_below_threshold" in dt["exclude_reason"])
            vantage_point[i] = dt["vp"]
            test_url[i] = dt["test_url"]
            start_time[i] = format_censoredplanet_datetime(dt["start_time"])
            end_time[i] = format_censoredplanet_datetime(dt["end_time"])
            delta_time[i] = (format_censoredplanet_datetime(dt["end_time"]) - format_censoredplanet_datetime(dt["start_time"])).total_seconds()
            country_name[i] = dt["location"]["country_name"]
            country_code[i] = dt["location"]["country_code"]

            i = i + 1 # Increase the record number

    # Create the dataframe
    # Object times are converted to pandas StringDtypes to improve efficiency
    df_dict = {
        "batch_datetime": batch_datetime,
        "average_matchrate": average_matchrate,
        "untagged_controls": untagged_controls,
        "untagged_response": untagged_response,
        "passed_liveness": passed_liveness,
        "connect_error": connect_error,
        "in_control_group": in_control_group,
        "anomaly": anomaly,
        "excluded": excluded,
        "excluded_is_CDN": excluded_is_CDN,
        "excluded_below_threshold": excluded_below_threshold,
        "vantage_point": pd.Series(vantage_point, dtype=pd.StringDtype()),
        "test_url": pd.Series(test_url, dtype=pd.StringDtype()),
        "start_time": start_time,
        "end_time": end_time,
        "delta_time": delta_time,
        "country_name": pd.Series(country_name, dtype=pd.StringDtype()),
        "country_code": pd.Series(country_code, dtype=pd.StringDtype()),
        "control_response_start_success": control_response_start_success,
        "control_response_end_success": control_response_end_success,
        "control_response_start_url": pd.Series(control_response_start_url, dtype=pd.StringDtype()),
        "control_response_start_has_type_a": control_response_start_has_type_a,
        "control_response_start_error": pd.Series(control_response_start_error, dtype=pd.StringDtype()),
        "control_response_start_rcode": control_response_start_rcode,
        "control_response_end_url": pd.Series(control_response_end_url, dtype=pd.StringDtype()),
        "control_response_end_has_type_a": control_response_end_has_type_a,
        "control_response_end_error": pd.Series(control_response_end_error, dtype=pd.StringDtype()),
        "control_response_end_rcode": control_response_end_rcode,
        "test_query_successful": test_query_successful,
        "test_query_unsuccessful_attempts": test_query_unsuccessful_attempts,
        "test_noresponse_1_url": pd.Series(test_noresponse_1_url, dtype=pd.StringDtype()),
        "test_noresponse_1_has_type_a": test_noresponse_1_has_type_a,
        "test_noresponse_1_error": pd.Series(test_noresponse_1_error, dtype=pd.StringDtype()),
        "test_noresponse_1_rcode": test_noresponse_1_rcode,
        "test_noresponse_2_url": pd.Series(test_noresponse_2_url, dtype=pd.StringDtype()),
        "test_noresponse_2_has_type_a": test_noresponse_2_has_type_a,
        "test_noresponse_2_error": pd.Series(test_noresponse_2_error, dtype=pd.StringDtype()),
        "test_noresponse_2_rcode": test_noresponse_2_rcode,
        "test_noresponse_3_url": pd.Series(test_noresponse_3_url, dtype=pd.StringDtype()),
        "test_noresponse_3_has_type_a": test_noresponse_3_has_type_a,
        "test_noresponse_3_error": pd.Series(test_noresponse_3_error, dtype=pd.StringDtype()),
        "test_noresponse_3_rcode": test_noresponse_3_rcode,
        "test_noresponse_4_url": pd.Series(test_noresponse_4_url, dtype=pd.StringDtype()),
        "test_noresponse_4_has_type_a": test_noresponse_4_has_type_a,
        "test_noresponse_4_error": pd.Series(test_noresponse_4_error, dtype=pd.StringDtype()),
        "test_noresponse_4_rcode": test_noresponse_4_rcode,
        "test_response_url": pd.Series(test_response_url, dtype=pd.StringDtype()),
        "test_response_has_type_a": test_response_has_type_a,
        "test_response_error": pd.Series(test_response_error, dtype=pd.StringDtype()),
        "test_response_rcode": test_response_rcode,
        "test_response_IP_count": test_response_IP_count,
    }

    # Add IP responses to dictionary
    for ip_index in range(0, MAX_IPs):

        df_dict.update(features_IP_list[ip_index])

    df = pd.DataFrame(df_dict)

    return df