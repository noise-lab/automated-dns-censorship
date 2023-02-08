def round_list(old_list):

    new_list = []

    for index in range(0, len(old_list)):

        new_list.append(round(old_list[index], 0))

    return new_list

def dict_to_key_value_list(old_dt):

    new_dt ={}

    for key in old_dt.keys():
        new_dt[key] = [old_dt[key]]

    return new_dt

# In sci-kit learn, the values are 1 for inliers and -1 for outliers.
# This method converts the list to 0 for inliers and 1 for outliers
def convert_target_features(input_list):

    output_list = []

    for index in range(0, len(input_list)):

        if input_list[index] == 1:
            output_list.append(0)

        elif input_list[index] == -1:
            output_list.append(1)

    return output_list

def calculate_counts(prediction_list, truth_list):

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    for index in range(0, len(prediction_list)):

        if prediction_list[index] == 0 and truth_list[index] == 0:

            tn_count += 1

        elif prediction_list[index] == 0 and truth_list[index] == 1:

            fn_count += 1

        elif prediction_list[index] == 1 and truth_list[index] == 0:

            fp_count += 1

        elif prediction_list[index] == 1 and truth_list[index] == 1:

            tp_count += 1

    assert(sum([tp_count, tn_count, fp_count, fn_count]) == len(prediction_list))

    return tp_count, tn_count, fp_count, fn_count
