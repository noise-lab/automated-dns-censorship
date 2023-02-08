"""
This is the harness that runs the ML models.
Here we run linear_model.SGDOneClassSVM, a linear-approximation of the OCSVM
 """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost as xgb
import time
from multiprocessing import Process
import os
from ml_harness_helper_methods import *
import shap

def get_local_statistics_df(dvalid, prediction_list, truth_list, model, save_file):

    accuracy = metrics.accuracy_score(truth_list, prediction_list)
    f1_score = metrics.f1_score(truth_list, prediction_list)

    tp_count, tn_count, fp_count, fn_count = calculate_counts(prediction_list, truth_list)

    fpr = fp_count / (fp_count + tn_count)

    if (tp_count+fn_count > 0):
        tpr = tp_count / (tp_count + fn_count)
    else:
        tpr = -1

    if (fn_count + tp_count > 0):
        fnr = fn_count / (fn_count + tp_count)
    else:
        fnr = -1

    tnr = tn_count / (tn_count + fp_count)

    if (tp_count+fp_count > 0):
        precision = tp_count / (tp_count + fp_count)
    else:
        precision = -1

    # Create AUC and precision/recall curves
    auc = -1
    if (1 in truth_list): # The AUC cannot be calculated unless there are positives in the truth column
                                                    # The AUC also cannot be calculated when we are using the Presumed_Censored column = 0 because TPR is constant (i.e. 0)

        fpr_list, tpr_list, _ = metrics.roc_curve(truth_list, model.predict(dvalid), pos_label=1)
        roc_display = metrics.RocCurveDisplay(fpr=fpr_list, tpr=tpr_list)
        auc = metrics.roc_auc_score(truth_list, model.predict(dvalid))

        prec, recall, _ = metrics.precision_recall_curve(truth_list, round_list(model.predict(dvalid)), pos_label=1)
        pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        plt.savefig(fname=save_file+r'curves.png')

    suffix = "_New_Model"

    df_dict = {
        'Accuracy'+suffix: accuracy,
        'F1-Score'+suffix: f1_score,
        'TPR'+suffix: tpr,
        'FPR'+suffix: fpr,
        'Precision'+suffix: precision,
        'TNR'+suffix: tnr,
        'FNR'+suffix: fnr,
    }

    df = pd.DataFrame.from_dict(dict_to_key_value_list(df_dict))

    return df, auc

def get_results(dvalid, validation_target_df, model, model_params, save_folder, t_num, time_elapsed):

    predicted_results_list = round_list(model.predict(dvalid))
    true_results_list = validation_target_df.to_numpy()

    if sklearn_bool == True:

        # If the model comes from scikit-learn, we need to convert the feature indicator into 0 for inlier and 1 for outlier
        predicted_results_list = convert_target_features(predicted_results_list)

    assert(-1 not in predicted_results_list) # Ensure the above statement was executed correctly

    # Get explanation values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(dvalid)

    feature_names = dvalid.feature_names
    importance_dict = {}

    for index in range(0, len(shap_values[0])):

        mean_absolute_value = np.mean(np.abs(shap_values[:][index]))
        importance_dict[feature_names[index]] = mean_absolute_value

    sorted_features_list = sorted(importance_dict, key=importance_dict.__getitem__, reverse=True)
    sorted_num_list = sorted(importance_dict.values(), reverse=True)

    printout_dict = {"Features": sorted_features_list, "Importance": sorted_num_list}

    printout_df = pd.DataFrame.from_dict(printout_dict)

    printout_df.to_csv(path_or_buf=(save_folder + r"feature_importances.csv"), index=False)

    # Create the column for time elapsed

    # Create the columns in the dataframe associated with the model prediction
    local_statistics_prediction_df, auc_prediction = get_local_statistics_df(dvalid, predicted_results_list, true_results_list, model, save_folder)
    # Create the columns in the dataframe from the model parameters
    local_statistics_parameters_df = pd.DataFrame.from_dict(dict_to_key_value_list(model_params))
    # Create the columns with the elapsed time and the AUC
    prefix_dict = {'Model Name': model_name, 'Version': version, 'Model Set': t_num, 'Model Run-Time': time_elapsed, 'AUC': auc_prediction}
    prefix_df = pd.DataFrame.from_dict(dict_to_key_value_list(prefix_dict))

    partial_df_list = [prefix_df, local_statistics_prediction_df, local_statistics_parameters_df]

    local_statistics_complete_df = pd.concat(partial_df_list, ignore_index=True, axis=1) # Concatenate the columns

    return local_statistics_complete_df

def run_ml_model(training_descriptive_df, training_target_df, validation_descriptive_df, validation_target_df, model_params, save_folder, t_num):

    print("Begin training model T = " +str(t_num), flush=True)
    begin_time = time.time()
    dtrain = xgb.DMatrix(training_descriptive_df, label=training_target_df, feature_names=list(training_descriptive_df.columns))
    dvalid = xgb.DMatrix(validation_descriptive_df, label=validation_target_df, feature_names=list(validation_descriptive_df.columns))
    evals = [(dtrain, 'train') , (dvalid, 'valid')]
    model = xgb.train(model_params, dtrain, num_boost_round=10, evals=evals)
    time_elapsed = time.time() - begin_time
    print("End training model T= " +str(t_num), flush=True)
    print("Time Elapsed: " +str(time_elapsed)+ " seconds.", flush=True)
    model.save_model(save_folder +'model')

    print("Begin validating results for T= " + str(t_num), flush=True)

    results_df = get_results(dvalid, validation_target_df, model, model_params, save_folder, t_num, time_elapsed)

    results_df.to_csv(save_folder +r"local_stats.csv", index=False)

    print("Finished validating and saving results for T= " +str(t_num), flush=True)

# Use AUC for evaluation as opposed to curve (need to change disable_default_eval_metric and eval_metric)

def month_to_month_year(month):

    if month == 7:
        return 7, 2021

    if month == 8:
        return 8, 2021

    if month == 9:
        return 9, 2021

    if month == 10:
        return 10, 2021

    if month == 11:
        return 11, 2021

    if month == 12:
        return 12, 2021

    if month == 13:
        return 1, 2022

model_params = {'nthread': 20, # TODO change as required
            'predictor': 'cpu_predictor',
            'verbosity': 1,
            'booster': 'gbtree',
            'scale_pos_weight': None, # Is this being used? Rebalance dataset to 50-50 if unsure
            'disable_default_eval_metric': False,
            'eta': 0.3,
            'gamma': 0,
            'max_depth': 7,
            'min_child_weight': 0.8,
            'max_delta_step': 0,
            'subsample': 1,
            'colsample_bytree': 1,
            'lambda': 1,
            'alpha': 0,
            'tree_method': 'hist', # Try approx as well
            'objective': 'binary:logistic', # Try binary logistic if this does not work
            'eval_metric': 'error',
}

home_folder_name = r"/home/jambrown/" # TODO change this to your home folder
country_code = "CN" # TODO change country code as required
country_name = "China" # TODO change country name as required

version = "Satellite_Timing_Experiments"
home_file_name = home_folder_name +"CP_Analysis/"
version_filename = home_file_name + r"ML_Results/XGBOOST/V" +str(version)+ "/"

model_name = "XGBOOST"
sklearn_bool = False
os.mkdir(version_filename)

print("Begin machine learning harness")


t_name_list = []

for train_month_count in [1, 2, 3, 4, 5, 6]:

    for starting_train_month in range(7, 14-train_month_count):

        train_month_range = np.arange(starting_train_month, starting_train_month+train_month_count)

        for test_month in range(train_month_range[-1]+1, 14):

            t_name = "Training_" +str(train_month_range) +"_Test_"+ str(test_month)
            t_name_list.append(t_name)

            print(t_name)
            print("Training months: " +str(train_month_range))
            print("Test month: " +str(test_month))

            # Get training df

            training_descriptive_df_list = []
            training_target_df_list = []

            for training_month in train_month_range:

                og_train_month, og_train_year = month_to_month_year(training_month)

                training_descriptive_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/" +str(og_train_month)+ "_" +str(og_train_year)+ "/TRAINING_Mixed_descriptiveFeatures_fullDataset.gzip"
                partial_descriptive_df = pd.read_parquet(path=training_descriptive_file_name, engine='pyarrow')
                training_descriptive_df_list.append(partial_descriptive_df)

                training_target_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/" + str(og_train_month) + "_" + str(og_train_year) + "/TRAINING_Mixed_targetFeature_anomaly.csv"
                partial_target_df = pd.read_csv(training_target_file_name)
                training_target_df_list.append(partial_target_df)

            training_descriptive_df = pd.concat(training_descriptive_df_list, ignore_index=True, axis=0)
            training_target_df = pd.concat(training_target_df_list, ignore_index=True, axis=0)

            # Get target df
            og_test_month, og_test_year = month_to_month_year(test_month)
            test_descriptive_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/" +str(og_test_month)+ "_" +str(og_test_year)+ "/TESTING_Mixed_descriptiveFeatures_fullDataset.gzip"
            test_target_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/" +str(og_test_month)+ "_" +str(og_test_year)+ "/TESTING_Mixed_targetFeature_anomaly.csv"

            test_descriptive_df = pd.read_parquet(path=test_descriptive_file_name, engine='pyarrow')

            test_target_df = pd.read_csv(test_target_file_name)

            training_samples = training_descriptive_df.shape[0]

            print("Training samples: " +str(training_samples))

            contamination = sum(np.squeeze(training_target_df.to_numpy()))/training_samples

            print("Percent contamination: " +str(contamination))

            save_folder = version_filename + r"T" +str(t_name) + r"/"

            os.mkdir(save_folder)

            model_params['scale_pos_weight'] = (training_samples - sum(np.squeeze(training_target_df.to_numpy())))/sum(np.squeeze(training_target_df.to_numpy()))

            run_ml_model(training_descriptive_df, training_target_df, test_descriptive_df, test_target_df,  model_params, save_folder, t_name)

# Create csv containing all pertinent information about the models, their parameters, and the results
partial_df_list = []

for t_name in t_name_list:

    model_file_name = version_filename + r"T" +str(t_name) + r"/local_stats.csv"

    partial_df = pd.read_csv(model_file_name)

    partial_df_list.append(partial_df)

master_df = pd.concat(partial_df_list, ignore_index=True, axis=0)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

#Save to file
master_df.to_csv(path_or_buf=(version_filename + r"statistics.csv"), index=False)
print("End of machine learning harness for Version " +str(version) +".", flush=True)
