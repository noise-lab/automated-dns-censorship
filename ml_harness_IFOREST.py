"""
This is the harness that runs the ML models.
Here we run linear_model.SGDOneClassSVM, a linear-approximation of the OCSVM
 """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from pyod.models.iforest import IForest
import time
from multiprocessing import Process
import os
from joblib import dump
from ml_harness_helper_methods import *
import shap

home_folder_name = r"/home/jambrown/" # TODO change this to your home folder
training_samples = 250000 # TODO change training sample count as required

model_name = "IFOREST_PyOD"
version = 10
version_filename = home_folder_name+ r"CP_Analysis/ML_Results/IFOREST/V" +str(version)+ "/"
sklearn_bool = False
model_set_list = [1, 2, 3, 4]
validation_samples = int(training_samples/10)
testing_samples = int(training_samples/10)
os.mkdir(version_filename)

# TODO change params as required
params_1 = {'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': None,
            'max_features': 1.0,
            'bootstrap': False,
            'n_jobs': 5,
            'behaviour': 'old',
            'random_state': 12345678,
            'verbose': 1,
}

params_2 = {'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': None,
            'max_features': 1.0,
            'bootstrap': False,
            'n_jobs': 5,
            'behaviour': 'old',
            'random_state': 12345678,
            'verbose': 1,
}

params_3 = {'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': None,
            'max_features': 1.0,
            'bootstrap': False,
            'n_jobs': 5,
            'behaviour': 'old',
            'random_state': 12345678,
            'verbose': 1,
}

params_4 = {'n_estimators': 200,
            'max_samples': 'auto',
            'contamination': None,
            'max_features': 0.25,
            'bootstrap': False,
            'n_jobs': 5,
            'behaviour': 'old',
            'random_state': 12345678,
            'verbose': 1,
}

params_5 = {'n_estimators': 200,
            'max_samples': 'auto',
            'contamination': None,
            'max_features': 0.25,
            'bootstrap': False,
            'n_jobs': 5,
            'behaviour': 'old',
            'random_state': 12345678,
            'verbose': 1,
}

def get_local_statistics_df(test_df, prediction_list, truth_list, model, save_file, anomaly_bool):

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
    if anomaly_bool == False and (1 in truth_list): # The AUC cannot be calculated unless there are positives in the truth column
                                                    # The AUC also cannot be calculated when we are using the Presumed_Censored column = 0 because TPR is constant (i.e. 0)

        fpr_list, tpr_list, _ = metrics.roc_curve(truth_list, model.decision_function(test_df), pos_label=1)
        roc_display = metrics.RocCurveDisplay(fpr=fpr_list, tpr=tpr_list)
        auc = metrics.roc_auc_score(truth_list, model.decision_function(test_df))

        prec, recall, _ = metrics.precision_recall_curve(truth_list, model.decision_function(test_df), pos_label=1)
        pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        plt.savefig(fname=save_file+r'curves.png')

    if anomaly_bool == True:
        suffix = "_CP_Model"
    else:
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

def get_results(validation_set_df, validation_truth_df, validation_comparison_df, validation_anomaly, comparison_anomaly, model, model_params, save_folder, t_num, time_elapsed):

    predicted_results_list = model.predict(validation_set_df.to_numpy())
    true_results_list = validation_truth_df.to_numpy()
    comparison_results_list = validation_comparison_df.to_numpy()

    if sklearn_bool == True:

        # If the model comes from scikit-learn, we need to convert the feature indicator into 0 for inlier and 1 for outlier
        predicted_results_list = convert_target_features(predicted_results_list)

    assert(-1 not in predicted_results_list) # Ensure the above statement was executed correctly

    # Get explanation values
    exp = shap.TreeExplainer(model)  # Explainer
    shap_values = exp.shap_values(validation_set_df.to_numpy())  # Calculate SHAP values

    feature_names = list(validation_set_df.columns)
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
    local_statistics_prediction_df, auc_prediction = get_local_statistics_df(validation_set_df, predicted_results_list, true_results_list, model, save_folder, validation_anomaly)
    # Create the columns in the dataframe associated with the Censored Planet Anomaly prediction
    local_statistics_comparison_df, _ = get_local_statistics_df(validation_set_df, comparison_results_list, true_results_list, model, save_folder, comparison_anomaly)
    # Create the columns in the dataframe from the model parameters
    local_statistics_parameters_df = pd.DataFrame.from_dict(dict_to_key_value_list(model_params))
    # Create the columns with the elapsed time and the AUC
    prefix_dict = {'Model Name': model_name, 'Version': version, 'Model Set': t_num, 'Model Run-Time': time_elapsed, 'AUC': auc_prediction}
    prefix_df = pd.DataFrame.from_dict(dict_to_key_value_list(prefix_dict))

    partial_df_list = [prefix_df, local_statistics_prediction_df, local_statistics_comparison_df, local_statistics_parameters_df]

    local_statistics_complete_df = pd.concat(partial_df_list, ignore_index=True, axis=1) # Concatenate the columns

    return local_statistics_complete_df

def run_ml_model(training_set_df, validation_set_df, validation_truth_df, validation_comparison_df, validation_anomaly, comparison_anomaly, model_params, save_folder, t_num):

    print("Begin training model T = " +str(t_num), flush=True)
    begin_time = time.time()
    clf = IForest(**model_params)
    clf.fit(training_set_df)
    time_elapsed = time.time() - begin_time
    print("End training model T= " +str(t_num), flush=True)
    print("Time Elapsed: " +str(time_elapsed)+ " seconds.", flush=True)
    dump(clf, save_folder +'model.joblib')

    print("Begin validating results for T= " + str(t_num), flush=True)

    results_df = get_results(validation_set_df, validation_truth_df, validation_comparison_df, validation_anomaly,
                             comparison_anomaly, clf, model_params, save_folder, t_num, time_elapsed)

    results_df.to_csv(save_folder +r"local_stats.csv", index=False)

    print("Finished validating and saving results for T= " +str(t_num), flush=True)

print("Begin machine learning harness")

home_file_name = home_folder_name+ r"CP_Analysis/"
ps = list()
for model_set in model_set_list:

    if model_set == 1:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/all_months_combined/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Mixed_descriptiveFeatures_fullDataset.gzip'
        training_contam_file_name = ml_ready_data_file_name +r'TRAINING_Mixed_targetFeature_GFWatch_Censored.csv'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_GFWatch_Censored.csv'
        comparison_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_anomaly.csv'
        validation_anomaly = False
        comparison_anomaly = True

        model_params = params_1

    elif model_set == 2:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/all_months_combined/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Mixed_descriptiveFeatures_fullDataset.gzip'
        training_contam_file_name = ml_ready_data_file_name + r'TRAINING_Clean_targetFeature_GFWatch_Censored.csv'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name + r'VALIDATION_Clean_targetFeature_GFWatch_Censored.csv'
        comparison_file_name = ml_ready_data_file_name + r'VALIDATION_Clean_targetFeature_anomaly.csv'
        validation_anomaly = False
        comparison_anomaly = True

        model_params = params_2

    elif model_set == 3:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/all_months_combined/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Mixed_descriptiveFeatures_fullDataset.gzip'
        training_contam_file_name = ml_ready_data_file_name + r'TRAINING_Mixed_targetFeature_GFWatch_Censored.csv'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_anomaly.csv'
        comparison_file_name = ml_ready_data_file_name + r'VALIDATION_Mixed_targetFeature_GFWatch_Censored.csv'
        validation_anomaly = True
        comparison_anomaly = False

        model_params = params_3

    elif model_set == 4:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/all_months_combined/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        training_contam_file_name = ml_ready_data_file_name + r'TRAINING_Clean_targetFeature_GFWatch_Censored.csv'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name + r'VALIDATION_Mixed_targetFeature_GFWatch_Censored.csv'
        comparison_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_anomaly.csv'
        validation_anomaly = False
        comparison_anomaly = True

        model_params = params_4

    elif model_set == 5:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes_V2/all_months_combined/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        training_contam_file_name = ml_ready_data_file_name + r'TRAINING_Clean_targetFeature_GFWatch_Censored.csv'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name + r'VALIDATION_Clean_targetFeature_GFWatch_Censored.csv'
        comparison_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_targetFeature_anomaly.csv'
        validation_anomaly = False
        comparison_anomaly = True

        model_params = params_4

    training_set_df = pd.read_parquet(path=training_set_file_name,
        engine='pyarrow').iloc[0:training_samples] # Only take the number of training samples specified

    training_contam_df = pd.read_csv(training_contam_file_name).iloc[0:training_samples] # Only take the number of training samples specified

    validation_set_df = pd.read_parquet(path=validation_set_file_name,
        engine='pyarrow').iloc[0:validation_samples]

    validation_truth_df = pd.read_csv(validation_truth_file_name).iloc[0:validation_samples]

    validation_comparison_df = pd.read_csv(comparison_file_name).iloc[0:validation_samples]

    if model_set == 4 or model_set == 5: # Set the contamination to the smallest possible value
        model_params["contamination"] = 0.001

    else:
        model_params["contamination"] = sum(np.squeeze(training_contam_df.to_numpy()))/training_samples

    print("Percent contamination in T" +str(model_set)+": " +str(model_params["contamination"]))

    save_folder = version_filename + r"T" +str(model_set) + r"/"

    os.mkdir(save_folder)

    p = Process(target=run_ml_model,
                args=(training_set_df, validation_set_df, validation_truth_df,
                      validation_comparison_df, validation_anomaly, comparison_anomaly, model_params, save_folder, model_set))
    ps.append(p)
    p.start()

for p in ps:
    p.join()

# Create csv containing all pertinent information about the models, their parameters, and the results
partial_df_list = []

for model_set in model_set_list:

    model_file_name = version_filename + r"T" +str(model_set) + r"/local_stats.csv"

    partial_df = pd.read_csv(model_file_name)

    partial_df_list.append(partial_df)

master_df = pd.concat(partial_df_list, ignore_index=True, axis=0)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

#Save to file
master_df.to_csv(path_or_buf=version_filename + r"statistics.csv", index=False)
print("End of machine learning harness for Version " +str(version) +".", flush=True)
