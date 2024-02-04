#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import glob
import os
import pandas as pd
from collections import defaultdict
import configparser

SEED = 1



parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--feature_csv', help='input feature data csv', default="./features/matminer_racs_all_clean.csv", type=str,required=False)
parser.add_argument('--label_csv', help='input label data csv', default="./MOFs_oms/id_prop_oxo.csv", type=str,required=False)
parser.add_argument('--output_dir', help='path to the save trained models', default="./output", type=str, required=False)
parser.add_argument('--prop', help='target variable', choices=['oxo','h'], type=str,required=False)


args =  parser.parse_args()
# config = configparser.ConfigParser()
# config.read('config.ini')


def prepare_dataset(args):
    assert str(args.prop) in str(args.label_csv)
    df_fea = pd.read_csv(args.feature_csv, index_col = 0)
    fea_cols = [col for col in df_fea.columns if not col in ['sample', 'Metal_index', 'MOF Name']]
    df_label = pd.read_csv(args.label_csv)
    label_col = "Oxo Formation Energy" if args.prop == 'oxo' else "Hydrogen Affinity Energy"
    # print(df_fea.index)
    # print(df_fea[['sample', 'Metal_index']].info())

    df_all = df_label.merge(df_fea, how = 'inner', left_on = ['sample', 'Site'], 
                           right_on = ['sample', 'Metal_index'])

    df_all['id'] = df_all['sample'] +'_'+ df_all['Site'].astype(str)
    features = df_all[fea_cols].values
    ids = df_all.id.values
    label = df_all[label_col].values
    return ids, features,label


def dataset_split(ids, X, y, train_ratio=0.8, random=False, SEED=1):
    if random:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1-train_ratio, random_state=SEED)
    else:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1-train_ratio, shuffle=False)
    return  ids_train, ids_test, X_train, X_test, y_train, y_test

    



   

# Main function
def run_regressor(args):

    ids, X, y = prepare_dataset(args)
    # Split the data into training and testing sets
    ids_train, ids_test, X_train, X_test, y_train, y_test = dataset_split(ids, X, y, train_ratio=0.8)


    # # Initialize and fit a linear regression model
    logging.info(f"Started fitting to model for test + train: {len(X)} samples")
    nan_indices = np.isnan(X_train)
    if np.any(nan_indices):
        print("NaN values found  X_train at indices:", np.where(nan_indices)[0])
        print("Values:", X_train[nan_indices])
    else:
        print("No NaN values X_train in the array.")

    nan_indices = np.isnan(X)
    if np.any(nan_indices):
        print("NaN values found  X at indices:", np.where(nan_indices)[0])
        print("Values:", X_test[nan_indices])
    else:
        print("No NaN values X in the array.")
    regression_model = RandomForestRegressor(n_jobs = 32, n_estimators=300, max_depth=12) #LinearRegression()
    regression_model.fit(X_train, y_train)
    result = defaultdict(list)
    # Predict using the test set
    

    # Evaluate the model
    result = defaultdict(list)
    y_pred = regression_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    logging.info(f"{args.prop}: Train mean_absolute_error: {mae}")
    result['prop'].append(args.prop)
    result['Train_mae'].append(mae)
    result['Train_mad'].append(np.mean(np.abs(y_train - np.mean(y_train))))

    y_pred = regression_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"{args.prop}: Test mean_absolute_error: {mae}")


    result['Test_mae'].append(mae)
    result['Test_mad'].append(mean_absolute_error(y_test, [np.mean(y_test)]*len(y_test)))
    df_pred = pd.DataFrame({'ids_test':ids_test, 'labels': y_test, 'predictions': y_pred})
    df_pred.to_csv(os.path.join(args.output_dir, f"rf_pred_{args.prop}_{len(y)}_idx.csv"))
    print(result)
    df_rst = pd.DataFrame.from_dict(result)
    df_rst.to_csv(os.path.join(args.output_dir, f"rf_rst_{args.prop}_{len(y)}_idx.csv"))
    return df_rst

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    df_rst = run_regressor(args)
    
