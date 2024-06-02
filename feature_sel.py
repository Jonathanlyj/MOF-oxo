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
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from feature_engine.selection import RecursiveFeatureAddition
SEED = 1



parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--feature_csv', help='input feature data csv', default="./mof_features/racs_all_clean_q3_oxi_state_metal_one_node_only.csv", type=str,required=False)
parser.add_argument('--label_csv', help='input label data csv', default="./MOFs_oms/id_prop_oxo_3_NN_shuffled.csv", type=str,required=False)
parser.add_argument('--output_dir', help='path to the save selected features', default="./output", type=str, required=False)
parser.add_argument('--prop', help='target variable', choices=['oxo','h'], type=str,required=False)
parser.add_argument('--algo', help='ml model to use', default="rf", type=str, choices=['rf','gp', 'xgb'])
args =  parser.parse_args()
# config = configparser.ConfigParser()
# config.read('config.ini')



def prepare_dataset_sel(args, predict=False):

    assert str(args.prop) in str(args.label_csv)
    df_fea = pd.read_csv(args.feature_csv, index_col = 0)
    fea_cols = [col for col in df_fea.columns if not col in ['sample', 'Metal_index', 'MOF Name']]
    df_label = pd.read_csv(args.label_csv)
    label_col = "Oxo Formation Energy" if args.prop == 'oxo' else "Hydrogen Affinity Energy"
        # print(df_fea.index)
        # print(df_fea[['sample', 'Metal_index']].info())
    if not predict:
        df_all = df_label.merge(df_fea, how = 'inner', left_on = ['sample', 'Site'], 
                            right_on = ['sample', 'Metal_index'])
    else:
        df_all = df_fea.merge(df_label, how='left', right_on=['sample', 'Site'], 
                            left_on=['sample', 'Metal_index'], indicator=True)

        # Select rows where the merge indicator shows it's only in the left dataframe
        df_all = df_all[df_all['_merge'] == 'left_only']

        # Drop the merge indicator column
        df_all = df_all.drop(columns='_merge')
        
    df_all['id'] = df_all['sample'] +'_'+ df_all['Metal_index'].astype(str)
    features = df_all[fea_cols]
    ids = df_all.id.values
    label = None
    if not predict:
        label = df_all[label_col].values
    return ids, features,label



    


# Main function
def feature_selection(args, seed=1):

    ids, X, y = prepare_dataset_sel(args)
    # Split the data into training and testing sets
    ids_train, X_train, y_train = ids, X, y
    # ids_test, X_test, _ = prepare_dataset(args, predict=True)



    # # Initialize and fit a linear regression model
    logging.info(f"Started fitting to model for test + train: {len(X)} samples")
    regression_model = RandomForestRegressor(n_jobs = 32, n_estimators=300, max_depth=12, random_state=seed)

    # if args.algo == 'gp':
    #     kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    #     gaussian_process = GaussianProcessRegressor(
    #         kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
    #     )
    #     regression_model = gaussian_process
    tr = RecursiveFeatureAddition(estimator=regression_model, threshold=0.005, scoring="neg_mean_absolute_error", cv=5)
    Xt = tr.fit_transform(X_train, y_train)
    print(tr.get_feature_names_out())
    print(tr.performance_drifts_)
    return Xt

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    Xt = feature_selection(args)
    file_name = f"{args.prop}_{args.algo}_feature_sel_3.csv"
    if args.output_dir:
        file_name = os.path.join(args.output_dir, file_name)
    Xt.to_csv(file_name, index = None)
    
