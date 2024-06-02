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
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared

SEED = 1



parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--feature_csv', help='input feature data csv', default="./mof_features/racs_all_clean_q3_oxi_state_metal_one_node_only.csv", type=str,required=False)
parser.add_argument('--label_csv', help='input label data csv', default="./MOFs_oms/id_prop_oxo.csv", type=str,required=False)
parser.add_argument('--extra_label_csv', help='extra input label data csv', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save trained models', default="./output", type=str, required=False)
parser.add_argument('--prop', help='target variable', choices=['oxo','h'], type=str,required=False)
parser.add_argument('--qbc', type=int,required=False)
parser.add_argument('--kfold', help='enable k-fold cross-validation', action='store_true')
parser.add_argument('--full', help='use full data for training', action='store_true')
parser.add_argument('--algo', help='ml model to use', default="rf", type=str, choices=['rf','xgb', 'lgbm', 'gp'])
args =  parser.parse_args()
# config = configparser.ConfigParser()
# config.read('config.ini')


# fea_sel_cols = ["racs_bb-linker_connecting_prop-X_scope-2_propagg-diff_corragg-sum_bbagg-sum","racs_bb-nodes_prop-X_scope-1_propagg-diff_corragg-sum_bbagg-sum",\
#     "racs_bb-nodes_prop-z_scope-1_propagg-diff_corragg-sum_bbagg-sum","racs_bb-nodes_prop-z_scope-1_propagg-product_corragg-sum_bbagg-sum",\
#         "racs_bb-nodes_prop-X_scope-0_propagg-product_corragg-sum_bbagg-sum","racs_bb-nodes_prop-z_scope-0_propagg-product_corragg-sum_bbagg-sum","Metal_Mn"]

fea_sel_cols = ["racs_bb-nodes_prop-covalent_radius_scope-1_propagg-diff_corragg-sum_bbagg-sum", "racs_bb-nodes_prop-covalent_radius_scope-3_propagg-product_corragg-sum_bbagg-sum", \
    "racs_bb-nodes_prop-z_scope-2_propagg-product_corragg-sum_bbagg-sum", "racs_bb-nodes_prop-X_scope-0_propagg-product_corragg-sum_bbagg-sum", "racs_bb-nodes_prop-z_scope-0_propagg-product_corragg-sum_bbagg-sum",\
    "encoded_Metal", "Oxidation_Metal_Site_oximachine"]

def prepare_dataset(args, label_csv, predict=False):

    assert str(args.prop) in str(label_csv)
    df_fea = pd.read_csv(args.feature_csv, index_col = 0)
    fea_cols = [col for col in df_fea.columns if not col in ['sample', 'Metal_index', 'MOF Name']]
    if args.algo == 'gp':
        fea_cols = [fea for fea in fea_cols if fea in fea_sel_cols]
    df_label = pd.read_csv(label_csv)
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
    features = df_all[fea_cols].values
    ids = df_all.id.values
    label = None
    if not predict:
        label = df_all[label_col].values
    return ids, features,label


def dataset_split(ids, X, y, train_ratio=0.8, random=False, SEED=1):
    if random:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1-train_ratio, random_state=SEED)
    else:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1-train_ratio, shuffle=False)
    return  ids_train, ids_test, X_train, X_test, y_train, y_test

    


# Main function
def run_regressor(args, seed=1):

    ids, X, y = prepare_dataset(args, label_csv = args.label_csv)
    if args.extra_label_csv:
        ids_ext, X_ext, y_ext = prepare_dataset(args, label_csv = args.extra_label_csv)
    # Split the data into training and testing sets
    test_ratio = 0.2
    if args.kfold:
        pred_path = os.path.join(args.output_dir, f"{args.algo}_pred_{args.prop}_{len(y)}_idx_cv.csv")
        if os.path.exists(pred_path):
            os.remove(pred_path)
        # kf = KFold(n_splits = round(1/test_ratio), shuffle=True, random_state=seed + 10)
        kf = KFold(n_splits = round(1/test_ratio), shuffle=False)
        mads = []
        maes = []
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            _, test_ids = ids[train_index], ids[test_index]
            print(X_train.shape, X_test.shape)
            if args.extra_label_csv:
                X_train = np.concatenate((X_train, X_ext), axis=0)
                y_train = np.concatenate((y_train, y_ext), axis=0)
            logging.info(f"Started fitting to model for test + train: {len(X)} samples")
            if args.algo == 'rf':
                regression_model = RandomForestRegressor(n_jobs = 32, n_estimators=300, max_depth=12, random_state=seed) #LinearRegression()
            elif args.algo == 'xgb':
                regression_model = XGBRegressor(n_estimators=300, max_depth=12, eta=0.01, subsample=1.0, colsample_bytree=0.9)
                # regression_model = XGBRegressor(n_jobs = 8)
            elif args.algo == 'gp':
                # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
                # kernel = 1 * RationalQuadratic(length_scale=1.0, alpha=0.1)
                # kernel = 1 * RBF(length_scale=1.0) + 1 * Matern(length_scale=1.0, nu=1.5)
                # kernel = 1 * RBF(length_scale=1.0) * 1 * DotProduct(sigma_0=1.0)
                kernel = 1 * RationalQuadratic(length_scale=1.0, alpha=0.1) + 1 * RBF(length_scale=1.0)
                # kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                # kernel = 1 * DotProduct(sigma_0=1.0)

                regression_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=9)

            regression_model.fit(X_train, y_train)
            result = defaultdict(list)
            y_pred = regression_model.predict(X_train)
            mae = mean_absolute_error(y_train, y_pred)
            logging.info(f"{args.prop}: Train mean_absolute_error: {mae}")

            y_pred = regression_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            logging.info(f"{args.prop}: Test mean_absolute_error: {mae}")
            maes.append(mae)
            mad = mean_absolute_error(y_test, [np.mean(y_train)]*len(y_test))
            mads.append(mad)
            
            df_pred = pd.DataFrame({'ids_test':test_ids, 'labels': y_test, 'predictions': y_pred})
            print(f"MAD:{mad}")

            if not os.path.exists(pred_path):
                df_pred.to_csv(pred_path)
            else:
                predictions_df = pd.read_csv(pred_path, index_col = 0)
                predictions_df = pd.concat([predictions_df, df_pred])
                predictions_df.to_csv(pred_path)
        logging.info(f"mean MAD for test set: {np.mean(mads)}")
        logging.info(f"mean MAE for test set: {np.mean(maes)}")
    else:
        if args.full:
            ids_train, X_train, y_train = ids, X, y
            ids_test, X_test, _ = prepare_dataset(args, label_csv = args.label_csv, predict=True)
        else:
            ids_train, ids_test, X_train, X_test, y_train, y_test = dataset_split(ids, X, y, train_ratio=1 - test_ratio)
        if args.qbc:
            assert args.full == True
            X_train, y_train = resample(X_train, y_train, random_state=seed)

        # # Initialize and fit a linear regression model
        logging.info(f"Started fitting to model for test + train: {len(X)} samples")
        if args.algo == 'rf':
            regression_model = RandomForestRegressor(n_jobs = 32, n_estimators=300, max_depth=12, random_state=seed, \
                max_features=0.3, bootstrap=True, max_samples = 0.5)
        regression_model.fit(X_train, y_train)
        # Evaluate the model
        result = defaultdict(list)
        y_pred = regression_model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        logging.info(f"{args.prop}: Train mean_absolute_error: {mae}")
        result['prop'].append(args.prop)
        result['Train_mae'].append(mae)
        result['Train_mad'].append(np.mean(np.abs(y_train - np.mean(y_train))))
        y_pred = regression_model.predict(X_test)
        if args.full:
            df_pred = pd.DataFrame({'ids_test':ids_test, f'predictions_seed_{seed}': y_pred})
            # df_pred.to_csv(os.path.join(args.output_dir, f"rf_pred_{args.prop}_{len(y)}_idx_iter_{seed}.csv"))
        else:
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            logging.info(f"{args.prop}: Test mean_absolute_error: {mae}")
            result['Test_mae'].append(mae)
            result['Test_mad'].append(mean_absolute_error(y_test, [np.mean(y_train)]*len(y_test)))
            df_pred = pd.DataFrame({'ids_test':ids_test, 'labels': y_test, 'predictions': y_pred})
            df_pred.to_csv(os.path.join(args.output_dir, f"{args.algo}_pred_{args.prop}_{len(y)}_idx.csv"))
            print(result)
            df_rst = pd.DataFrame.from_dict(result)
            df_rst.to_csv(os.path.join(args.output_dir, f"{args.algo}_rst_{args.prop}_{len(y)}_idx.csv"))
            # Predict using the test set
        return df_pred

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    if args.qbc:
        dfs = []
        for i in range(args.qbc):
            df_pred = run_regressor(args, seed=i)
            dfs.append(df_pred)

        # check if orders are the same
        assert dfs[0]['ids_test'].equals(dfs[3]['ids_test'])

        df_all = pd.concat(dfs, axis=1)
        df_all = df_all.loc[:,~df_all.columns.duplicated()]
        row_variances = df_all.drop('ids_test', axis=1).var(axis=1)
        df_all['row_variances'] = row_variances
        df_all['row_std'] = np.sqrt(row_variances)
        df_sorted = df_all.sort_values(by='row_variances', ascending=False)
        # df_sorted.to_csv(os.path.join(args.output_dir, f"{args.algo}_pred_{args.prop}_idx_iter_all_var_query_2.csv"))
        
    else:
        df_pred = run_regressor(args)
    
