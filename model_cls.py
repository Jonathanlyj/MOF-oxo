import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score,recall_score, precision_score, roc_auc_score, roc_curve, fbeta_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import glob
import os
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared
from sklearn.metrics import ConfusionMatrixDisplay
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier

SEED = 1

parser = argparse.ArgumentParser(description='run ml classifier on dataset')
parser.add_argument('--feature_csv', help='input feature data csv', default="./mof_features/racs_all_clean_metal_one.csv", type=str, required=False)
parser.add_argument('--label_csv', help='input label data csv', type=str,required=False)
parser.add_argument('--train_label_csv', help='input label data csv', default="./MOFs_oms/id_class_01_2_3_shuffled_train.csv", type=str, required=False)
parser.add_argument('--test_label_csv', help='input label data csv', default="./MOFs_oms/id_class_01_2_3_shuffled_test.csv", type=str, required=False)
parser.add_argument('--output_dir', help='path to the save trained models', default="./classification", type=str, required=False)
parser.add_argument('--kfold', help='enable k-fold cross-validation', action='store_true')
parser.add_argument('--full', help='use full data for training', action='store_true')
parser.add_argument('--algo', help='ml model to use', default="rf", type=str, choices=['rf', 'xgb', 'gp', 'h2o', 'autosk', 'tpot'])
args = parser.parse_args()

fea_sel_cols = ['racs_bb-linker_connecting_prop-X_scope-2_propagg-diff_corragg-sum_bbagg-sum', 'racs_bb-linker_functional_prop-I_scope-3_propagg-product_corragg-sum_bbagg-sum', \
        'racs_bb-nodes_prop-X_scope-1_propagg-diff_corragg-sum_bbagg-sum', 'racs_bb-nodes_prop-X_scope-1_propagg-product_corragg-sum_bbagg-sum',\
        'racs_bb-nodes_prop-z_scope-1_propagg-diff_corragg-sum_bbagg-sum', 'racs_bb-nodes_prop-z_scope-1_propagg-product_corragg-sum_bbagg-sum',\
        'racs_bb-linker_functional_prop-I_scope-0_propagg-product_corragg-sum_bbagg-sum', 'racs_bb-nodes_prop-X_scope-0_propagg-product_corragg-sum_bbagg-sum',\
        'racs_bb-nodes_prop-z_scope-0_propagg-product_corragg-sum_bbagg-sum', 'encoded_Metal', "Oxidation_Metal_Site_oximachine"]

def prepare_dataset(args, label_csv, predict=False):
    df_fea = pd.read_csv(args.feature_csv, index_col=0)
    fea_cols = [col for col in df_fea.columns if col not in ['sample', 'Metal_index', 'MOF Name']]
    if args.algo == 'gp':
        fea_cols = [fea for fea in fea_cols if fea in fea_sel_cols]
    df_label = pd.read_csv(label_csv)
    label_col = "label"
    if not predict:
        df_all = df_label.merge(df_fea, how='inner', left_on=['sample', 'Site'], right_on=['sample', 'Metal_index'])
    else:
        df_all = df_fea.merge(df_label, how='left', right_on=['sample', 'Site'], left_on=['sample', 'Metal_index'], indicator=True)
        df_all = df_all[df_all['_merge'] == 'left_only']
        df_all = df_all.drop(columns='_merge')

    df_all['id'] = df_all['sample'] + '_' + df_all['Metal_index'].astype(str)
    features = df_all[fea_cols].values
    ids = df_all.id.values
    label = None
    if not predict:
        label = df_all[label_col].values
    return ids, features, label

def dataset_split(ids, X, y, train_ratio=0.8, random=False, SEED=1):
    if random:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1 - train_ratio, random_state=SEED)
    else:
        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=1 - train_ratio, shuffle=False)
    return ids_train, ids_test, X_train, X_test, y_train, y_test

def plot_metrics(y_test, y_pred, y_prob, figname="cls_", label_values = [0,1]):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values)
    cmap = LinearSegmentedColormap.from_list('red_gradient', ['#FFFFFF','#FA7921', '#E55934'])
    

    # Increase font size
    plt.rcParams.update({'font.size': 14})
    disp.plot(cmap=cmap)
    plt.savefig(f"{figname}_cm.png")

    # threshold = 0.5

    # Generate new predicted labels based on the threshold
    # y_pred_new = (y_prob >= threshold).astype(int)



    # Recalculate metrics

    accuracy = accuracy_score(y_test, y_pred)
    b_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)  # ROC AUC is threshold-independent
    beta = 2
    f_beta = fbeta_score(y_test, y_pred, beta=beta)

    # Print metrics
    # print(f' Accuracy: {accuracy}')
    # print(f'Balanced Accuracy: {b_accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    print(f'ROC AUC: {roc_auc}')
    # print(f'F score with beta {beta:.2f}: {f_beta}')
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{figname}_roc.png")

def run_classifier(args, seed=1):
    if args.label_csv:
        ids, X, y = prepare_dataset(args, label_csv=args.label_csv)
    test_ratio = 0.2
    if args.kfold:
        pred_path = os.path.join(args.output_dir, f"{args.algo}_pred_cls_{len(y)}_idx_cv.csv")
        if os.path.exists(pred_path):
            os.remove(pred_path)
        kf = KFold(n_splits=round(1 / test_ratio), shuffle=False)
        all_y_test = []
        all_y_pred = []
        all_y_prob = []

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            _, test_ids = ids[train_index], ids[test_index]
            
            if args.extra_label_csv:
                X_train = np.concatenate((X_train, X_ext), axis=0)
                y_train = np.concatenate((y_train, y_ext), axis=0)
            
            if args.algo == 'rf':
                classifier_model = RandomForestClassifier(n_jobs=32, n_estimators=300, max_depth=12, random_state=seed)
                # classifier_model = RandomForestClassifier(n_estimators=200, random_state=42,min_samples_leaf=1,min_samples_split=5)
            elif args.algo == 'xgb':
                classifier_model = XGBClassifier(n_estimators=300, max_depth=12, eta=0.01, subsample=1.0, colsample_bytree=0.9)
            elif args.algo == 'gp':
                kernel = 1 * RationalQuadratic(length_scale=1.0, alpha=0.1) + 1 * RBF(length_scale=1.0)
                classifier_model = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)


            # print(y_train)
            unique, counts = np.unique(y_train, return_counts=True)

            unique, counts = np.unique(y_test, return_counts=True)
            # print(unique, counts)
            
            classifier_model.fit(X_train, y_train)
            y_pred = classifier_model.predict(X_test)
            y_prob = classifier_model.predict_proba(X_test)
            
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob[:, 1])

        # Convert lists to numpy arrays for metric calculations
        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)
        label0 = f"dataset_{args.label_csv.split('/')[-1].split('.')[0].split('_')[2]}"
        label1 = f"dataset_{args.label_csv.split('/')[-1].split('.')[0].split('_')[3]}"
        fig_path = args.label_csv.split('/')[-1].split('.')[0]
        fig_path = f"{fig_path}_{args.algo}"
        fig_path = os.path.join(args.output_dir, fig_path)
        plot_metrics(all_y_test, all_y_pred, all_y_prob,fig_path,label_values=[label0, label1])
    else:
        if args.full:
            ids_train, X_train, y_train = ids, X, y
            ids_test, X_test, _ = prepare_dataset(args, label_csv=args.label_csv, predict=True)
        elif args.train_label_csv:
            ids_train, X_train, y_train = prepare_dataset(args, label_csv=args.train_label_csv)
            ids_test, X_test, y_test = prepare_dataset(args, label_csv=args.test_label_csv)
        else:
            ids_train, ids_test, X_train, X_test, y_train, y_test = dataset_split(ids, X, y, train_ratio=1 - test_ratio)
        
        if args.algo == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200, 300],  # Given value
                'max_depth': [None, 6, 10, 12],  # Examples of depths to test
                'bootstrap': [False, True],  # Similar to bootstrap
                'max_features': [0.5, 1],  
                'min_samples_leaf': [6, 10, 12], 
                'criterion': ["gini", "entropy"]
            }
            rf = RandomForestClassifier()
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
            # classifier_model = RandomForestClassifier(input_matrix, bootstrap=True, criterion=entropy, max_features=0.5, min_samples_leaf=12, min_samples_split=14, n_estimators=100
            # Fit the grid search to the data
            grid_search.fit(X_train, y_train)
            # Print the best parameters and the best score
            print("Best parameters found: ", grid_search.best_params_)
            print("Best ROC AUC found: ", grid_search.best_score_)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
        elif args.algo == 'xgb':
            param_distributions = {
                'n_estimators': [ 100, 200, 300],
                'max_depth': [ 3, 5, 7],
                'learning_rate': [ 0.01, 0.1, 0.2],
                'subsample': [ 0.6, 0.8, 1.0],
                'colsample_bytree': [ 0.6, 0.8, 1.0],
                'gamma': [ 0, 0.1, 0.2, 0.3],
                'reg_alpha': [ 0, 0.01, 0.1, 1],
                'reg_lambda': [ 0, 0.01, 0.1, 1],
                'scale_pos_weight': [ 1, 10, 25, 50]
            }

            # Create a base model
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc')

            # Instantiate the grid search model
            random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_distributions, n_iter=2000, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1)
            random_search.fit(X_train, y_train)
            # Print the best parameters and the best score
            print("Best parameters found: ", random_search.best_params_)
            print("Best ROC AUC found: ", random_search.best_score_)
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
        elif args.algo == 'gp':
            # kernel = 1 * RationalQuadratic(length_scale=1.0, alpha=0.1) + 1 * RBF(length_scale=1.0)
            # classifier_model = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100
            model = GaussianProcessClassifier()
            kernels = [
                1.0 * RBF(length_scale=1.0),
                1.0 * Matern(length_scale=1.0, nu=1.5),
                1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0),
                1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0),
                1.0 * DotProduct(sigma_0=1.0) ** 2
            ]
            param_grid = {
                'kernel': kernels,
                'n_restarts_optimizer': [0, 1, 2]
            }
            # Set up GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            # Best parameters and model
            print("Best ROC AUC found: ", grid_search.best_score_)
            print("Best Parameters:", grid_search.best_params_)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

        elif args.algo == 'tpot':
            classifier_model = TPOTClassifier(generations=5, population_size=20, cv=5, n_jobs=32, random_state=seed, verbosity=2, config_dict='TPOT sparse', scoring='roc_auc')
            classifier_model.fit(X_train, y_train)
            y_pred = classifier_model.predict(X_test)
            y_prob = classifier_model.predict_proba(X_test)[:, 1]
        elif args.algo == 'h2o':
            response_column = 'label'
            h2o.init()
            X_train_h2o = h2o.H2OFrame(X_train)
            y_train_h2o = h2o.H2OFrame(y_train, column_names=[response_column])
            y_train_h2o[response_column] = y_train_h2o[response_column].asfactor()
            # Combine X and y into a single frame
            train_h2o = X_train_h2o.cbind(y_train_h2o)
            # Set the response column as the last column

            train_h2o.set_names(list(X_train_h2o.names) + [response_column])

            # Specify the columns to be used for training
            x = train_h2o.columns[:-1]
            y = response_column

            # Run H2O AutoML with 5-fold cross-validation
            aml = H2OAutoML(max_runtime_secs=3600,  # Set a reasonable runtime
                            nfolds=5,
                            seed=1,
                            sort_metric='AUC')
            aml.train(x=x, y=y, training_frame=train_h2o)
            # Get the best model from the leaderboard
            classifier_model = aml.leader
            # Print the leaderboard
            print(aml.leaderboard)
            model_path = h2o.save_model(model=classifier_model, path=args.output_dir, force=True)
            print("Model saved to:", model_path)
            # classifier_model = h2o.load_model("/scratch/yll6162/MOF-oxo/classification/StackedEnsemble_BestOfFamily_7_AutoML_3_20240601_231929")
            # Convert X_test to H2OFrame
            # Get the cross-validation metrics for the best model


            # Get the base models
            base_models = classifier_model.base_models
            print("Base models in the stacked ensemble:")
            for model_id in base_models:
                model = h2o.get_model(model_id)
                print(f"Model ID: {model_id}, Model Type: {model.algo}, AUC: {model.auc()}")
            X_test_h2o = h2o.H2OFrame(X_test)
            pred = classifier_model.predict(X_test_h2o).as_data_frame()
            y_pred = pred['predict'].values.flatten()
            y_prob = pred['p1'].values 

        label0 = f"dataset_{args.train_label_csv.split('/')[-1].split('.')[0].split('_')[2]}"
        label1 = f"dataset_{args.train_label_csv.split('/')[-1].split('.')[0].split('_')[3]}"
        if args.label_csv:
            fig_path = args.label_csv.split('/')[-1].split('.')[0]
        else:
            fig_path = args.test_label_csv.split('/')[-1].split('.')[0]
        fig_path = f"{fig_path}_{args.algo}"
        fig_path = os.path.join(args.output_dir, fig_path)
        test_save_path = os.path.join(args.output_dir, f"{args.algo}_{label0}_{label1}_pred_cls_test_prob.csv")
        df = pd.DataFrame({'ids_test':ids_test, 'y_test': y_test,'y_prob': y_prob})
        df.to_csv(test_save_path, index=False)
        

        plot_metrics(y_test, y_pred, y_prob,fig_path,label_values=[label0, label1])
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    run_classifier(args)