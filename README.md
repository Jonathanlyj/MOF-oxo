# MOF-oxo


python model.py --feature_csv ./mof_features/racs_all_clean.csv --label_csv ./MOFs_oms/id_prop_oxo.csv --prop oxo


python model.py --feature_csv ./mof_features/racs_all_clean.csv --label_csv ./MOFs_oms/id_prop_oxo_1.csv --prop oxo --kfold --algo rf


python feature_sel.py --feature_csv ./mof_features/racs_all_clean.csv --label_csv ./MOFs_oms/id_prop_oxo_1.csv --prop oxo 