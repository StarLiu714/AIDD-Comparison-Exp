# pip install deepchem rdkit pandas xgboost==1.7.6


import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from deepchem.feat import CircularFingerprint
import json
from itertools import product
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as CircularFingerprint


# Global: Path to the parameters file
PARAMS_JSON_PATH = 'parameters_rf.json'
MODEL_NAME = 'RandForest'


def process_dataset(
        dataset_file, dataset_idx, params, smile_str, seed, kfold=5, model_name=MODEL_NAME
        ):
    score_log_path = f'Benchmark/score_{model_name}_{dataset_idx}.log'
    params['dataset'] = [dataset_file]
    params['seed'] = [seed]

    with open(score_log_path, 'a') as log_file:
        log_file.write("\n----------------------------------------------------------------\n")
        log_file.write(f"Seed: {seed}, \n\n")

    # Load dataset
    data = pd.read_csv(dataset_file)
    # Split dataset
    train_valid_data, test_data = train_test_split(data, test_size=0.1, random_state=seed)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    combinations = list(product(*params.values()))
    param_keys = list(params.keys())
    best_r2, best_rmse, best_params, best_model = -float('inf'), float('inf'), combinations[0], None
    metrics = [
        dc.metrics.Metric(dc.metrics.pearson_r2_score),
        dc.metrics.Metric(dc.metrics.rms_score)
        ]

    for values in combinations:
        # Assuming the Random Forest parameters are included in the values, adjust according to your parameter setup
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap = values[:6]
        fold_results = []

        for fold, (train_index, valid_index) in enumerate(kf.split(train_valid_data)):
            print(f"Processing fold {fold + 1} with seed {seed} and params {values}...")
            train_data, valid_data = train_valid_data.iloc[train_index], train_valid_data.iloc[valid_index]

            # Example of feature generation with CircularFingerprint, adjust as needed
            featurizer = CircularFingerprint(size=2048)
            X_train = featurizer.featurize(train_data[smile_str].values)
            X_valid = featurizer.featurize(valid_data[smile_str].values)
            X_test = featurizer.featurize(test_data[smile_str].values)

            y_train = train_data['TARGET'].values
            y_valid = valid_data['TARGET'].values
            y_test = test_data['TARGET'].values

            train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
            val_dataset = dc.data.NumpyDataset(X=X_valid, y=y_valid)
            test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

            transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
            for dataset in [train_dataset, val_dataset, test_dataset]:
                for transformer in transformers:
                    dataset = transformer.transform(dataset)

            rf_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, 
                bootstrap=bootstrap
            )
            dc_model = dc.models.SklearnModel(
                rf_model, model_dir=f"RF_model_{seed}_{fold}"
                )
            dc_model.fit(train_dataset)

            results = dc_model.evaluate(val_dataset, metrics)
            fold_results.append((results['pearson_r2_score'], results['rms_score']))

            with open(score_log_path, 'a') as log_file:
                log_file.write(f"Fold {fold+1}: R2: {fold_results[fold][0]}, RMSE: {fold_results[fold][1]}\n")

        avg_r2 = np.mean([score[0] for score in fold_results])
        avg_rmse = np.mean([score[1] for score in fold_results])
        if avg_r2 > best_r2 or (avg_r2 == best_r2 and avg_rmse < best_rmse):
            best_r2, best_rmse = avg_r2, avg_rmse
            best_params = values
            best_model = dc_model

        with open(score_log_path, 'a') as log_file:
            log_file.write(f"Result:\n")
            log_file.write(f"\n".join(f"{k}: {v}" for k, v in zip(param_keys, values)))
            log_file.write(f"\nAvg R2: {avg_r2}, Avg RMSE: {avg_rmse}\n\n")

    test_results = best_model.evaluate(test_dataset, metrics)
    test_r2 = test_results['pearson_r2_score']
    test_rmse = test_results['rms_score']

    with open(score_log_path, 'a') as log_file:
        log_file.write("\n----------------------------------------------------------------\n")
        log_file.write(f"Best R2: {best_r2}, Best Validation RMSE: {best_rmse}\n")
        log_file.write("Best Params: " + ", ".join(f"{k}: {v}" for k, v in zip(param_keys, best_params)) + "\n")
        log_file.write(f"Test R2: {test_r2}, Test RMSE: {test_rmse}\n")

    return test_r2, test_rmse


if __name__ == "__main__":
    with open(PARAMS_JSON_PATH, 'r') as file:
        params = json.load(file)

    smile_str = params['smile_str'][0]

    for idx, dataset_file in enumerate(params['dataset']):
        score_log_path = f'Benchmark/score_{MODEL_NAME}_{idx}.log'
        with open(score_log_path, 'a') as log_file:
            log_file.write("----------------------------------------------------------------\n")
            log_file.write(f"Dataset: {dataset_file}\n")
        with open(PARAMS_JSON_PATH, 'r') as file:
            params = json.load(file)
        test_results = []
        for seed in params['seed']:
            test_r2, test_rmse = process_dataset(
                dataset_file, idx, params, smile_str, seed, kfold=5
                )
            test_results.append([test_r2, test_rmse])
        avg_r2, avg_rmse = np.mean([test_result[0] for test_result in test_results]), np.mean([test_result[1] for test_result in test_results])
        with open(score_log_path, 'a') as log_file:
            log_file.write("\n----------------------------------------------------------------\n----------------------------------------------------------------\n")
            log_file.write(f"{len(test_results)} Seeds Avg: \nR2: {avg_r2}, RMSE: {avg_rmse}")

