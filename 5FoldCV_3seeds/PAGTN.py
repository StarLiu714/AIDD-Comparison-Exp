


import deepchem as dc
from deepchem.models import PagtnModel
from deepchem.feat import PagtnMolGraphFeaturizer
import torch
import json
from itertools import product
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np


# Global: Path to the parameters file
MODEL_NAME = 'PAGTN'
PARAMS_JSON_PATH = f'parameters_{MODEL_NAME}.json'


def process_dataset(
        dataset_file, dataset_idx, params, smile_str, seed, device,
        kfold=5, model_name=MODEL_NAME):
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
    test_data.to_csv('test_dataset.csv', index=False)

    # Prepare for 5-fold cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    combinations = list(product(*params.values()))
    param_keys = list(params.keys())
    best_r2, best_rmse, best_params, best_model = -float('inf'), float('inf'), combinations[0], None
    metrics = [dc.metrics.Metric(dc.metrics.pearson_r2_score), dc.metrics.Metric(dc.metrics.rms_score)]

    for values in combinations:
        batch_size, learning_rate, embedding, _, _, n_tasks, epoch = values[:7]
        fold_results = []  # To store results for each fold

        for fold, (train_index, valid_index) in enumerate(kf.split(train_valid_data)):
            print(f"Processing fold {fold + 1} with seed {seed} and params {values}...")
            train_data, valid_data = train_valid_data.iloc[train_index], train_valid_data.iloc[valid_index]
            featurizer = PagtnMolGraphFeaturizer(max_length=5)
            X_train = featurizer.featurize(train_data[smile_str].values)
            X_valid = featurizer.featurize(valid_data[smile_str].values)
            X_test = featurizer.featurize(test_data[smile_str].values)

            y_train = train_data['TARGET'].values
            y_valid = valid_data['TARGET'].values
            y_test = test_data['TARGET'].values

            # 创建NumpyDataset
            train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
            val_dataset = dc.data.NumpyDataset(X=X_valid, y=y_valid)
            test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

            transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
            for dataset in [train_dataset, val_dataset, test_dataset]:
                for transformer in transformers:
                    dataset = transformer.transform(dataset)

            # Initialize model for regression
            model = PagtnModel(
                mode="regression", n_tasks=n_tasks, hidden_features=embedding,
                batch_size=batch_size, learning_rate=learning_rate
                )
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
                model.model.to(device)
            model.fit(train_dataset, nb_epoch=epoch)

            results = model.evaluate(val_dataset, metrics)
            fold_results.append((results['pearson_r2_score'], results['rms_score']))

        # Calculate average R2 and RMSE across folds
        avg_r2 = np.mean([score[0] for score in fold_results])
        avg_rmse = np.mean([score[1] for score in fold_results])
        if avg_r2 > best_r2 or (avg_r2 == best_r2 and avg_rmse < best_rmse):
            best_r2, best_rmse = avg_r2, avg_rmse
            best_params = values
            best_model = model

        with open(score_log_path, 'a') as log_file:
            log_file.write(f"Result:\n")
            log_file.write(f"\n".join(f"{k}: {v}" for k, v in zip(param_keys, values)))
            log_file.write(f"\nAvg R2: {avg_r2}, Avg RMSE: {avg_rmse}\n\n")

    # Evaluate on test set
    test_results = best_model.evaluate(test_dataset, metrics)
    test_r2 = test_results['pearson_r2_score']
    test_rmse = test_results['rms_score']

    # Log the best results
    with open(score_log_path, 'a') as log_file:
        log_file.write("\n----------------------------------------------------------------\n")
        log_file.write(f"Best R2: {best_r2}, Best Validation RMSE: {best_r2}\n")
        log_file.write("Best Params: " + ", ".join(f"{k}: {v}" for k, v in zip(param_keys, best_params)) + "\n")
        log_file.write(f"Test R2: {test_r2}, Test RMSE: {test_rmse}\n")

    return test_r2, test_rmse


# Main script
if __name__ == "__main__":
    with open(PARAMS_JSON_PATH, 'r') as file:
        params = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')

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
            test_r2, test_rmse = process_dataset(dataset_file, idx, params, smile_str, seed, device, kfold=5)
            test_results.append([test_r2, test_rmse])
        avg_r2, avg_rmse = np.mean([test_result[0] for test_result in test_results]), np.mean([test_result[1] for test_result in test_results])
        with open(score_log_path, 'a') as log_file:
            log_file.write("\n----------------------------------------------------------------\n----------------------------------------------------------------\n")
            log_file.write(f"{len(test_results)} Seeds Avg: \nR2: {avg_r2}, RMSE: {avg_rmse}")
