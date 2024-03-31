import deepchem as dc
from deepchem.models.torch_models.dmpnn import DMPNNModel
from deepchem.feat import DMPNNFeaturizer
import json
import torch
from itertools import product
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as CircularFingerprint


# Global: Path to the parameters file
MODEL_NAME = 'DMPNN'
PARAMS_JSON_PATH = f'parameters_{MODEL_NAME}.json'
K_FOLD = 3
test_csv_path = "Benchmark/SMILES12k/SMILES12k.csv"
test_data = pd.read_csv(test_csv_path)


def process_dataset(
        dataset_file, dataset_idx, params, smile_str, seed, device,
        kfold=5, model_name=MODEL_NAME
        ):
    score_log_path = f'Benchmark/score_{model_name}_{dataset_idx}.log'
    params['tr_val_dataset'] = [dataset_file]
    params['seed'] = [seed]

    with open(score_log_path, 'a') as log_file:
        log_file.write("\n----------------------------------------------------------------\n")
        log_file.write(f"Seed: {seed}, \n\n")

    # Load dataset
    featurizer = DMPNNFeaturizer()
    data = pd.read_csv(dataset_file)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    combinations = list(product(*params.values()))
    param_keys = list(params.keys())
    best_auc, best_accuracy, best_params, best_model = -float('inf'), -float('inf'), combinations[0], None
    metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score),
        dc.metrics.Metric(dc.metrics.accuracy_score)
        ]

    for values in combinations:
        # Unpack the Random Forest parameters
        batch_size, learning_rate, embedding, _, _, n_tasks, epoch = values[:7]
        fold_results = []

        for fold, (train_index, valid_index) in enumerate(kf.split(data)):
            print(f"Processing fold {fold + 1} with seed {seed} and params {values}...")
            train_data, valid_data = data.iloc[train_index], data.iloc[valid_index]

            # Feature generation with CircularFingerprint
            X_train = featurizer.featurize(train_data[smile_str].values)
            X_valid = featurizer.featurize(valid_data[smile_str].values)

            y_train = train_data['TARGET'].values
            y_valid = valid_data['TARGET'].values

            train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
            val_dataset = dc.data.NumpyDataset(X=X_valid, y=y_valid)

            transformers = [
                dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
            for transformer in transformers:
                train_dataset = transformer.transform(train_dataset)
                val_dataset = transformer.transform(val_dataset)

            model = DMPNNModel(
                mode='classification', n_tasks=n_tasks, n_classes=2,
                batch_size=batch_size, learning_rate=learning_rate
                )
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
                model.model.to(device)
            model.fit(train_dataset, nb_epoch=epoch)

            results = model.evaluate(val_dataset, metrics)
            fold_results.append((results[metrics[0].name], results[metrics[1].name]))

            with open(score_log_path, 'a') as log_file:
                log_file.write(f"Fold {fold+1}: AUROC: {results[metrics[0].name]}, Accuracy: {results[metrics[1].name]}\n")

        avg_auc = np.mean([score[0] for score in fold_results])
        avg_accuracy = np.mean([score[1] for score in fold_results])
        if avg_auc > best_auc or (avg_auc == best_auc and avg_accuracy > best_accuracy):
            best_auc, best_accuracy = avg_auc, avg_accuracy
            best_params = values
            best_model = model

        with open(score_log_path, 'a') as log_file:
            log_file.write(f"Result:\n")
            log_file.write(f"\n".join(f"{k}: {v}" for k, v in zip(param_keys, values)))
            log_file.write(f"\nAvg AUROC: {avg_auc}, Avg Accuracy: {avg_accuracy}\n\n")

    with open(score_log_path, 'a') as log_file:
        log_file.write("\n----------------------------------------------------------------\n")
        log_file.write(f"\nCurrent Dataset: {dataset_file}\n")
        log_file.write(f"Best Result:\n")
        log_file.write(f"\n".join(f"{k}: {v}" for k, v in zip(param_keys, best_params)))
        log_file.write(f"\nBest Avg AUROC: {best_auc}, Accuracy: {best_accuracy}\n\n")

    # Retrain the best model on the entire training set
    print(f"Retraining the best model on the entire training set...")
    X_tr_val = featurizer.featurize(data[smile_str].values)
    X_test = featurizer.featurize(test_data[smile_str].values)
    y_tr_val = data['TARGET'].values
    y_test = test_data['TARGET'].values

    tr_val_dataset = dc.data.NumpyDataset(X=X_tr_val, y=y_tr_val)
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)
    transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    for dataset in [tr_val_dataset, test_dataset]:
        for transformer in transformers:
            dataset = transformer.transform(dataset)

    # Predicting test set with the best model
    best_model.fit(tr_val_dataset, nb_epoch=epoch)
    print(f"Predicting test set with best model...")
    test_prob_predictions = best_model.predict(test_dataset)
    test_predictions = np.argmax(test_prob_predictions, axis=1)
    test_data['Prob_Predictions'] = test_prob_predictions[:, 0]
    test_data['Prediction'] = test_predictions
    test_data.to_csv(f"{model_name}_test_predictions_{seed}.csv", index=False)

    return best_auc, best_accuracy


if __name__ == "__main__":
    with open(PARAMS_JSON_PATH, 'r') as file:
        params = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')

    smile_str = params['smile_str'][0]

    for idx, dataset_file in enumerate(params['tr_val_dataset']):
        score_log_path = f'Benchmark/score_{MODEL_NAME}_{idx}.log'
        with open(score_log_path, 'a') as log_file:
            log_file.write("----------------------------------------------------------------\n")
            log_file.write(f"Dataset: {dataset_file}\n")
        with open(PARAMS_JSON_PATH, 'r') as file:
            params = json.load(file)
        test_results = []
        for seed in params['seed']:
            best_auc, best_accuracy = process_dataset(
                dataset_file, idx, params, smile_str, seed, device, kfold=K_FOLD
                )
            test_results.append([best_auc, best_accuracy])
        avg_auc, avg_accuracy = np.mean([test_result[0] for test_result in test_results]), np.mean([test_result[1] for test_result in test_results])
        with open(score_log_path, 'a') as log_file:
            log_file.write("\n----------------------------------------------------------------\n----------------------------------------------------------------\n")
            log_file.write(f"{len(test_results)} Seeds Avg: \nAUROC: {avg_auc}, Accuracy: {avg_accuracy}")