import time
import argparse
import datetime

import numpy as np
import pandas as pd

import torch

import itertools
import yaml

from src.utils.preprocess_utils import create_formatted_csvs, get_datasets, get_dataloaders
from src.utils.utils import GRIDSEARCH_CSV
from src.training.main import main

def get_gridsearch_config(config_path):
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    hyperparameters = config['hyperparameters']
    print('hyperparameters keys', list(hyperparameters.keys()))

    all_config_list = []
    for param_name in hyperparameters.keys():
        all_config_list.append(hyperparameters[param_name])

    return all_config_list

def gridsearch(config_path, do_save, device):
    all_config_list = get_gridsearch_config(config_path)

    training_remaining = np.prod([len(config) for config in all_config_list])
    list_datasets = all_config_list[0]
    training_remaining /= len(list_datasets) # No cross-dataset evaluation for training
    print('Training to do:', training_remaining)

    # Save gridsearch training to csv
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_path = GRIDSEARCH_CSV+'results_{}.csv'.format(current_time)
    results_dict = {'train_dataset_name': [],
                    'test_dataset_name': [], 
                    'model_type': [],
                    'optimizer_type': [], 
                    'loss_criterion': [], 
                    'lr': [], 
                    'epochs': [], 
                    'batch_size': [], 
                    'patience_es': [], 
                    'scheduler_type': [],
                    'patience_lr': [], 
                    'save_condition': [],
                    'fix_length': [],
                    'best_epoch': [], 
                    'train_loss': [], 
                    'val_loss': [], 
                    'train_acc': [], 
                    'val_acc': [], 
                    'test_acc': [], 
                    'end_time': []}

    # Start gridsearch
    prev_model_type = None
    prev_train_dataset_name = None
    start_time = time.time()
    for params in itertools.product(*all_config_list):
        # /!\ Has to be in the same order as in the config.yaml file /!\ #
        train_dataset_name, test_dataset_name, \
        model_type, optimizer_type, \
        loss_criterion, lr, epochs, \
        batch_size, patience_es, \
        scheduler_type, patience_lr, \
        save_condition, fix_length = params

        # No cross-dataset evaluation for training
        if train_dataset_name != test_dataset_name:
            continue

        if prev_model_type != model_type or train_dataset_name != prev_train_dataset_name:
            print("prev_model_type", prev_model_type)
            print("model_type", model_type)
            print("prev_train_dataset_name", prev_train_dataset_name)
            print("train_dataset_name", train_dataset_name)
            print("Changing tokenizer...")
            ENGLISH, tokenizer, train_data, val_data, test_data = get_datasets(train_dataset_name, test_dataset_name, 
                                                                               model_type, fix_length)
            prev_model_type = model_type
            prev_train_dataset_name = train_dataset_name

        print('fix_length:', fix_length)
        print('batch_size:', batch_size)
        print("train_dataset_name", train_dataset_name)

        dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

        history_training = main(dataloaders, ENGLISH, model_type, optimizer_type, 
                               loss_criterion, lr, batch_size, epochs, patience_es, 
                               do_save, device, 
                               do_print=False, training_remaining=training_remaining, 
                               scheduler_type=scheduler_type, patience_lr=patience_lr, 
                               save_condition=save_condition)

        # Save training results to csv
        best_epoch = history_training['best_epoch']
        for key in results_dict.keys():
            if key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
                results_dict[key].append(history_training[key][best_epoch])
            elif key == 'train_dataset_name':
                results_dict[key].append(train_dataset_name)
            elif key == 'test_dataset_name':
                results_dict[key].append(test_dataset_name)
            elif key == 'epochs':
                results_dict[key].append(epochs)
            elif key == 'batch_size':
                results_dict[key].append(batch_size)
            elif key == 'fix_length':
                results_dict[key].append(fix_length)
            else:
                results_dict[key].append(history_training[key])

        results_csv = pd.DataFrame(data=results_dict)
        results_csv.to_csv(csv_path)

        training_remaining -= 1

    time_elapsed = time.time() - start_time
    print('\nGridsearch complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="gridsearch config", default="gridsearch_config.yml")
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
    parser.add_argument("--device", default='' , help="cpu or cuda for gpu")

    args = parser.parse_args()

    # Data processing
    config_path = args.config_path

    # Hyperparameters
    do_save = args.do_save

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    gridsearch(config_path, do_save, device)
