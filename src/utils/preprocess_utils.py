import os
import re
import time
import argparse

import emoji
import numpy as np

import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset

import nltk
nltk.download('stopwords')

import spacy
import pandas as pd

from sklearn.model_selection import train_test_split

import transformers

dict_dataset_name_unprocessed = {'offenseval': {'train': 'data/training_data/offenseval-training-v1.tsv',\
                                 'test': 'data/test_data/testset-levela.tsv', 'test_labels': 'data/test_data/labels-levela.csv'}, 
                                 'covidhate' : {'train': 'data/training_data/covid_hate_training.tsv',\
                                 'test': 'data/test_data/covid_hate_test.tsv'},
                                 'SBF' : {'train': 'data/training_data/SBFv2_training.tsv',\
                                 'test': 'data/test_data/SBFv2_test.tsv'},
                                 'implicithate' : {'train': 'data/training_data/implicit_hate_v1_stg1_posts_training.tsv',\
                                 'test': 'data/test_data/implicit_hate_v1_stg1_posts_test.tsv'}
                                 }

dict_dataset_name_processed = {'offenseval': {'train': 'data/offenseval_train.csv',\
                               'val': 'data/offenseval_val.csv', 'test': 'data/offenseval_test.csv'}, 
                               'covidhate' : {'train': 'data/covidhate_train.csv',\
                               'val': 'data/covidhate_val.csv', 'test': 'data/covidhate_test.csv'},
                               'SBF' : {'train': 'data/SBF_train.csv',\
                               'val': 'data/SBF_val.csv', 'test': 'data/SBF_test.csv'},
                               'implicithate' : {'train': 'data/implicithate_train.csv',\
                               'val': 'data/implicithate_val.csv', 'test': 'data/implicithate_test.csv'}
                               }

def clean_line(line: str) -> list:
    """preprocesses a line of text"""
    line = re.sub(r'#([^ ]*)', r'\1', line)
    line = re.sub(r'https[^\t| ]*', 'URL', line)
    line = re.sub(r'http[^\t| ]*', 'URL', line)
    line = emoji.demojize(line)
    line = re.sub(r'(:.*?:)', r' \1 ', line)
    line = re.sub(' +', ' ', line)
    line = line.rstrip('\n').split('\t')
    return line

def format_training_file(dataset_name='', module_path=''):
    # TODO Remove this function and preprocess the datasets to tsv before the training pipeline
    tweets = []
    classes = []
    for line in open(module_path+dict_dataset_name_unprocessed[dataset_name]['train'],'r',encoding='utf-8'):
        line = clean_line(line)
        if dataset_name == 'offenseval':
            tweets.append(line[1])
            classes.append(int(line[2]=='OFF'))
        elif dataset_name == 'covidhate':
            tweets.append(line[1])
            classes.append(int(line[2].strip('\t')=='2'))
        elif dataset_name == 'SBF':
            offensive = set(['1', '1.0', '0', '0.0'])
            if len(line) >= 18 and line[5] in offensive:
                message = "".join(line[i] for i in range(15, len(line) - 4))
                if len(message) >= 1:
                    tweets.append(message)
                    classes.append(line[5])
        elif dataset_name == 'implicithate':
            hate_labels = set(['implicit_hate', 'explicit_hate'])
            if len(line) >= 3:
                tweets.append(line[1])
                classes.append(int(line[2] in hate_labels))

    # print("length of tweets, classes", len(tweets), len(classes))
    return tweets[1:], classes[1:]

def format_test_file(dataset_name='', module_path=''):
    # TODO Remove this function and preprocess the datasets to tsv before the training pipeline
    tweets_test = []
    y_test = []

    if dataset_name == 'offenseval':
        for line in open(module_path+dict_dataset_name_unprocessed[dataset_name]['test'],'r',encoding='utf-8'):
            line = clean_line(line)
            tweets_test.append(line[1])
        for line in open(module_path+dict_dataset_name_unprocessed[dataset_name]['test_labels'],'r',encoding='utf-8'):
            line = line.rstrip('\n').split('\t')
            y_test.append(int(line[0][-3:]=='OFF'))
        return tweets_test[1:], y_test
    else:
        for line in open(module_path+dict_dataset_name_unprocessed[dataset_name]['test'],'r',encoding='utf-8'):
            line = clean_line(line)
            if dataset_name == 'covidhate':
                tweets_test.append(line[1])
                y_test.append(int(line[2].strip('\t')=='2'))
            elif dataset_name == 'SBF':
                offensive = set(['1', '1.0', '0', '0.0'])
                if len(line) >= 18 and line[5] in offensive:
                    message = ""
                    for i in range(15, len(line) - 4):
                        message += line[i]
                    if len(message) >= 1:
                        tweets_test.append(message)
                        y_test.append(line[5])
            elif dataset_name == 'implicithate':
                hate_labels = set(['implicit_hate', 'explicit_hate'])
                if len(line) >= 3:
                    tweets_test.append(line[1])
                    y_test.append(int(line[2] in hate_labels))
        return tweets_test[1:], y_test[1:]


def train_val_split_tocsv(tweets, classes, val_size=0.2, module_path='', dataset_name=''):
    tweets_train, tweets_val, y_train, y_val = train_test_split(tweets, classes, test_size=val_size, random_state=42)

    df_train = pd.DataFrame({'text': tweets_train, 'label': y_train})
    df_val = pd.DataFrame({'text': tweets_val, 'label': y_val})

    train_csv_name = module_path+dict_dataset_name_processed[dataset_name]['train']
    val_csv_name = module_path+dict_dataset_name_processed[dataset_name]['val']
    df_train.to_csv(train_csv_name, index=False)
    df_val.to_csv(val_csv_name, index=False)
    return (train_csv_name, val_csv_name)

def test_tocsv(tweets_test, y_test, module_path='', dataset_name = ''):
    df_test = pd.DataFrame({'text': tweets_test, 'label': y_test})
    test_csv_name = module_path+dict_dataset_name_processed[dataset_name]['test']
    df_test.to_csv(test_csv_name, index=False)
    return test_csv_name

def create_fields_dataset(model_type, fix_length=None, module_path='', train_dataset_name='', test_dataset_name=''):
    tokenizer = None
    if model_type == "DistillBert":
        tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print('pad_index', pad_index)
        field = Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=pad_index, fix_length=fix_length)
    elif model_type == "DistillBertEmotion":
        tokenizer = transformers.DistilBertTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print('pad_index', pad_index)
        field = Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=pad_index, fix_length=fix_length)
    else:
        spacy_en = spacy.load("en_core_web_sm")
        def tokenizer_func(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        field = Field(sequential=True, use_vocab=True, tokenize=tokenizer_func, lower=True, fix_length=fix_length,
                      stop_words = nltk.corpus.stopwords.words('english'))

    label = LabelField(dtype=torch.long, batch_first=True, sequential=False)
    fields = [('text', field), ('label', label)]
    print("field objects created")
    train_data, val_data = TabularDataset.splits(
        path = '',
        train=module_path+dict_dataset_name_processed[train_dataset_name]['train'],
        test=module_path+dict_dataset_name_processed[train_dataset_name]['val'],
        format='csv',
        fields=fields,
        skip_header=True,
    )
    _, test_data = TabularDataset.splits(
        path = '',
        train=module_path+dict_dataset_name_processed[test_dataset_name]['train'],
        test=module_path+dict_dataset_name_processed[test_dataset_name]['test'],
        format='csv',
        fields=fields,
        skip_header=True,
    )

    return (field, tokenizer, label, train_data, val_data, test_data)

#Create train and test iterators to use during the training loop
def create_iterators(train_data, test_data, batch_size, dev, shuffle=False):
    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        shuffle=shuffle,
        device=dev,
        batch_size=batch_size,
        sort = False,
        )
    return train_iterator, test_iterator

def get_vocab_stoi_itos(field, tokenizer=None):
    return (tokenizer.encode, tokenizer.encode) if tokenizer else (field.vocab.stoi, field.vocab.itos)

def create_formatted_csvs(train_dataset_name, test_dataset_name, module_path=''):
    # preprocessing of the train/validation tweets, then test tweets
    tweets, classes = format_training_file(dataset_name=train_dataset_name, module_path=module_path)
    tweets_test, y_test = format_test_file(dataset_name=test_dataset_name, module_path=module_path)
    print("file loaded and formatted..")
    train_csv_name, val_csv_name = train_val_split_tocsv(tweets, classes, val_size=0.2, module_path=module_path, dataset_name=train_dataset_name)
    test_csv_name = test_tocsv(tweets_test, y_test, module_path=module_path, dataset_name=test_dataset_name)
    print("data split into train/val/test")
    return(train_csv_name, val_csv_name, test_csv_name)

def get_datasets(train_dataset_name, test_dataset_name, model_type, fix_length=None, module_path=''):
    field, tokenizer, label, train_data, val_data, test_data = create_fields_dataset(model_type, fix_length, 
                                                                                     module_path=module_path, 
                                                                                     train_dataset_name=train_dataset_name, 
                                                                                     test_dataset_name=test_dataset_name)

    # build vocabularies using training set
    print("fields and dataset object created")
    field.build_vocab(train_data, max_size=10000, min_freq=2)
    label.build_vocab(train_data)
    print("vocabulary built..")

    return (field, tokenizer, train_data, val_data, test_data)

def get_dataloaders(train_data, val_data, test_data, batch_size, device):
    train_iterator, val_iterator = create_iterators(train_data, val_data, batch_size, device, shuffle=True)
    _, test_iterator = create_iterators(train_data, test_data, 1, device, shuffle=False)
    print("dataloaders created..")

    return {'train': train_iterator, 'val': val_iterator, 'test': test_iterator}

def preprocessed_datasets_exist(dataset_name):
    dataset_dict = dict_dataset_name_processed[dataset_name]

    train_path, val_path, test_path = dataset_dict['train'], dataset_dict['val'], dataset_dict['test']

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Dataset name to preprocess", default="SBF")
    parser.add_argument("--force_preprocess", help="Force preprocessing", default=0, type=int)

    args = parser.parse_args()

    # Data preprocessing
    dataset_name = args.dataset_name
    force_preprocess = args.force_preprocess

    start_time = time.time()
    if dataset_name == 'all':
        for dataset_name in dict_dataset_name_unprocessed:
            if not preprocessed_datasets_exist(dataset_name) or force_preprocess:
                print("Preprocess {}".format(dataset_name))
                create_formatted_csvs(dataset_name, dataset_name)
                print("duration: ", time.time()-start_time)
            else:
                print("{} preprocessing already done".format(dataset_name))
            start_time = time.time()
    else:
        create_formatted_csvs(dataset_name, dataset_name)
        print("duration: ", time.time()-start_time)
