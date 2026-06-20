### Modified from UnsupPSE's filter.py and bucc_f-score.py files ###

import argparse
import logging
import numpy as np
import os
import pandas as pd
import utils as utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--main_path', type=str, required=True, help='Path of the similarity files.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name used for the embeddings.')
    parser.add_argument('--language_pair', type=str, required=True, help='Language pair (e.g., hsb-de).')
    parser.add_argument('--gold_path', type=str, required=True, help='Path of the gold files (.gold).')
    parser.add_argument('--threshold', default=2.0, type=float, 
                      help='Dynamic threshold hyperparameter (default: 2.0).')

    return parser.parse_args()


MAIN_PATH = './results_full_xlmr/dictionaries' # TO REPLACE

# Threshold dictionary
threshold_dict = {f'{i}': f'{i * 0.1:.1f}' for i in range(0, 31)}

# Filtering the sentence pairs
def dynamic_threshold(sentence_pair_list, threshold=2.0):
    '''Compute a dynamic threshold for mining.'''
    score_list = []
    for line in sentence_pair_list:
        best = line.split('\t')[2]
        score_list.append(float(best))
    s = np.array(score_list)
    final_threshold = s.mean() + threshold * s.std()
    # print(f'The dynamic threshold value is {final_threshold}.')
    return final_threshold


def filter_file(input, method='dynamic', threshold=2.0, test=False):
    assert method in ['static', 'dynamic'], f'The method is unknown: {method}'

    sentence_pair_list = open(input, 'r').read()
    # split_sp_list = sentence_pair_list.split('\n')
    split_sp_list = utils.text_to_line(sentence_pair_list)

    # Convert sentence pair file into DataFrame
    data_list = []
    for line in split_sp_list:
        split_line = line.split('\t')
        data_list.append([f'{split_line[0]}\t{split_line[1]}', float(split_line[2])])
    sentence_pair_df = pd.DataFrame(data_list, columns=['sentence_pair', 'sim_score'])

    # Set the threshold
    final_threshold = threshold

    if method == 'dynamic':
        # print(f'The dynamic threshold value is {threshold}.')
        if test: # Testing set
            final_threshold = dynamic_threshold(split_sp_list, threshold)
        else: # Training set
            final_threshold_dict = dict()
            for th in threshold_dict.keys():
                final_threshold_dict[th] = dynamic_threshold(split_sp_list, float(threshold_dict[th]))

    # Filtering
    if test:
        filter_df = sentence_pair_df[sentence_pair_df['sim_score'] > final_threshold]
        return list(filter_df['sentence_pair'])
    else:
        filter_dict = dict()
        for th in threshold_dict.keys():
            filter_df = sentence_pair_df[sentence_pair_df['sim_score'] > final_threshold_dict[th]]
            filter_dict[th] = list(filter_df['sentence_pair'])
        return filter_dict

def best_mining_threshold(hyperparameter_metric_dict):
    '''Find the best threshold on the training set.'''
    f_score_dict = {threshold: prf[2] for threshold, prf in hyperparameter_metric_dict.items()}
    best_th = max(f_score_dict, key=f_score_dict.get)
    
    # Check if the best value is a border threshold or not
    th_list = list(f_score_dict.keys())
    if best_th in [th_list[0], th_list[-1]]:
        print(f'The best threshold is at the edge: {best_th} in {th_list}')
    return int(best_th)

# Scoring of the mined pairs
def compute_prf_score(prediction_list, gold_list):
    '''Compute precision, recall, and F1 score for sentence mining.'''
    gold_labels = dict()
    # with open(gold, 'r') as fin:
        # for line in fin:
    for line in gold_list:
        src, trg = line.split('\t')
        if src in gold_labels:
            raise ValueError(f'Found ID multiple times in gold: {src}')

        gold_labels[src] = trg

    tp = 0
    fp = 0
    N = len(gold_labels)
    seen = set()
    # with open(prediction, 'r') as fin:
        # for line in fin:
    for line in prediction_list:
        src, trg = line.split('\t')
        if src in seen:
            raise ValueError(f'Found ID multiple times in prediction: {src}')

        seen.add(src)
        if src in gold_labels:
            if trg == gold_labels[src]:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1

    fn = N - tp
    # logger.debug(f'tp: {tp} fp: {fp} fn: {fn} N: {N}')
    # print(f'tp: {tp} fp: {fp} fn: {fn} N: {N}')

    P = 0.0
    R = 0.0
    F1 = 0.0
    if tp + fp > 0:
        P = tp / (tp + fp)
    if tp + fn > 0:
        R = tp / (tp + fn)
    if P + R > 0:
        F1 = (2 * P * R) / (P + R)

    # Multiply by 100 and floor values
    P = round(P * 100, 4)
    R = round(R * 100, 4)
    F1 = round(F1 * 100, 4)

    # print('PRECISION, RECALL, F1')
    # print(f'{P}, {R}, {F1}')
    return P, R, F1


def score_file(filtered_pair, gold_list, test=False):
    '''Compute PRF metrics after filtering.'''
    if test: # Test data
        assert type(filtered_pair) is list, 'The filtered TEST pair is not a list.'
        return compute_prf_score(filtered_pair, gold_list)
    
    else: # Train data
        assert type(filtered_pair) is dict, 'The filtered TRAIN pair is not a dictionary.'
        mine_score_dict = dict()
        for th in threshold_dict.keys():
            mine_score_dict[th] = compute_prf_score(filtered_pair[th], gold_list)
        return mine_score_dict


# Pipeline function
def mine_and_evaluate(model_name, language_pair, gold_path, threshold=2.0):
    '''Mine sentence pairs and compute PRF metrics for both train and test sets.'''
    print(f'Mining for {model_name} on {language_pair}')

    # On the training set
    train_filter_dict = filter_file(f'{MAIN_PATH}/DOC.{model_name}.{language_pair}.train.sim',
                                    method='dynamic', threshold=threshold, test=False)
    assert type(train_filter_dict['20']) is list, 'The training set is not correclty filtered.'

    gold_train_list = utils.Text(f'{gold_path}/{language_pair}/{language_pair}.train.gold').split_file
    mine_score_dict = score_file(train_filter_dict, gold_train_list, test=False)
    print(mine_score_dict)
    best_threshold = best_mining_threshold(mine_score_dict)
    print(f'Best mining threshold on the training set: {best_threshold}')

    # On the test set
    test_filter_list = filter_file(f'{MAIN_PATH}/DOC.{model_name}.{language_pair}.test.sim',
                                   method='dynamic', threshold=float(threshold_dict[str(best_threshold)]), test=True)
    assert type(test_filter_list) is list, 'The test set is not correclty filtered.'

    gold_test_list = utils.Text(f'{gold_path}/{language_pair}/{language_pair}.test.gold').split_file
    mine_score_test = score_file(test_filter_list, gold_test_list, test=True)
    return mine_score_test

def new_mine_and_evaluate(main_path, model_name, language_pair, gold_path, threshold=2.0):
    '''Mine sentence pairs and compute PRF metrics for both train and test sets.'''
    print(f'Mining for {model_name} on {language_pair}')

    # main_path = MAIN_PATH
    # On the training set
    train_filter_dict = filter_file(os.path.join(main_path, f'DOC.{model_name}.{language_pair}.train.sim'),
                                    method='dynamic', threshold=threshold, test=False)
    assert type(train_filter_dict['20']) is list, 'The training set is not correclty filtered.'

    gold_train_list = utils.Text(f'{gold_path}/{language_pair}/{language_pair}.train.gold').split_file
    mine_score_dict = score_file(train_filter_dict, gold_train_list, test=False)
    print(mine_score_dict)
    best_threshold = best_mining_threshold(mine_score_dict)
    print(f'Best mining threshold on the training set: {best_threshold}')

    # On the test set
    test_filter_list = filter_file(os.path.join(main_path, f'DOC.{model_name}.{language_pair}.test.sim'),
                                   method='dynamic', threshold=float(threshold_dict[str(best_threshold)]), test=True)
    assert type(test_filter_list) is list, 'The test set is not correclty filtered.'

    gold_test_list = utils.Text(f'{gold_path}/{language_pair}/{language_pair}.test.gold').split_file
    mine_score_test = score_file(test_filter_list, gold_test_list, test=True)
    return mine_score_test


# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     args = getArguments()

#     main(**vars(args))