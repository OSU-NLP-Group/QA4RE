import pickle
import pandas as pd
import os
import openai
import numpy as np
import ipdb
import re
import tqdm

from utils.data_utils import *
from scipy import special
import json
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

def filter_by_dict(df, filter_v):
    for k, v in filter_v.items():
        df = df[df[k] == v]
    return df

def evaluate_re(df, label_verbalizer, pos_label_list, prediction_name='predictions', nota_eval=True, nota_eval_average='micro'):
    if nota_eval:
        precision, recall, f1, support = precision_recall_fscore_support(y_pred=df[prediction_name], y_true=df['verbalized_label'], labels=list(label_verbalizer.values()), average=nota_eval_average)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(y_pred=df[prediction_name], y_true=df['verbalized_label'],
                                                 labels=[label_verbalizer[l] for l in pos_label_list],
                                                # average='weighted')
                                                 average='micro')
    return {'f1': f1, 'precision': precision, 'recall': recall}


def batch_re_eval_print(df, prediction_name, label_verbalizer, pos_label_list, return_results=False):
    """Evaluate the RE task with different metrics on the given dataframe."""

    closed_micro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=False)
    print('Closed Set Performance (Micro):')
    print(closed_micro_metric_dict)

    print('-'*100)

    open_macro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=True, nota_eval_average='macro') # todo
    print('Open Set Performance (Macro):')
    print(open_macro_metric_dict)

    print('-'*100)

    open_micro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=True) # todo
    print('Open Set Performance (Micro):')
    print(open_micro_metric_dict)

    print('-'*100)

    verbalized_pos_lables = [label_verbalizer[l] for l in pos_label_list]
    df['binary_preds'] = df[prediction_name].apply(lambda x: 1 if x in verbalized_pos_lables else 0)
    binary_preds = df['binary_preds'].to_list()
    binary_labels = df['verbalized_label'].apply(lambda x: 1 if x in verbalized_pos_lables else 0)

    f1 = f1_score(binary_labels, binary_preds, labels=range(2), average="micro")
    print("binary P vs N micro F1 score:", f1)

    f1 = f1_score(binary_labels, binary_preds, labels=range(2), average="macro")
    print("binary P vs N macro F1 score:", f1)

    if return_results:
        return closed_micro_metric_dict, open_macro_metric_dict, open_micro_metric_dict
