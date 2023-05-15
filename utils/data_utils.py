"""
"""
import pandas as pd
import numpy as np
import ipdb
import pickle
import re
import os
import yaml
from transformers import GPT2Tokenizer
# gpt_2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import copy
import wandb
from copy import deepcopy
from torch.utils.data import TensorDataset
from typing import OrderedDict
from glob import glob


def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def exp_dir_setup(args):
    """setup experiment directory."""

    # output_dir = '../../../outputs/{}/{}/{}/{}'.format(args.task, args.dataset, args.ex_name, args.run_setting)
    output_dir = '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine, args.run_setting)

    yaml_config = load_yaml(args.config_path)  # make sure every key has just one value.
    for key, value in yaml_config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            yaml_config[key] = value[0]

    if args.debug:
        output_dir += '/debug'
        experiment_num = len(glob(output_dir + '/*'))
    else:
        experiments = glob(output_dir + '/*')
        debug_path = output_dir + '/debug'
        if debug_path in experiments:
            experiments.remove(debug_path)  # remove debug dir if exists

        # reverse see through.
        for exp in experiments[::-1]:
            exp_yaml = load_yaml(exp + '/config.yaml')
            if exp_yaml == yaml_config:
                experiment_num = int(exp.split('/')[-1])
                output_dir += '/{}'.format(experiment_num)
                return yaml_config, output_dir, experiment_num

        # Create experiment directory
        experiment_num = len(experiments)

    output_dir += '/{}'.format(experiment_num)
    os.makedirs(output_dir)
    yaml.dump(yaml_config, open(output_dir + '/config.yaml', 'w'))

    return yaml_config, output_dir, experiment_num

def ent_finder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return i
    if len(pattern) == 0:
        raise ValueError('Not found pattern')
    else:
        return ent_finder(mylist, pattern[:-1])


def get_embedding(row, model, tokenizer, mode='cls'):
    sent = row['sents']
    ent1 = row['ent1']
    ent2 = row['ent2']
    input_dict = tokenizer(sent, return_tensors='pt').to('cuda')
    embedding = model(**input_dict)['last_hidden_state'].cpu().numpy()[0]

    if mode == 'cls':
        embedding = embedding[0]
    elif mode == 'avg':
        embedding = np.mean(embedding, axis=0)
    elif mode == 'ent':
        ent1_tokens = tokenizer(ent1, return_tensors='pt')['input_ids'][0][1:-1].tolist()
        ent2_tokens = tokenizer(ent2, return_tensors='pt')['input_ids'][0][1:-1].tolist()

        ent1_start_pot = ent_finder(input_dict['input_ids'][0].tolist(), ent1_tokens)
        ent2_start_pot = ent_finder(input_dict['input_ids'][0].tolist(), ent2_tokens)

        # sent_emb = embedding[0]
        # h_ent_emb = embedding[ent1_start_pot]
        # t_ent_emb = embedding[ent2_start_pot]

        # # embedding = np.mean(np.concatenate([sent_emb, h_ent_emb, t_ent_emb]), axis=0)
        embedding = np.mean(embedding[[0, ent1_start_pot, ent2_start_pot]], axis=0)
    else:
        raise ValueError('Mode {} not supported'.format(mode))
    return embedding


def get_embeddings(df, model, tokenizer, mode='cls', embedding_task='RE', retrieval_col='sents'):
    """
    Get embeddings for a dataframe of sentences. used to support RE and Sentence classification tasks. Now focus on RE.
    """
    embeddings = []

    with torch.no_grad():
        for i, row in tqdm(df.iterrows()):
            embedding = get_embedding(row, model, tokenizer, mode, retrieval_col)
            embeddings.append(embedding)

    embeddings = np.array(embeddings)
    norm_embeddings = embeddings.T / np.linalg.norm(embeddings, axis=1)

    return norm_embeddings.T


def get_demonstrations(train, dev, prompt_size, sampling_strategy, random_seed=42,
                        cross_val=False):
    if sampling_strategy == 'random':
        dev = get_random_demos(train, dev, prompt_size, cross_val, random_seed)
    else:
        dev = get_retrieval_k_demos(train, dev, prompt_size, sampling_strategy, cross_val)

    return dev


def get_random_demos(train, dev, prompt_size, cross_val, random_seed=42):
    if cross_val:
        dev = train.copy()
        dev = dev.reset_index()

    random_prompts = []

    for i, row in dev.iterrows():

        if cross_val:
            curr_train = train[train.index != i]
        else:
            curr_train = train

        prompt_samples = curr_train.sample(prompt_size, random_state=np.random.RandomState(random_seed+i))
        prompt_samples = prompt_samples.prompts.values

        empty_prompt = row['empty_prompts']

        random_prompts.append('\n\n'.join(prompt_samples) + '\n\n' + empty_prompt)

    dev['data_prompts'] = random_prompts
    dev['prompt_samples'] = 'random'

    return dev


def get_retrieval_k_demos(train, dev, prompt_size, sampling_strategy, cross_val=False):
    if cross_val:
        dev = train.copy()
        dev = dev.reset_index()

    bert_model = sampling_strategy
    
    mode = 'cls'

    model = AutoModel.from_pretrained(bert_model).to('cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    train_embeddings = get_embeddings(train.sents.values, model, tokenizer, mode)

    knn_prompt_samples = []
    knn_prompts = []

    with torch.no_grad():
        for i, row in dev.iterrows():  # todo fix bug, pos examples are not ordered.

            test_sent = row['sents']

            sent_emb = get_embedding(test_sent, model, tokenizer, mode=mode)
            sent_emb = sent_emb / np.linalg.norm(sent_emb)

            if cross_val:
                if i == 0:
                    cross_val_train_embeddings = train_embeddings[1:]
                elif i == len(dev):
                    cross_val_train_embeddings = train_embeddings[-1:]
                else:
                    cross_val_train_embeddings = np.vstack((train_embeddings[:i], train_embeddings[i + 1:]))

                assert len(cross_val_train_embeddings) == len(dev) - 1, ipdb.set_trace()
                sims = cross_val_train_embeddings.dot(sent_emb)
                indices = np.concatenate([range(i), range(i+1,len(train_embeddings))])
            else:
                sims = train_embeddings.dot(sent_emb)
                indices = np.arange(len(train_embeddings))

            real_indices = indices.astype('int')
            sorted_indices = np.argsort(sims, kind='stable')[::-1]
            assert sims[sorted_indices[0]] > sims[sorted_indices[-1]], ipdb.set_trace()
            selected_sims = sims[sorted_indices][:prompt_size]

            real_sorted_indices = real_indices[sorted_indices]
            selected_real_indices = real_sorted_indices[:prompt_size]
            selected_prompts = train.prompts.values[selected_real_indices]

            empty_prompt = row['empty_prompts']

            knn_prompt_samples.append((selected_prompts, selected_sims))
            knn_prompts.append('\n\n'.join(selected_prompts) + '\n\n' + empty_prompt)

        dev['prompt_samples'] = knn_prompt_samples
        dev['data_prompts'] = knn_prompts

    return dev

