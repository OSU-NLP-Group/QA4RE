import os
import sys
import pickle
import time
import random
import ipdb
import re
import signal
from collections import defaultdict
from cmath import cos
from contextlib import nullcontext
from typing import OrderedDict

import spacy
import scipy
import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tenacity import retry, stop_after_attempt, wait_random_exponential


from utils.data_utils import *
from utils.eval_utils import *

openai.api_key = os.environ["OPENAI_KEY"]

GPT_MODEL_TYPE_DICT = {
    "davinci": "text",
    "text-davinci-001": "text",
    "text-davinci-002": "text",
    "text-davinci-003": "text",
    "gpt-3.5-turbo": "chat",
    "gpt-3.5-turbo-0301": "chat",
    "gpt-4": "chat",
    "gpt-4-0314": "chat",
}

GPT_MODEL_MAX_TOKEN_DICT = {
    "davinci": 2048,
    "text-davinci-001": 2048,
    "text-davinci-002": 4096,
    "text-davinci-003": 4096,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
}

# per thousand token cost. 04-11
GPT_MODEL_MAX_COST_DICT = {
    "davinci": 0.02,
    "text-davinci-001": 0.02,
    "text-davinci-002": 0.02,
    "text-davinci-003": 0.02,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0301": 0.002,
    "gpt-4": 0.03,  # 8k contet version / 0.03 for prompt and 0.06 for completion.
    "gpt-4-0314": 0.03,
}


def estimate_cost(prompt, engine, tokenizer):
    """estimate the cost for input prompt based on the num of token."""
    tokens = tokenizer.encode(prompt)
    return len(tokens) * GPT_MODEL_MAX_COST_DICT[engine] / 1000


def estimate_cost_df(df, engine, tokenizer):
    """estimate the cost for each example."""
    cost_list = []
    for i, row in df.iterrows():
        cost = estimate_cost(row['final_input_prompts'], engine, tokenizer)
        cost_list.append(cost)
    return cost_list


# Necessity of adding '\n' to the end tokens?
def build_logit_biases(
    words, max_tokens, tokenizer, logit_bias_weight=100, end_tokens=["<|endoftext|>"]
):
    """build the logit biases for the API call to increase chance of generating target tokens.
    Pretty helpful for multi-choice QA format.
    return: a dictionary of {token_id: logit_bias_weight}
    """
    logit_biases = {}
    if max_tokens == 1:  # first_token_verbalized_labels. for datasets like bioIE.
        for word in words:
            word = tokenizer.encode(" " + word)[0]
            logit_biases[word] = logit_bias_weight
    else:
        for word in words:
            tokens = tokenizer.encode(" " + word)
            for token in tokens:
                logit_biases[token] = logit_bias_weight
    for end_token in end_tokens:
        end_token = tokenizer.encode(end_token)
        logit_biases[end_token[0]] = logit_bias_weight

    return logit_biases


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def call_gpt_engine(
    engine,
    final_input_prompt,
    logit_biases={},
    max_tokens=1,
    temperature=0.0,
    num_logprobs=5,
    stop_tokens=["\n", "<|endoftext|>"],
    **kwargs,
):
    """
    kwargs: other parameters for the API call. for future use.
    """
    if GPT_MODEL_TYPE_DICT[engine] == "text":
        sample = openai.Completion.create(
            engine=engine,
            prompt=final_input_prompt,
            max_tokens=int(max_tokens),
            temperature=temperature,
            logit_bias=logit_biases,
            stop=stop_tokens,
            logprobs=num_logprobs,
        )
    elif GPT_MODEL_TYPE_DICT[engine] == "chat":
        sample = openai.ChatCompletion.create(
            model=engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_input_prompt},
            ],
            temperature=temperature,
            stop=stop_tokens,
            max_tokens=int(max_tokens),
        )
    else:
        raise ValueError("Unknown GPT model type")

    time.sleep(0.1)

    return sample
