"""General utils for all RE projs."""
import os
from copy import deepcopy
import pandas as pd
from utils.general_utils import *
from projs.re_templates import *


def add_type_constraints(df):
    type_constraints = []
    for i, row in df.iterrows():
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        type_cons = f"{ent1_type.strip()}:{ent2_type.strip()}"

        type_constraints.append(type_cons)
    df['type_cons'] = type_constraints
    return df


def get_type_constrained_random_demos(train, dev, in_context_size, random_seed=42, cross_val=False):
    if cross_val:
        dev = train.copy()
        dev = dev.reset_index()

    random_prompts = []

    for i, row in dev.iterrows():

        if cross_val:
            curr_train = train[train.index != i]
        else:
            curr_train = train
        prompt = ''
        type_cons = row['type_cons']
        subset_df = deepcopy(curr_train[curr_train["type_cons"] == type_cons])
        if len(subset_df) < in_context_size:  # not enough examples, add from the whole train set.
            extra_subset_df = curr_train.sample(n=in_context_size-len(subset_df), random_state=random_seed)
            subset_df = pd.concat([subset_df, extra_subset_df])
        else:
            subset_df = subset_df.sample(n=in_context_size, random_state=random_seed)

        subset_df = subset_df.reset_index(drop=True)

        for j, sub_row in subset_df.iterrows():
            prompt += subset_df['demo_prompt'][j] + "\n\n"

        empty_prompt = row['empty_prompts']

        random_prompts.append(prompt + empty_prompt)

    dev['data_prompts'] = random_prompts
    dev['prompt_samples'] = 'random'

    return dev


def process_arguments(args):
    """Process arguments to under different conditions.
    """
    if args.run_setting.startswith('retrieval_few_shot'):
        args.setting = 'few_shot'
    else:
        args.setting = 'zero_shot'  # fixed few shot is easilly conducted in zero shot method as the train set is fixed.

    if args.debug:
        args.train_subset_samples = 10
        args.dev_subset_samples = 4
        args.test_subset_samples = 10

    if args.mode == 'dev':
        args.run_subset_num = args.dev_subset_samples
    elif args.mode == 'test':
        args.run_subset_num = args.test_subset_samples
    else:
        raise ValueError(f'Unknown mode {args.mode}')
    args.config_path = os.path.join(args.data_root, args.dataset, 'configs', args.prompt_config_name)
    # # args.config_path = os.path.join(args.data_root, 'configs', args.prompt_config_name)
    # if args.dataset in ['TACRED', 'RETACRED', 'semeval', 'TACREV'] and not args.type_constrained:
    #     raise ValueError(f'{args.dataset} can only be supported by type_constrained.')

    if args.dataset == 'TACRED' or args.dataset == 'TACREV':
        # LABEL_TEMPLATES = TACRED_LABEL_TEMPLATES
        args.LABELS = TACRED_LABELS
        args.LABEL_TEMPLATES = SURE_TACRED_LABEL_TEMPLATES
        args.VALID_CONDITIONS_REV = TACRED_VALID_CONDITIONS_REV
        args.LABEL_VERBALIZER = TACRED_LABEL_VERBALIZER
        args.NOTA_RELATION = "no_relation"
    elif args.dataset == 'RETACRED':
        args.LABELS = RETACRED_LABELS
        args.LABEL_TEMPLATES = SURE_RETACRED_LABEL_TEMPLATES
        args.VALID_CONDITIONS_REV = RETACRED_VALID_CONDITIONS_REV
        args.LABEL_VERBALIZER = RETACRED_LABEL_VERBALIZER
        args.NOTA_RELATION = "no_relation"
    elif args.dataset == 'semeval':  # it's actually all available relations
        args.LABELS = SEMEVAL_LABELS
        args.LABEL_TEMPLATES = SURE_SEMEVAL_LABEL_TEMPLATES
        args.VALID_CONDITIONS_REV = SEMEVAL_VALID_CONDITIONS_REV
        args.LABEL_VERBALIZER = SEMEVAL_LABEL_VERBALIZER
        args.NOTA_RELATION = "Other"

    args.POS_LABELS = list(set(args.LABELS) - set([args.NOTA_RELATION]))

    print_args(args)

    return args
