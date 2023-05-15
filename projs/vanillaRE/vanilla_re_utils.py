import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from transformers import GPT2Tokenizer

from projs.re_templates import *
from projs.re_utils import *
from utils.gpt3_utils import *
from utils.eval_utils import *
from utils.general_utils import *


def fill_prompt_format_re_df(df, all_label_verbalizer, prompt_config, sent_col='sents'):
    """
    prompts are full, with sentence, entity, and label.
    empty_prompts with only sentence, entity.
    So prompts are used for demonstrations while empty_prompts are used for test examples
    """
    prompts = []
    empty_prompts = []
    labels = []

    prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['example_format'] + ' {}'
    empty_prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['example_format']

    for i, row in df.iterrows():
        sent = row[sent_col]
        entity1 = row['ent1']
        entity2 = row['ent2']
        label = row['label']
        label = all_label_verbalizer[str(label)]
        prompt = prompt_sample_structure.format(sent, entity1, entity2, label)
        empty_prompt = empty_prompt_sample_structure.format(sent, entity1, entity2)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)
        labels.append(label)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['verbalized_label'] = labels
    unique_labels = set(labels)
    df['unique_labels'] = [unique_labels for _ in empty_prompts]

    return df


def params_update(args, params, df):
    """
    Prepare the verbalizer and task instructions for the run.
    """
    # for all run settings, we need to add second step examples.
    params['task_instructions'] = params['task_instructions'].split('\n')[0]  # only the first line is used.
    params['class_enumerations'] = []
    params['label_verbalizers'] = []
    if args.no_class_explain:
        if 'below with explanations' in params['task_instructions']:
            params['task_instructions'] = params['task_instructions'].replace('below with explanations', 'below')
    else:
        if 'below with explanations' not in params['task_instructions']:
            params['task_instructions'] = params['task_instructions'].replace('below', 'below with explanations')
    # add type constrained examples for df
    # df = deepcopy(df)[:num_example]
    for i, row in df.iterrows():  # reverse relation
        example_instruction = ""
        ent1_type, ent2_type = row['ent1_type'].strip(), row['ent2_type'].strip()
        entity_type_constrain = ent1_type + ":" + ent2_type  # check they are uppercase.
        if args.type_constrained:  # only add type constrained examples.
            possible_rels = args.VALID_CONDITIONS_REV[entity_type_constrain] + [args.NOTA_RELATION]  # add no_relation, it's possible for any time.
        else:
            possible_rels = list(args.LABEL_TEMPLATES.keys())
            # make no_relation the last one.
            possible_rels.remove(args.NOTA_RELATION)
            possible_rels.append(args.NOTA_RELATION)

        for rel in possible_rels:
            if args.no_class_explain:
                example_instruction += "- " + rel + "\n"
            else:
                templates = args.LABEL_TEMPLATES[rel]
                filled_templates = []
                for template in templates:
                    filled_template = template.format(subj="Entity 1", obj='Entity 2')
                    filled_templates.append(filled_template)
                example_instruction += "- " + rel + ": " + " OR ".join(filled_templates) + "\n"  # fixed demonstrations.
        params['class_enumerations'].append(example_instruction.strip())
        params['label_verbalizers'].append({rel: args.LABEL_VERBALIZER[rel] for rel in possible_rels}) # key, value are not the same for bio RE datasets.

    return params


def get_prediction(params, openai_api_response, verbalized_labels, engine, first_token_of_each_verb_label):
    """get the prediction from the openai api response.
    return generated_content, and parsed prediction.
    """
    def get_pred_from_generated_content(generated_content, verbalized_labels):
        """get the prediction from the generated content, for multiple token generation and chatCompletion"""
        candidates = []
        for label in verbalized_labels:
            if label in generated_content:
                candidates.append(label)
            if len(candidates) == 0:  # not generated, randomly choose one.
                pred = random.choice(verbalized_labels)
            elif len(candidates) > 1:  # get the first one in generation order
                start_indexes = [generated_content.index(can) for can in candidates]
                pred = candidates[np.argmin(start_indexes)]
            else:
                pred = candidates[0]
        return pred

    if GPT_MODEL_TYPE_DICT[engine] == 'text':  # able to get prob and some logit.
        text = openai_api_response['choices'][0]['text']
        if params['max_tokens'] == 1:
            # first_token_of_verb = [tokenizer.decode(tokenizer.encode(" " + label)[0]) for label in verbalized_labels]
            probs = []
            top_logprobs = openai_api_response['choices'][0]['logprobs']['top_logprobs'][0]
            for token in first_token_of_each_verb_label:
                if token in top_logprobs:
                    probs.append(top_logprobs[token])
                else:
                    probs.append(-100)
            return verbalized_labels[np.argmax(probs)]
        else:
            generated_content = openai_api_response['choices'][0]['text']
            pred = get_pred_from_generated_content(generated_content, verbalized_labels)

    elif GPT_MODEL_TYPE_DICT[engine] == 'chat':
        generated_content = openai_api_response['choices'][0]['message']['content']
        text = generated_content
        pred = get_pred_from_generated_content(generated_content, verbalized_labels)
    
    else:
        raise ValueError('GPT_MODEL_TYPE_DICT[engine] should be text or chat')
    return text, pred


def build_final_input(df, params, engine, tokenizer):
    """build the test ready prompts for the API call."""
    max_tokens = params['max_tokens']
    build_final_input = []
    for i, row in tqdm(df.iterrows()):
        task_instructions = params['task_instructions'].strip()
        class_enumerations = params['class_enumerations'][i].strip()
        prompt = class_enumerations + '\n\n' + row['data_prompts']
        final_input_prompt = task_instructions + '\n' + prompt

        tokens = tokenizer.encode(final_input_prompt)
        max_allowed_token_num = GPT_MODEL_MAX_TOKEN_DICT[engine]
        if len(tokens) + max_tokens > max_allowed_token_num:  # only necessary for few-shot setting.
            print(f"Warning: the prompt is too long for {engine}, will remove all the demonstrations.")
            while len(tokens) + max_tokens > max_allowed_token_num:
                final_input_prompt = task_instructions + '\n' + class_enumerations + '\n\n' + row['empty_prompts']
                tokens = tokenizer.encode(final_input_prompt)
        build_final_input.append(final_input_prompt.strip())  # make sure not ends with ' '.
    return build_final_input


def build_logit_biases_df(df, params, tokenizer):
    """build the logit biases for each row in the dataframe.
    return: all_logit_biases, all_verbalized_labels, all_first_token_of_each_verb_label (for single token generation evaluation)
    """
    all_logit_biases = []
    all_verbalized_labels = []
    all_first_token_of_each_verb_label = []
    for i, row in df.iterrows():
        label_verbalizer = params['label_verbalizers'][i]

        label_list = np.sort(list(label_verbalizer.keys()), kind='stable')
        verbalized_labels = [label_verbalizer[l] for l in label_list]
        logit_biases = build_logit_biases(verbalized_labels, params['max_tokens'], tokenizer)
        all_logit_biases.append(logit_biases)
        all_verbalized_labels.append(verbalized_labels)
        first_token_of_each_verb_label = [tokenizer.decode(tokenizer.encode(" " + label)[0]) for label in verbalized_labels]
        all_first_token_of_each_verb_label.append(first_token_of_each_verb_label)
    return all_logit_biases, all_verbalized_labels, all_first_token_of_each_verb_label


def prompt_preparation(args, params, subset_train_df=None, test_df=None):
    """prepare the prompt for the API query."""
    # 3. fill the template for final task.
    test_df = deepcopy(test_df)
    test_df = test_df[:args.run_subset_num]
    # reindex
    test_df = test_df.reset_index(drop=True)

    test_df = add_type_constraints(test_df)
    if subset_train_df is not None:
        subset_train_df = add_type_constraints(subset_train_df)

    params = params_update(args, params, test_df)
    final_input_df = fill_prompt_format_re_df(test_df, args.LABEL_VERBALIZER, params, 'sents')
    if args.setting == 'few_shot':
        subset_train_df = fill_prompt_format_re_df(subset_train_df, args.LABEL_VERBALIZER, params, 'sents')
        # 4. retrieval in-context examples for retrieval few-shot.
        # for each example, get demonstrations with retrieval / random.
        if args.type_constrained and args.sampling_strategy == 'random':  # TODO test
            final_input_df = get_type_constrained_random_demos(subset_train_df, final_input_df, args.in_context_size, args.random_seed)
        else:
            final_input_df = get_demonstrations(subset_train_df, final_input_df, args.sampling_strategy, args.random_seed)
    elif args.setting == 'zero_shot':  # directly input for zero-shot!
        assert subset_train_df is None
        final_input_df['data_prompts'] = final_input_df['empty_prompts']
    else:
        raise ValueError('setting should be either few_shot or zero_shot')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    final_input_df['final_input_prompts'] = build_final_input(final_input_df, params, args.engine, tokenizer)
    logit_biases, verbalized_labels, all_first_token_of_each_verb_label = build_logit_biases_df(final_input_df, params, tokenizer)
    final_input_df['logit_biases'] = logit_biases
    final_input_df['verbalized_labels'] = verbalized_labels
    final_input_df['first_token_of_each_verb_label'] = all_first_token_of_each_verb_label
    final_input_df['cost'] = estimate_cost_df(final_input_df, args.engine, tokenizer)

    return final_input_df

