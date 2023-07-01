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


def fill_prompt_format_re_df(args, df, prompt_config, sent_col='masked_sents'):
    """
    prompts are full, with sentence and label.
    empty_prompts with only sentences.
    So prompts are used for demonstrations while empty_prompts are used for test examples
    """
    prompts = []
    empty_prompts = []
    index2rels = []
    all_choices = []
    correct_choices = []

    for i, row in df.iterrows():
        # sent = row[sent_col]
        ent1 = row['ent1']
        ent2 = row['ent2']
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        sent = row[sent_col].replace('ENT1', f"[{ent1}]").replace('ENT2', f"[{ent2}]")
        # label = all_label_verbalizer[str(label)]
        label = row['label']
        type_cons = row['type_cons']
        prompt_sample_structure = prompt_config['example_format'] + ' {}'
        empty_prompt_sample_structure = prompt_config['example_format']
        example = sent + "\nOptions:\n"
        correct_templates = args.LABEL_TEMPLATES[label]
        correct_template_index = []
        prediction_range = []
        valid_relations = args.VALID_CONDITIONS_REV[type_cons] + [args.NOTA_RELATION]
        start_chr = 'A'
        index2rel = {}
        for valid_relation in valid_relations:
            for template in args.LABEL_TEMPLATES[valid_relation]:
                # filled_template = template.format(subj=ent1, obj=ent2)
                filled_template = template.format(subj=f"[{ent1}]", obj=f"[{ent2}]")
                if template in correct_templates:
                    correct_template_index.append(start_chr)
                prediction_range.append(start_chr)
                example += f"{start_chr}. {filled_template}\n"
                index2rel[start_chr] = valid_relation
                start_chr = chr(ord(start_chr) + 1)

        if correct_template_index == []:  # some type constraint cases not included
            print("Warning: no correct template found!")
            print("sent: ", sent)
            print("ent1: ", ent1)
            print("ent2: ", ent2)
            print("ent1_type: ", ent1_type)
            print("ent2_type: ", ent2_type)
            print("label: ", label)
            print("valid_relations: ", valid_relations)
            print("correct_templates: ", correct_templates)

            print("#" * 50)
            correct_template_index.append(chr(ord(start_chr) - 1))
        correct_index = correct_template_index[0]

        prompt = prompt_sample_structure.format(example, correct_index)
        empty_prompt = empty_prompt_sample_structure.format(example)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)

        correct_choices.append(correct_template_index)
        index2rels.append(index2rel)
        all_choices.append(prediction_range)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['correct_choices'] = correct_choices
    df['index2rels'] = index2rels
    df['all_choices'] = all_choices

    return df


def build_final_input(df, params, engine, tokenizer):
    """build the test ready prompts for the API call."""
    max_tokens = params['max_tokens']
    build_final_input = []
    if engine in GPT_MODEL_MAX_TOKEN_DICT:
            max_allowed_token_num = GPT_MODEL_MAX_TOKEN_DICT[engine]
    else:
        print(f"Warning: unknown engine {engine}, make sure you are using huggingface LLMs or update the GPT_MODEL_MAX_TOKEN_DICT.")
        max_allowed_token_num = 2000 #  token num for FLAN-T5

    for i, row in tqdm(df.iterrows()):
        task_instructions = params['task_instructions'].strip()
        final_input_prompt = task_instructions + '\n\n' + row['data_prompts']

        tokens = tokenizer.encode(final_input_prompt)
        if len(tokens) + max_tokens > max_allowed_token_num:  # only necessary for few-shot setting.
            print(f"Warning: the prompt is too long for {engine}, will remove all the demonstrations.")
            while len(tokens) + max_tokens > max_allowed_token_num:  # remove all the demos.
                final_input_prompt = task_instructions + '\n\n' + row['empty_prompts']
                tokens = tokenizer.encode(final_input_prompt)
        build_final_input.append(final_input_prompt.strip())  # make sure not ends with ' '.
    return build_final_input


def build_logit_biases_df(df, params, tokenizer):
    """build the logit biases for each row in the dataframe.
    return: all_logit_biases, all_first_token_of_each_verb_label
    """
    all_logit_biases = []
    all_first_token_of_each_verb_label = []
    for i, row in df.iterrows():
        verbalized_labels = row['verbalized_labels']
        logit_biases = build_logit_biases(verbalized_labels, params['max_tokens'], tokenizer)
        all_logit_biases.append(logit_biases)
        first_token_of_each_verb_label = [tokenizer.decode(tokenizer.encode(" " + label)[0]) for label in verbalized_labels]
        all_first_token_of_each_verb_label.append(first_token_of_each_verb_label)
    return all_logit_biases, all_first_token_of_each_verb_label


def get_prediction(args, params, openai_api_response, all_choices, correct_choices, index2rel, engine, first_token_of_each_verb_label):
    # correct_template_indexes
    """get the prediction from the openai api response.
    return generated_content, and parsed prediction.
    """
    def get_pred_from_generated_content(generated_content, all_choices):
        """get the prediction from the generated content, for multiple token generation and chatCompletion"""
        candidates = []
        for choice in all_choices:
            if choice in generated_content:
                candidates.append(choice)
            if len(candidates) == 0:  # not generated, randomly choose one.
                pred = random.choice(all_choices)
            elif len(candidates) > 1:  # get the first one in generation order
                start_indexes = [generated_content.index(can) for can in candidates]
                pred = candidates[np.argmin(start_indexes)]
            else:
                pred = candidates[0]
        return pred

    if len(correct_choices) == 0:  # answer is not in the choices.  # type constrains wrong, < 0.3%
        return args.nota_relation

    if GPT_MODEL_TYPE_DICT[engine] == 'text':  # able to get prob and some logit.
        text = openai_api_response['choices'][0]['text']
        if params['max_tokens'] == 1:
            probs = []
            top_logprobs = openai_api_response['choices'][0]['logprobs']['top_logprobs'][0]
            for token in first_token_of_each_verb_label:
                if token in top_logprobs:
                    probs.append(top_logprobs[token])
                else:
                    probs.append(-100)
            pred_index = all_choices[np.argmax(probs)]
        else:
            generated_content = openai_api_response['choices'][0]['text']
            pred_index = get_pred_from_generated_content(generated_content, all_choices)
    elif GPT_MODEL_TYPE_DICT[engine] == 'chat':
        generated_content = openai_api_response['choices'][0]['message']['content']
        text = generated_content
        pred_index = get_pred_from_generated_content(generated_content, all_choices)
    else:
        raise ValueError('GPT_MODEL_TYPE_DICT[engine] should be text or chat')

    return text, index2rel[pred_index]



def prompt_preparation(args, params, subset_train_df=None, test_df=None):
    """prepare the prompt for the API query."""
    # 3. fill the template for final task.
    test_df = deepcopy(test_df)
    test_df = test_df[:args.run_subset_num]
    # reindex
    test_df = test_df.reset_index(drop=True)

    # add constrains to both dfs
    test_df = add_type_constraints(test_df)
    if subset_train_df is not None:
        subset_train_df = add_type_constraints(subset_train_df)

    final_input_df = fill_prompt_format_re_df(args, test_df, params, 'masked_sents')
    if args.setting == 'few_shot':
        subset_train_df = fill_prompt_format_re_df(args, subset_train_df, params, 'masked_sents')
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
    # labels are answer indexes in QA4RE
    final_input_df['verbalized_labels'] = final_input_df['all_choices']
    logit_biases, all_first_token_of_each_verb_label = build_logit_biases_df(final_input_df, params, tokenizer)
    final_input_df['logit_biases'] = logit_biases
    final_input_df['first_token_of_each_verb_label'] = all_first_token_of_each_verb_label
    final_input_df['cost'] = estimate_cost_df(final_input_df, args.engine, tokenizer)

    return final_input_df
