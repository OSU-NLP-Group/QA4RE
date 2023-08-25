"""Haven't checked retrieval few shot settings"""
import sys
import os
from argparse import ArgumentParser
# from glob import glob

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.join(sys.path[0], '../../..'))

from utils.general_utils import *
from projs.re_utils import *
from utils.hf_model_utils import *
from projs.vanillaRE.vanilla_re_utils import *

def call_hf_llm_df(args, input_ready_df, output_dir):
    """Call huggingface LLMs with examples in the input_ready_df."""
    predictions = []
    model_generations = []
    time_list = []
    # load model
    tokenizer, model = load_huggingface_model(args.model, args.num_gpus, args.use_t5)

    for i, row in tqdm(input_ready_df.iterrows()):
        time_start = time.time()
        final_input_prompt = row['final_input_prompts']
        verbalized_labels = row['verbalized_labels']
        # first_token_of_each_verb_label = row['first_token_of_each_verb_label']
        # call Huggingface model
        if args.use_t5:
            generation = call_hf_llm_enc_dec(model, tokenizer, final_input_prompt, verbalized_labels)
        else:
            generation = call_hf_llm_causal(model, tokenizer, final_input_prompt, verbalized_labels)
        model_generations.append(generation)
        # todo, check 
        # prediction = index2rel[generation]
        predictions.append(generation)

        time_list.append(time.time() - time_start)

    input_ready_df['model_generations'] = model_generations
    input_ready_df['predictions'] = predictions
    input_ready_df['time'] = time_list
    input_ready_df.to_csv(os.path.join(output_dir, f'subset.{args.mode}.output.csv'), index=False)

    return input_ready_df


def run_vanilla_re(args):
    if args.mode == 'dev':
        subset_test = pd.read_csv(os.path.join(args.data_root, args.dataset, args.dev_subset_name), sep='\t')
    elif args.mode == 'test':
        subset_test = pd.read_csv(os.path.join(args.data_root, args.dataset, args.test_subset_name), sep='\t')
    prompt_params, output_dir, experiment_num = exp_dir_setup(args)  # every run, except debug mode, we will create a new experiment dir.

    print("prompt_params: ", prompt_params)

    if args.run_setting.startswith('retrieval_few_shot'):  # train set as retrieval pool for retrieval few shot setting
        subset_train = pd.read_csv(os.path.join(args.data_root, args.dataset, args.train_subset_name), sep='\t')
    elif args.run_setting == 'zero_shot' or args.run_setting == 'fixed_few_shot':
        subset_train = None
    else:
        raise NotImplementedError

    input_ready_df = prompt_preparation(args, prompt_params, subset_train, subset_test)
    check_example(input_ready_df, ['final_input_prompts', 'verbalized_label'])
    # output_df = call_gpt_engine_df(args, input_ready_df, prompt_params, output_dir)
    output_df = call_hf_llm_df(args, input_ready_df, output_dir)
    check_example(output_df, ['final_input_prompts', 'model_generations', 'predictions'])

    metric_dict = evaluate_re(output_df, args.LABEL_VERBALIZER, args.POS_LABELS,)
    # sum up the time and cost
    metric_dict['time'] = output_df['time'].sum()
    metric_dict['cost'] = output_df['cost']
    metric_dict.update(prompt_params)
    metric_dict.update(vars(args))

    pd.DataFrame([metric_dict]).to_csv(output_dir + '/' + args.mode + '.metrics')

    batch_re_eval_print(output_df, 'predictions', args.LABEL_VERBALIZER, args.POS_LABELS, return_results=False)

    output_df.to_csv(output_dir + '/' + args.mode + '.gpt3.output.csv', sep='\t')

    return


def main():
    parser = ArgumentParser()
    # general args
    parser.add_argument('--debug', action='store_true')  # for debug.
    parser.add_argument('--task', default='RE')
    parser.add_argument('--method', default='vanilla_re')  # default, no change.
    parser.add_argument('--data_root', type=str, default='../../data/')
    # parser.add_argument('--engine', type=str, default='text-davinci-003')
    parser.add_argument('--model', type=str, default='google/flan-t5-small')
    parser.add_argument('--use_t5', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--ex_name', default='vanilla_re')  # folder name to save.

    # dataset args
    parser.add_argument('--dataset', type=str, default='TACRED')
    parser.add_argument('--train_subset_name', type=str, default='train_subset.csv')
    parser.add_argument('--dev_subset_name', type=str, default='dev_subset.csv')
    parser.add_argument('--test_subset_name', type=str, default='test_subset.csv')
    parser.add_argument('--train_subset_samples', type=int, default=100)
    parser.add_argument('--dev_subset_samples', type=int, default=250)
    parser.add_argument('--test_subset_samples', type=int, default=1000)

    # experiment args
    parser.add_argument('--mode', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--prompt_config_name', type=str, default='prompt_config.yaml')
    parser.add_argument('--type_constrained', action='store_true')
    parser.add_argument('--no_class_explain', action='store_true')  # used for vanilla + TEMP ablation experiment
    parser.add_argument('--in_context_size', type=int, default=0)  # number of demonstrations, 0 for zero-shot setting
    parser.add_argument('--run_setting', choices=['retrieval_few_shot_zero_seed', 'fixed_few_shot', 'zero_shot'], default='zero_shot')  # default true few shot setting. use train set to cross validate hyperparameters
    parser.add_argument('--sampling_strategy', default='random', choices=['roberta-large', 'random'])  # sentence embedding model for retrieval few shot setting

    args = parser.parse_args()
    # for using GPT-3 series code.
    args.engine = args.model
    args = process_arguments(args)

    # pd.read_csv(os.path.join(args.data_root, args.dataset, args.dev_subset_name), sep='\t')
    if args.mode == 'dev':
        print(f"....Running {args.dev_subset_samples} Dev set evaluation....")
    else:
        print(f"....Running {args.test_subset_samples} Test set evaluation....")

    run_vanilla_re(args)


if __name__ == '__main__':
    main()
