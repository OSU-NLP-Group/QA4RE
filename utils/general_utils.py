import random


def print_args(args):
    print('Arguments:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')


def check_example(df, print_col_list, num=3):
    # randomly select three row to check.
    indexes = random.sample(range(len(df)), min(num, len(df)))
    for i in indexes:
        row = df.iloc[i]
        for col in print_col_list:
            print(f'{col}: {row[col]}')
        print("#" * 100)
    return


def error_analysis(df, label_col='label', pred_col='rel_predictions', print_col_list=['test_ready_prompts', 'predictions', 'rel_predictions', 'label', 'index2rel', 'correct_template_indexes']):
    for i, row in df.iterrows():
        if row[label_col] != row[pred_col]:
            print(f"Error at {i}")
            for col in print_col_list:
                print(col, row[col])
            print("#" * 100)
    return
