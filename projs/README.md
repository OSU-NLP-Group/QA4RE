## for Vanilla
```
# in the vanillaRE folder
DATA=TACRED
mode=test # or dev
model=text-davinci-003

# remove --debug for series run
python vanilla_re.py --mode $mode --no_class_explain --ex_name vanilla --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --engine $model --debug

# for vanilla + TEMP ablation
# remove --debug for entire run
python vanilla_re.py --mode $mode --ex_name vanilla-class-explain --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --engine $model --debug

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine, args.run_setting)

```


## for QA4RE
```
# in the QA4RE folder
DATA=TACRED
mode=test # or dev
model=text-davinci-003
# remove --debug for entire run
python qa4re.py --mode $mode --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name qa4re_prompt_config.yaml --engine $model --debug

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine, args.run_setting)
```
