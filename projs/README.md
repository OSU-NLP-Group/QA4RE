## for Vanilla
### OpenAI GPT Engine
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

### Huggingface LLMs
```
# in the vanillaRE folder
DATA=TACRED
mode=test # or dev
model=google/flan-t5-small

# remove --debug for series run
python vanilla_re_hf_llm.py --mode $mode --no_class_explain --ex_name vanilla --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --model $model --use_t5 --debug


# for vanilla + TEMP ablation
# remove --debug for entire run
python vanilla_re_hf_llm.py --mode $mode --ex_name vanilla-class-explain --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --model $model --use_t5 --debug

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine.replace('/', '-'), args.run_setting)
```


## for QA4RE
### OpenAI GPT Engine
```
# in the QA4RE folder
DATA=TACRED
mode=test # or dev
model=text-davinci-003
# remove --debug for entire run
python qa4re.py --mode $mode --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name qa4re_prompt_config.yaml --engine $model --debug

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine, args.run_setting)
```


### Huggingface LLMs
```
# in the QA4RE folder
DATA=TACRED
mode=test # or dev
model=google/flan-t5-small
# remove --debug for entire run
python qa4re_hf_llm.py --mode $mode --dataset ${DATA} --run_setting zero_shot --prompt_config_name qa4re_prompt_config.yaml --model $model --use_t5 --debug

# type_constrained is used by default
# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine.replace('/', '-'), args.run_setting)
```

