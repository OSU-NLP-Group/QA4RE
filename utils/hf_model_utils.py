import torch
import random
import numpy as np
from tqdm import tqdm
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinNewTokensLengthLogitsProcessor, MinLengthLogitsProcessor
from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_t5_model(model_name: str, **kwargs):
    """
    Local T5 Model."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        offload_folder="offload", 
        torch_dtype=torch.float16, **kwargs)
        
    return tokenizer, model


def load_casual_model(model_name: str, **kwargs):
    """
    Local Casual Model like Alpaca."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        offload_folder="offload", 
        torch_dtype=torch.float16, **kwargs)
    return tokenizer, model


def load_huggingface_model(model_name: str, num_gpus: int, t5_model: bool):
    """
    Load a model from the HuggingFace library.
    """
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "45GiB" for i in range(num_gpus)},
        }

    if t5_model:
        tokenizer, model = load_t5_model(model_name, **kwargs)
    else:
        tokenizer, model = load_casual_model(model_name, **kwargs)
    if num_gpus == 1:
        model.cuda()
        
    return tokenizer, model


def add_to_lattice(sequence, lattice={}):
    if len(sequence) == 0:
        return {}
    else:
        element = sequence[0]

        lattice[element] = add_to_lattice(sequence[1:], lattice.get(element,{}))

        return lattice


def call_hf_llm_enc_dec(model, tokenizer, prompt, label_space):

    inputs = tokenizer([prompt])
    
    label_prefix_dict = {}
    label_lengths = []
    
    for label_ind, label in enumerate(label_space):
        label_inp = tokenizer(label,add_special_tokens=False).input_ids
        label_lengths.append(len(label_inp))
        label_inp += [tokenizer.eos_token_id]

        label_prefix_dict = add_to_lattice(label_inp, label_prefix_dict)

    def constrain_fnc(batch, input_ids):

        rel_prefix = []

        for tok in input_ids:
            if tok not in tokenizer.all_special_ids:
                rel_prefix.append(tok)        
        
        options = label_prefix_dict
        
        for tok in rel_prefix:
            options = options.get(int(tok),{})
            
        return list(options.keys())
    
    max_length = int(np.max(label_lengths))
    min_length = int(np.min(label_lengths))
    
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(min_length, eos_token_id=tokenizer.eos_token_id),
        ]
    )

    output_ids = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                prefix_allowed_tokens_fn=constrain_fnc,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
#     if outputs_unc in labels and outputs_unc != outputs:
#         ipdb.set_trace()
            
    if outputs not in label_space:
        ipdb.set_trace()
        print('random choice {}'.format(outputs))
        outputs = random.choice(label_space)
        
    return outputs

def call_hf_llm_causal(model, tokenizer, prompt, label_space):

    inputs = tokenizer([prompt])
    
    label_prefix_dict = {}
    label_lengths = []
    
    for label_ind, label in enumerate(label_space):
        label_inp = tokenizer(label,add_special_tokens=False).input_ids
        label_lengths.append(len(label_inp))
        label_inp += [tokenizer.eos_token_id]
        
        label_prefix_dict = add_to_lattice(label_inp, label_prefix_dict)
    
    prefix_to_ignore = len(inputs.input_ids[0])
     
    def constrain_fnc(batch, input_ids):
        
        rel_prefix = input_ids[prefix_to_ignore:]
        
        options = label_prefix_dict
        
        for tok in rel_prefix:
            options = options.get(int(tok),{})
            
        return list(options.keys())
    
    max_length = int(np.max(label_lengths))
    min_length = int(np.min(label_lengths))
    
    logits_processor = LogitsProcessorList(
        [
            MinNewTokensLengthLogitsProcessor(prefix_to_ignore, min_length, eos_token_id=tokenizer.eos_token_id),
        ]
    )

    output_ids = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                prefix_allowed_tokens_fn=constrain_fnc,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs = tokenizer.batch_decode(output_ids[:,len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()
    
    output_ids_unc = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs_unc = tokenizer.batch_decode(output_ids_unc[:,len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()
        
#     if outputs_unc in labels and outputs_unc != outputs:
#         ipdb.set_trace()
            
    if outputs not in label_space:
        ipdb.set_trace()
        print('random choice {}'.format(outputs))
        outputs = random.choice(label_space)
        
    return outputs
