import json
import random
from typing import List, Union, Optional

import torch
from rapidfuzz import fuzz, process
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.bettertransformer import BetterTransformer

special_tokens_dict = {
    'pad_token': '<|pad|>'
}

def get_embedding_model_path():
    model_path = {
        'Snowflake/snowflake-arctic-embed-s': '/home/work/shchoi/.cache/huggingface/hub/models--Snowflake--snowflake-arctic-embed-s/snapshots/e596f507467533e48a2e17c007f0e1dacc837b33',
        'Snowflake/snowflake-arctic-embed-s-tuned': '/home/work/shchoi/iEvaLM-CRS/output/user_Llama-3.1-8B-Instruct/emb_Snowflake/snowflake-arctic-embed-s/openmodel_Llama-3.2-1B-Instruct_top10_full_history/opendialkg_eval/full_non_repeated/train/checkpoint-105',
    }
    
    return model_path

def get_model_path():
    model_path = {
            'Llama-3.1-8B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
            'Llama-3.2-3B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72',
            'Llama-3.2-1B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',
            'Qwen2.5-1.5B-Instruct':'/home/work/shchoi/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
            'Qwen2.5-3B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1',
            'Qwen2.5-7B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75',
            'Qwen2.5-14B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8',
            'Falcon3-3B-Instruct': '/home/work/shchoi/.cache/huggingface/hub/models--tiiuae--Falcon3-3B-Instruct/snapshots/debc1ce254ecb90b7b485b7292e0e98e773a356d',
            'unsloth-Llama-3.2-3B-Instruct': 'unsloth/Llama-3.2-3B-Instruct',
            'unsloth-Llama-3.2-1B-Instruct': 'unsloth/Llama-3.2-1B-Instruct',
            'unsloth-Llama-3.1-8B-Instruct': 'unsloth/Meta-Llama-3.1-8B-Instruct',
        }
    
    return model_path

def get_model_list():
    model_path = get_model_path()
    model_list = []
    for server in model_path.keys():
        for model in model_path[server].keys():
            model_list.append(model)
    
    return model_list

def LLM_model_load(args, load_model, gpu_id, use_unsloth):
    device = f"cuda:{gpu_id}"
    model_path = get_model_path()
    
    # LLM load 
    generation_model_id = model_path[load_model]
    print(f'generation_model_id: {load_model}')
    if generation_model_id != None:
        if use_unsloth and gpu_id == args.gpu_id:
            print(f'use unsloth')
            from unsloth import FastLanguageModel
            from peft import PeftModel
            if args.use_lora_at_inference:
                generation_model, generation_model_tokenizer = FastLanguageModel.from_pretrained(generation_model_id, load_in_4bit = False, cache_dir = "/home/work/shchoi/.cache/huggingface/hub", device_map=device)
                generation_model = PeftModel.from_pretrained(generation_model, f"/home/work/shchoi/iEvaLM-CRS/full_{load_model}_rank_8_grad_acc_32_lr_5e-05_epochs_3/checkpoint-1344")
                FastLanguageModel.for_inference(generation_model)
            else:
                generation_model, generation_model_tokenizer = FastLanguageModel.from_pretrained(generation_model_id, load_in_4bit = False, cache_dir = "/home/work/shchoi/.cache/huggingface/hub", device_map=device)
                FastLanguageModel.for_inference(generation_model)
        else:
            print(f'use huggingface')
            generation_model_tokenizer = AutoTokenizer.from_pretrained(generation_model_id, padding_side='left', device_map = device,)
            generation_model = AutoModelForCausalLM.from_pretrained(
                generation_model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            if args.use_lora_at_inference:
                print(f'use lora')
                from peft import PeftModel
                generation_model = PeftModel.from_pretrained(generation_model, f"/home/work/shchoi/iEvaLM-CRS/full_{load_model}_rank_8_grad_acc_32_lr_5e-05_epochs_3/checkpoint-1344")
                generation_model = generation_model.model.merge_and_unload()
        if "Llama" in load_model:
            generation_model_tokenizer.pad_token = generation_model_tokenizer.eos_token # 아마 이거 때문에 qwen이나 Falcon에서 학습이 안 되었던 듯
        generation_model.generation_config.pad_token_id = generation_model_tokenizer.pad_token_id
        generation_model_terminators = [
            generation_model_tokenizer.eos_token_id,
            # system_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        base_generation_LLM = {
            'model': generation_model,
            'tokenizer': generation_model_tokenizer,
            'terminators': generation_model_terminators,
        }
    
    return base_generation_LLM

def load_jsonl_data(file):
    data_list = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def simple_collate(batch):
    return batch


def sample_data(data_list, shot=1, debug=False, number_for_debug=320):
    if debug:
        data_list = data_list[:number_for_debug]

    if shot < 1:
        data_idx = random.sample(range(len(data_list)), int(len(data_list) * shot))
        data_list = [data_list[idx] for idx in data_idx]
    elif shot > 1:
        data_idx = range(int(shot))
        data_list = [data_list[idx] for idx in data_idx]

    return data_list


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_id: int = 0,
    pad_tail: bool = True,
    device: torch.device = torch.device('cpu'),
    debug: bool = False,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(max(lens), 1)
    if debug and max_length is not None:
        t = max(t, max_length)

    output = torch.full((n, t), fill_value=pad_id, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask=None):
        """

        Args:
            x (bs, seq_len, hs)
            mask (bs, seq_len): False for masked token.

        Returns:
            (bs, hs)
        """
        attn = self.attn(x)  # (bs, seq_len, 1)
        if mask is not None:
            attn += (~mask).unsqueeze(-1) * -1e4
        attn = F.softmax(attn, dim=-1)
        x = attn.transpose(1, 2) @ x  # (bs, 1, hs)
        x = x.squeeze(1)
        return x


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].detach().clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# dbpedia get entity
# def get_entity(text, SPOTLIGHT_CONFIDENCE):
#     DBPEDIA_SPOTLIGHT_ADDR = " http://0.0.0.0:2222/rest/annotate"
#     headers = {"accept": "application/json"}
#     params = {"text": text, "confidence": SPOTLIGHT_CONFIDENCE}

#     response = requests.get(DBPEDIA_SPOTLIGHT_ADDR, headers=headers, params=params)
#     response = response.json()
#     return (
#         [f"<{x['@URI']}>" for x in response["Resources"]]
#         if "Resources" in response
#         else []
#     )

# rapidfuzz get entity
def get_entity(text, entity_list):
    extractions = process.extract(text, entity_list, scorer=fuzz.WRatio, limit=20)
    extractions = [extraction[0] for extraction in extractions if extraction[1] >= 90]
    return extractions