import argparse
import copy
import json
import os
import random
import time
import warnings
import torch
import openai
from tqdm import tqdm

import sys
sys.path.append("..")

from src.model.utils import get_entity, LLM_model_load
from src.model.recommender import RECOMMENDER
from utils import annotate_completion, get_instruction, get_entity_data, process_for_baselines, get_exist_dialog_set, get_dialog_data, set_seed, get_exist_dpo_data, check_proprietary_model
from simulate import batch_construct_DPO_data

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'openmodel'])
    parser.add_argument('--embedding_model', type=str, default = "text-embedding-3-small")
    parser.add_argument('--rec_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--user_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resp_max_length', type=int, default = 128)
    parser.add_argument('--inference_mode', type=str, choices = ['single-process', 'multi-process', 'batch'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--user_gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--beam_num', type=int, default=8)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--use_lora_at_inference', action='store_true')
    parser.add_argument('--topK', type=int, default=10)
    parser.add_argument('--reward_func_topK', type=int, default=10, choices =[-1, 1, 5, 10, 20, 30, 50, 100])
    parser.add_argument('--history', type=str, default='full')
    parser.add_argument('--rank', type=int, default=32)
    
    # remove argument for conventional CRS (refer to iEVALM official repository)
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    args.device = f'cuda:{args.gpu_id}'
    if 'unsloth' in args.rec_model: args.use_unsloth = True
    else: args.use_unsloth = False
    if 'unsloth' in args.user_model: args.user_use_unsloth = True
    else: args.user_use_unsloth = False

    if check_proprietary_model(args.user_model):
        pass
    else:
        user_LLM = LLM_model_load(args, args.user_model, args.user_gpu_id, args.user_use_unsloth)
        args.user_LLM = user_LLM['model']
        args.user_tokenizer = user_LLM['tokenizer']

    with open (f"{args.root_dir}/secret/api.json", "r") as f:
        secret_data = json.load(f)
    openai.api_key = secret_data['openai']
    # save_dir = f'{args.root_dir}/save_{args.turn_num}/chat/{args.crs_model}_{args.rec_model}/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    save_dir = f'{args.root_dir}/save_{args.turn_num}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_{args.split}_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed)
    
    # recommender
    recommender = RECOMMENDER(args)

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset) # TODO: instruction 받는 형태를 하나로 통일
    dialog_id2data = get_dialog_data(args)
    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dpo_data(save_dir)
    dialog_id_list = list(dialog_id_set)
    if 'process' in args.inference_mode:
        raise NotImplementedError()

    elif args.inference_mode =='batch':
        total_iterations = len(dialog_id_list) // args.batch_size  # Since sample_num is 8
        if len(dialog_id_list) % args.batch_size != 0:
            total_iterations += 1

        with tqdm(total=total_iterations, desc="Processing dialogs") as pbar:
            while len(dialog_id_list) > 0:
                # Sampling
                sample_num = min(args.batch_size, len(dialog_id_list))  # Ensure we don't sample more than available
                dialog_sub_list = list(random.sample(dialog_id_list, sample_num))
                dialog_id_list = [x for x in dialog_id_list if x not in dialog_sub_list]
                
                # Data preparation
                dialog_data_list = [dialog_id2data[dialog_id] for dialog_id in dialog_sub_list]
                batch_construct_DPO_data(copy.deepcopy(dialog_sub_list), copy.deepcopy(dialog_data_list), seeker_instruction_template, args, recommender, save_dir)
                
                # Update progress bar
                pbar.update(1)
                
    
    # train_dpo.py
    from datasets import load_dataset, Dataset
    from trl import DPOConfig, DPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    import torch, json, os, argparse
    from src.model.utils import get_model_path

    import torch
    from transformers import TrainingArguments
    from trl import DPOTrainer
    from train_dpo import create_triplets
    print("=====================================")
    print("start training DPO")
    gradient_accumulation_steps = 32
    epochs = 3
    lr = 5e-5
    output_dir = f"pref_tuned/{args.rec_model}_{args.embedding_model}_{args.dataset}_rank_{args.rank}_grad_acc_{gradient_accumulation_steps}_lr_{lr}_epochs_{epochs}"

    if 'unsloth' in args.rec_model: args.use_unsloth = True
    else: args.use_unsloth = False
    if 'unsloth' in args.user_model: args.user_use_unsloth = True
    else: args.user_use_unsloth = False
    args.root_dir = os.getcwd()
    # find the parent directory of root_dir
    while args.root_dir.split('/')[-1] != 'iEvaLM-CRS':
        args.root_dir = os.path.dirname(args.root_dir)
    print(f'root_dir: {args.root_dir}')
    save_dir = f'/home/work/shchoi/iEvaLM-CRS/save_5/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_train_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    print(f'save_dir: {save_dir}')

    dialog_data_list = []
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isdir(file_path):
            continue
        dialog_data = json.load(open(file_path, 'r'))
        dialog_data_list.append(dialog_data)
        
    args.kg_dataset = args.dataset.split('_')[1]
    model_id = get_model_path(args)[args.rec_model]
    print(args.gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        cache_dir="/home/work/shchoi/.cache/huggingface/hub",
        torch_dtype=torch.bfloat16,
        ).to(f'cuda:{args.gpu_id}')
    # import time
    # time.sleep(10)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/work/shchoi/.cache/huggingface/hub")
    if "Llama" in args.rec_model:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(dialog_data_list)
    train_dataset = train_dataset.map(create_triplets, remove_columns=train_dataset.features, fn_kwargs={"tokenizer": tokenizer})

    # print size of train_dataset
    print(f"Number of dialogues: {len(train_dataset)}")

    # print size of train_dataset
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*4,
        lora_dropout=0.05,
        target_modules=[
            'q_proj',
            'v_proj',
            'k_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


    training_args = DPOConfig(
        output_dir=output_dir, 
        logging_steps=10,
        per_device_train_batch_size=1,  # Start with small batch size
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increase effective batch size 
        num_train_epochs = epochs,
        # max_steps=1000,
        # save_steps=100,
        # gradient_checkpointing=True,
        learning_rate=lr,
        bf16=True,
        warmup_ratio = 0.1,
        # remove_unused_columns=False,
        # run_name="dpo_llama2",
        seed=42,
    )
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, peft_config=peft_config)
    trainer.train()