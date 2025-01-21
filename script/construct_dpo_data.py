import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import tiktoken

import openai
import nltk
from loguru import logger
from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from tqdm import tqdm
from multiprocessing import Process

import sys
sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER
from utils import annotate_completion, get_instruction, get_entity_data, process_for_baselines, get_exist_dialog_set, get_dialog_data, set_seed, get_exist_dpo_data
from simulate import construct_DPO_data, batch_construct_DPO_data

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
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--beam_num', type=int, default=8)
    parser.add_argument('--split', type=str, default='train', choices=['train'])
    parser.add_argument('--use_lora_at_inference', action='store_true')
    parser.add_argument('--topK', type=int, default=50)
    parser.add_argument('--history', type=str, default='full')
    # remove argument for conventional CRS (refer to iEVALM official repository)
    
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    args.device = f'cuda:{args.gpu_id}'
    with open (f"{args.root_dir}/secret/api.json", "r") as f:
        secret_data = json.load(f)
    openai.api_key = secret_data['openai']
    # save_dir = f'{args.root_dir}/save_{args.turn_num}/chat/{args.crs_model}_{args.rec_model}/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    save_dir = f'{args.root_dir}/save_{args.turn_num}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_lora_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_train_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed)
    
    # recommender
    recommender = RECOMMENDER(args)

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset) # TODO: instruction 받는 형태를 하나로 통일
    id2entity, entity_list = get_entity_data(args)
    dialog_id2data = get_dialog_data(args)
    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dpo_data(save_dir)
    dialog_id_list = list(dialog_id_set)
    if 'process' in args.inference_mode:
        if args.inference_mode == 'multi-process':
            processes = []
        for dialog_id in tqdm(dialog_id_list, desc="Processing Dialogs"):
            data = dialog_id2data[dialog_id]
            if args.inference_mode == 'single-process':
                construct_DPO_data(dialog_id, data, seeker_instruction_template, args, recommender, id2entity, entity_list, save_dir)

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
                batch_construct_DPO_data(copy.deepcopy(dialog_sub_list), copy.deepcopy(dialog_data_list), seeker_instruction_template, args, recommender, id2entity, entity_list, save_dir)
                
                # Update progress bar
                pbar.update(1)
                # except Exception as e:
                #     print(e)
                #     print(f'Error in dialog_id_list: {dialog_id_sub_list}')