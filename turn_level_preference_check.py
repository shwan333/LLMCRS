# 유저 취향 파악 -> 추천 성공률 향상으로 이어지나?
from preference_utils import *

import glob
import json
import openai
import os
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import threading
from multiprocessing import Process, Manager
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm
import argparse
from script.utils import my_wait_exponential, my_stop_after_attempt, my_before_sleep

def calculate_entropy(similarity_list):
    # calculate softmax
    probabilities = []
    for similarity in similarity_list:
        probabilities.append(np.exp(similarity))
    total = sum(probabilities)
    probabilities = [prob / total for prob in probabilities]
    
    # calculate entropy
    entropy = 0
    for prob in probabilities:
        entropy += prob * np.log(prob)
    return -entropy

def slice_dialogue(sample_result_path, additional_turn:int):
    sample_result = json.load(open(sample_result_path))
    total_turn_num = len(sample_result['simulator_dialog']['context'])
    initial_turn = total_turn_num - (additional_turn * 2)
    dialogs = [sample_result['simulator_dialog']['context'][:initial_turn + (idx * 2)] for idx in range(additional_turn+1)]
    
    return dialogs

def get_dialog_emb(dialog: dict, embedding_model: str):
    conv_str = ""
    for utterance in dialog:
        conv_str += f"{utterance['role']}: {utterance['content']} "
        
    request_timeout = 60
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            dialog_emb = openai.Embedding.create(
                model=embedding_model, input=conv_str, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)
    dialog_emb = dialog_emb['data'][0]["embedding"]
    
    return dialog_emb

def save_turn_level_preference(sample_result_path, result_dict: dict):
    instance_name = sample_result_path.split('/')[-1].split('.')[0]
    dialog_id = instance_name
    sample_result = json.load(open(sample_result_path))
    target_item_title = sample_result['rec'][0]
    target_item_emb = title2emb[target_item_title]
    dialog_id_list.append(dialog_id)
    
    id, additional_turn = get_additional_turn_num(sample_result_path)

    _, rec_success, _ = get_result(sample_result_path)
    dialogs = slice_dialogue(sample_result_path, int(additional_turn))
    dialog_embeddings = [get_dialog_emb(dialog, embedding_model) for dialog in dialogs]
    local_result = []        
    for idx, dialog_emb in enumerate(dialog_embeddings):
        title2sim = {}
        for title, item_emb in title2emb.items():
            title2sim[title] = cosine_similarity(dialog_emb, item_emb)

        dot_sim = dot_product_similarity(dialog_emb, target_item_emb)
        rank = calculate_rank(list(title2sim.values()), title2sim[target_item_title])
        softmax_value = softmax(list(title2sim.values()), title2sim[target_item_title])
        cosine_sim = cosine_similarity(dialog_emb, target_item_emb)
        consine_sim_above_avg = cosine_sim - np.mean(list(title2sim.values()))
        hyperbolic_sim = hyperbolic_tangent(dialog_emb, target_item_emb)
        entropy = calculate_entropy(list(title2sim.values()))
        
        local_result.append({
            'turn': idx,
            'dot_sim': dot_sim,
            'rec_success': rec_success,
            'cosine_sim': round(cosine_sim, 4),
            'consine_sim_above_avg': round(consine_sim_above_avg, 4),
            'hyperbolic_sim': round(hyperbolic_sim, 4),
            'softmax_value': softmax_value,
            'rank': rank,
            'entropy': entropy,
        })
    result_dict[id] = local_result

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--version', type=str, default='v0.1')
    # parser.add_argument('--base_LLM', type=str, default='gpt4o-mini', choices=['gpt4o-mini', "Llama-3.1-8B-Instruct", 'Qwen2.5-14B-Instruct'])
    # parser.add_argument('--planning_LLM', type=str, default='gpt4o-mini')
    parser.add_argument('--model', default = 'chatgpt', type=str)
    parser.add_argument('--rec_model', default = 'gpt-4o-mini', type=str)
    parser.add_argument('--dataset', default = 'opendialkg_eval', type=str, choices=['opendialkg_eval', 'redial_eval'])
    parser.add_argument('--topK', default = 10, type=int)
    parser.add_argument('--history', default = 'full', type=str)
    args = parser.parse_args()
    args.kg_dataset = args.dataset.split('_')[0]

    # base_LLM = 'my_ppo_merge_0_11-14-08-51'
    # base_LLM = 'Qwen2.5-14B-Instruct'
    # base_LLM = 'Llama-3.2-1B-Instruct'
    # base_LLM = "Llama-3.1-8B-Instruct"
    # base_LLM = 'gpt4o-mini'
    # planning_LLM = "gpt4o-mini"
    eval_data_size = 'full'
    eval_strategy = 'non_repeated'
    # dataset = 'opendialkg'
    # dataset = 'redial'
    split = 'test'
    root_dir = os.getcwd()
    
    with open (f"{root_dir}/secret/api.json", "r") as f:
        secret_data = json.load(f)
        
    file_path = f"{root_dir}/data/{args.kg_dataset}/id2info.json"
    with open (file_path, "r") as f:
        id2info = json.load(f)
        
    title2info = {}
    for idx, data in id2info.items():
        title2info[data['name']] = data

    openai.api_key = secret_data['openai']

    # preference elcitation hyperparameters
    metric = "cosine"
    embedding_model = "text-embedding-3-small"
            
    folder_path = f"{root_dir}/save_5/chat/{args.model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{eval_data_size}_{eval_strategy}"
    print(f'Target folder: {folder_path}')

    # Get data for success result for each beam
    results_path = glob.glob(folder_path + '/*.json')
    print(f'The number of results: {len(results_path)}')
    
    emb_file_path = f"{root_dir}/data/{args.dataset}/title2emb_{embedding_model}.json"
    if os.path.exists(emb_file_path):
        with open(emb_file_path, "r") as f:
            title2emb = json.load(f)
    else:
        title2emb = {}
        for title, info in tqdm(title2info.items(), desc="Item Embedding Generation"):
            target_item_info = str(info)
            request_timeout = 60
            
            for attempt in Retrying(
                reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
                wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
            ):
                with attempt:
                    target_item_emb = openai.Embedding.create(
                        model=embedding_model, input=target_item_info, request_timeout=request_timeout
                )
                request_timeout = min(30, request_timeout * 2)
            target_item_emb = target_item_emb['data'][0]["embedding"]
            title2emb[title] = target_item_emb

        emb_file_path = f"{root_dir}/data/{args.dataset}/title2emb_{embedding_model}.json"
        json.dump(title2emb, open(emb_file_path, "w"))

    dialog_id_list = []
    threads = []
    processes = []
    
    with Manager() as manager:
        shared_dict = manager.dict()  # Create a shared dictionary

        for sample_result_path in tqdm(results_path):
            p = Process(target=save_turn_level_preference, args=(sample_result_path, shared_dict,))
            p.start()
            processes.append(p)
            if len(processes) == 32:
                for p in processes:
                    p.join()
                processes = []
            
        if len(processes) > 0:
            for p in processes:
                p.join()
        
        result_dict = dict(shared_dict)
        with open(f'{root_dir}/preference_data/turn_level_preference_{args.model}_{args.dataset}_{eval_data_size}_{eval_strategy}.json', 'w') as f:
            json.dump(result_dict, f)