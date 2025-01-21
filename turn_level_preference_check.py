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
from src.model.recommender import RECOMMENDER
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

def get_dialog_emb(dialog: dict, recommender: RECOMMENDER):
    conv_str = ""
    for utterance in dialog:
        conv_str += f"{utterance['role']}: {utterance['content']} "
    
    dialog_emb = recommender.crs_model.annotate(recommender.args, conv_str)
    # request_timeout = 60
    # for attempt in Retrying(
    #     reraise=True, retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
    #     wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    # ):
    #     with attempt:
    #         dialog_emb = openai.Embedding.create(
    #             model=embedding_model, input=conv_str, request_timeout=request_timeout
    #         )
    #     request_timeout = min(30, request_timeout * 2)
    # dialog_emb = dialog_emb['data'][0]["embedding"]
    
    return dialog_emb

def save_turn_level_preference(sample_result_path, result_dict: dict, args):
    sample_result = json.load(open(sample_result_path))
    target_item_title = sample_result['rec'][0]
    target_item_emb = title2emb[target_item_title]
    target_item_emb = torch.tensor(target_item_emb, device = args.device)
    
    id, additional_turn = get_additional_turn_num(sample_result_path)
    _, rec_success, _ = get_result(sample_result_path)
    
    dialogs = slice_dialogue(sample_result_path, int(additional_turn))
    dialog_embeddings = [get_dialog_emb(dialog, embedding_model) for dialog in dialogs]
    dialog_embeddings = torch.tensor(dialog_embeddings, device = args.device) # (additional_turn + 1) * H
    
    all_item_embs = torch.tensor(list(title2emb.values()), device = args.device) # item_num * H
    
    sims = torch.matmul(dialog_embeddings, target_item_emb)
    dialog_norm = torch.norm(dialog_embeddings, dim = 1)
    target_norm = torch.norm(target_item_emb)
    cosine_sims = sims / (dialog_norm * target_norm)
    
    all_sims = torch.matmul(dialog_embeddings, all_item_embs.T)
    all_item_norm = torch.norm(all_item_embs, dim = 1)
    all_cosine_sim = all_sims / torch.outer(dialog_norm, all_item_norm)
    
    dot_WAR = sims - torch.mean(all_sims, dim = 1)
    topK_sims, _ = torch.topk(all_sims, 100)
    dot_topK_WAR = sims - torch.mean(topK_sims, dim = 1)
    
    cosine_WAR = cosine_sims - torch.mean(all_cosine_sim, dim = 1)
    topK_cosine_sims, _ = torch.topk(all_cosine_sim, 100)
    cosine_topK_WAR = cosine_sims - torch.mean(topK_cosine_sims, dim = 1)
    
    probabilities = torch.exp(all_sims)
    total = torch.sum(probabilities, dim = 1)
    probabilities = probabilities / total.unsqueeze(1)
    entropies = -torch.sum(probabilities * torch.log(probabilities), dim = 1)
    
    local_result = []
    for idx, _ in enumerate(dialogs):

        dot_sim = sims[idx].item()
        dot_sim_above_avg = dot_WAR[idx].item()
        dot_sim_above_topK_avg = dot_topK_WAR[idx].item()
        cosine_sim = cosine_sims[idx].item()
        cosine_sim_above_avg = cosine_WAR[idx].item()
        cosine_sim_above_topK_avg = cosine_topK_WAR[idx].item()
        entropy = entropies[idx].item()
        
        local_result.append({
            'turn': idx,
            'dot_sim': dot_sim,
            'dot_sim_above_avg': dot_sim_above_avg,
            'dot_sim_above_topK_avg': dot_sim_above_topK_avg,
            'cosine_sim': cosine_sim,
            'cosine_sim_above_avg': cosine_sim_above_avg,
            'cosine_sim_above_topK_avg': cosine_sim_above_topK_avg,
            'entropy': entropy,
            'rec_success': rec_success,
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
    parser.add_argument('--gpu_id', default = 0, type=int)
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'])
    
    args = parser.parse_args()
    args.kg_dataset = args.dataset.split('_')[0]
    args.device = f"cuda:{args.gpu_id}"
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
            
    folder_path = f"{root_dir}/save_5/chat/{args.model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}"
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
                reraise=True, retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
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

    threads = []
    processes = []
    
    with Manager() as manager:
        shared_dict = manager.dict()  # Create a shared dictionary

        for sample_result_path in tqdm(results_path):
            p = Process(target=save_turn_level_preference, args=(sample_result_path, shared_dict, args))
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
        with open(f'{root_dir}/preference_data/turn_level_preference_{args.model}_{args.rec_model}_top{args.topK}_{args.history}_history_{args.dataset}_{args.eval_data_size}_{args.eval_strategy}.json', 'w') as f:
            json.dump(result_dict, f)