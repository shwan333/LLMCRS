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

import sys
sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER
from utils import annotate_completion, get_instruction, get_entity_data, process_for_baselines

warnings.filterwarnings('ignore')

def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        exist_id_set.add(file_id)
    return exist_id_set

def get_dialog_data(args: argparse.Namespace) -> dict:
    dialog_id2data = {}
    with open(f'{args.root_dir}/data/{args.dataset}/{args.eval_data_size}_test_data_processed_{args.eval_strategy}.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line
            
    return dialog_id2data

if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt'])
    parser.add_argument('--embedding_model', type=str, default = "text-embedding-3-small", choices=["text-embedding-3-small"])
    parser.add_argument('--rec_model', type=str, default = "gpt-4o-mini", choices=["gpt-4o-mini"])
    parser.add_argument('--user_model', type=str, default = "gpt-4o-mini", choices=["gpt-4o-mini"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--resp_max_length', type=int)
    # remove argument for conventional CRS (refer to iEVALM official repository)
    
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    with open (f"{args.root_dir}/secret/api.json", "r") as f:
        secret_data = json.load(f)
    openai.api_key = secret_data['openai']
    
    save_dir = f'{args.root_dir}/save_{args.turn_num}/chat/{args.crs_model}/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}' 
    os.makedirs(save_dir, exist_ok=True)
    random.seed(args.seed)
    
    # recommender
    recommender = RECOMMENDER(args)

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset) # TODO: instruction 받는 형태를 하나로 통일
    id2entity, entity_list = get_entity_data(args)
    dialog_id2data = get_dialog_data(args)
    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dialog_set()
    dialog_id_list = list(dialog_id_set)
    
    for dialog_id in tqdm(dialog_id_list, desc="Processing Dialogs"):
        dialog_id = random.choice(dialog_id_list)
        data = dialog_id2data[dialog_id]
        conv_dict = copy.deepcopy(data) # for model
        context = conv_dict['context']

        goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
        goal_item_str = ', '.join(goal_item_list)
        seeker_instruct = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
        seeker_prompt = '''
            Conversation History
            #############
        '''
        context_dict = [] # for save

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
                seeker_prompt += f'Seeker: {text}\n'
            else:
                role_str = 'assistant'
                seeker_prompt += f'Recommender: {text}\n'
            context_dict.append({
                'role': role_str,
                'content': text
            })
            
        rec_success = False
        recommendation_template = "I would recommend the following items: {}:"

        for i in range(0, args.turn_num):
            # rec only
            rec_items, rec_labels = recommender.get_rec(conv_dict)
            
            for rec_label in rec_labels:
                if rec_label in rec_items[0]:
                    rec_success = True
                    break
            # rec only
            _, recommender_text = recommender.get_conv(conv_dict)
            recommender_text = process_for_baselines(args, recommender_text, id2entity, rec_items)
            
            if rec_success == True or i == args.turn_num - 1:
                rec_items_str = ''
                for j, rec_item in enumerate(rec_items[0][:50]):
                    rec_items_str += f"{j+1}: {id2entity[rec_item]}\n"
                recommendation_template = recommendation_template.format(rec_items_str)
                recommender_text = recommendation_template + recommender_text
            
            # public 
            recommender_resp_entity = get_entity(recommender_text, entity_list)
            
            conv_dict['context'].append(recommender_text)
            conv_dict['entity'] += recommender_resp_entity
            conv_dict['entity'] = list(set(conv_dict['entity']))
            
            context_dict.append({
                'role': 'assistant',
                'content': recommender_text,
                'entity': recommender_resp_entity,
                'rec_items': rec_items[0],
                'rec_success': rec_success
            })
            
            seeker_prompt += f'Recommender: {recommender_text}\n'
            
            # seeker
            year_pattern = re.compile(r'\(\d+\)')
            goal_item_no_year_list = [year_pattern.sub('', rec_item).strip() for rec_item in goal_item_list]
            seeker_text = annotate_completion(args, seeker_instruct, seeker_prompt).strip()
            
            seeker_response_no_movie_list = []
            for sent in nltk.sent_tokenize(seeker_text):
                use_sent = True
                for rec_item_str in goal_item_list + goal_item_no_year_list:
                    if fuzz.partial_ratio(rec_item_str.lower(), sent.lower()) > 90: # TODO: 이름만 없애도록 수정
                        use_sent = False
                        break
                if use_sent is True:
                    seeker_response_no_movie_list.append(sent)
            seeker_response = ' '.join(seeker_response_no_movie_list)
            if not rec_success:
                seeker_response = 'Sorry, ' + seeker_response
            seeker_prompt += f' {seeker_response}\n'
            
            # public
            seeker_resp_entity = get_entity(seeker_text, entity_list)
            
            context_dict.append({
                'role': 'user',
                'content': seeker_text,
                'entity': seeker_resp_entity,
            })
            
            conv_dict['context'].append(seeker_text)
            conv_dict['entity'] += seeker_resp_entity
            conv_dict['entity'] = list(set(conv_dict['entity']))
            
            if rec_success:
                break
        
        # score persuativeness
        conv_dict['context'] = context_dict
        data['simulator_dialog'] = conv_dict

        # save
        with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f: 
            json.dump(data, f, ensure_ascii=False, indent=2)