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
from utils import annotate_completion, get_instruction, get_entity_data, process_for_baselines, get_exist_dialog_set, get_dialog_data

def simulate_iEvaLM(dialog_id: str, data: dict, seeker_instruction_template: str, args: argparse.Namespace, recommender: RECOMMENDER, id2entity: dict, entity_list: list, save_dir: str):
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