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

import sys
sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER

warnings.filterwarnings('ignore')

def my_before_sleep(retry_state):
    logger.debug(
        f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number

def annotate_completion(args: argparse.Namespace, instruct: str, prompt: str, logit_bias=None) -> str:
    if logit_bias is None:
        logit_bias = {}

    prompt += '''
        #############
    '''
    messages = [{'role': 'system', 'content': instruct}, {'role': 'user', 'content': prompt}]

    request_timeout = 20
    for attempt in Retrying(
            reraise=True,
            retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
    ):
        with attempt:
            response = openai.ChatCompletion.create(
                model= args.user_model, messages= messages, temperature= 0, max_tokens= 128, request_timeout=request_timeout,
            )['choices'][0]['message']['content']
        request_timeout = min(300, request_timeout * 2)

    return response

def get_entity_data(args: argparse.Namespace) -> tuple[dict, list]:
    with open(f'{args.root_dir}/data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f: # TODO: 1) 이해하기 쉽게 수정 및 2) hard coding 없애기
        entity2id = json.load(f)
    
    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())

    return id2entity, entity_list

def get_dialog_data(args: argparse.Namespace) -> dict:
    dialog_id2data = {}
    with open(f'{args.root_dir}/data/{args.dataset}/test_data_processed.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line
            
    return dialog_id2data

def process_for_baselines(args: argparse.Namespace, recommender_text: str, id2entity:dict, rec_items:list) -> str:
    # barcor
    if args.crs_model == 'barcor':
        recommender_text = recommender_text.lstrip('System;:')
        recommender_text = recommender_text.strip()
    
    # unicrs
    if args.crs_model == 'unicrs':
        if args.dataset.startswith('redial'):
            movie_token = '<movie>'
        else:
            movie_token = '<mask>'
        recommender_text = recommender_text[recommender_text.rfind('System:') + len('System:') + 1 : ]
        for i in range(str.count(recommender_text, movie_token)):
            recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[i]], 1)
        recommender_text = recommender_text.strip()
        
    return recommender_text

def get_instruction(dataset):
    if dataset.startswith('redial'):
        item_with_year = True
    elif dataset.startswith('opendialkg'):
        item_with_year = False

    if item_with_year is True:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''

    return recommender_instruction, seeker_instruction_template