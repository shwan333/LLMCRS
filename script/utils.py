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
import asyncio
import openai
import nltk
import numpy as np
import torch
from loguru import logger
from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base

import sys
sys.path.append("..")

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
        if retry_state.outcome == openai.APITimeoutError:
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
        if retry_state.outcome == openai.APITimeoutError:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number
    
def get_exist_dialog_set(save_dir: str) -> set:
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        exist_id_set.add(file_id)
    return exist_id_set

def get_exist_dpo_data(save_dir: str) -> set:
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        file_token = file_id.split('_')
        dialog_id = '_'.join(file_token[:-1])
        exist_id_set.add(dialog_id)
    return exist_id_set

def get_dialog_data(args: argparse.Namespace) -> dict:
    dialog_id2data = {}
    with open(f'{args.root_dir}/data/{args.dataset}/{args.eval_data_size}_{args.split}_data_processed_{args.eval_strategy}.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line
            
    return dialog_id2data

def annotate_completion(args: argparse.Namespace, instruct: str, prompt: str, logit_bias=None) -> str:
    if logit_bias is None:
        logit_bias = {}

    prompt += '''
        #############
    '''
    messages = [{'role': 'system', 'content': instruct}, {'role': 'user', 'content': prompt}]
    
    if check_proprietary_model(args.user_model):
        request_timeout = 20
        for attempt in Retrying(
                reraise=True,
                retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
                wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
        ):
            with attempt:
                response = args.openai_client.chat.completions.create(
                    model= args.user_model, messages= messages, temperature= 0, max_tokens= 128, request_timeout=request_timeout, seed = args.seed
                ).choices[0].message.content
            request_timeout = min(300, request_timeout * 2)
    else:
        formatted_messages = args.user_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            padding = True,
        )
            
        # Tokenize without padding first to find max length
        encodings = [args.user_tokenizer.encode(text) for text in formatted_messages]
        max_length = max(len(encoding) for encoding in encodings)
            
        input_ids = args.user_tokenizer(
            formatted_messages,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(f"cuda:{args.user_gpu_id}")

        outputs = args.user_LLM.generate(
            **input_ids,
            num_return_sequences = 1,
            max_new_tokens=128,
            eos_token_id=args.user_tokenizer.eos_token_id,
            temperature=1.0,
            early_stopping=False,
            min_length = -1, # 얘만 원래 없음.
            top_k = 0.0, # 얘만 다름.
            top_p = 1.0,
            do_sample = False,
            pad_token_id = args.user_tokenizer.eos_token_id,
        )

        responses = []
        for idx in range(outputs.shape[0]):
            response = args.user_tokenizer.decode(outputs[idx][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
            responses.append(response.strip())
        
        if len(responses) == 1:
            response = responses[0]
        else:
            raise ValueError(f"Multiple responses generated: {responses}")

    return response

def annotate_batch_completion(args: argparse.Namespace, instruct_list: list[str], prompt_list: list[str], logit_bias=None) -> list[str]:
    async def fetch_chat_completion(args, messages, logit_bias):
        request_timeout = 60
        for attempt in Retrying(
            reraise=True,
            retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60),
            stop=(my_stop_after_attempt(8)),
            before_sleep=my_before_sleep
        ):
            with attempt:
                response = await args.openai_async_client.chat.completions.create(  # Use `acreate` for async call
                    model=args.user_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=128,
                    timeout=request_timeout,
                )
        request_timeout = min(300, request_timeout * 2)
        return response.choices[0].message.content

    async def main_async(prompts):
        tasks = [fetch_chat_completion(args, prompt, logit_bias) for prompt in prompts]
        return await asyncio.gather(*tasks)  # Gather all async tasks

    messages_list = [
        [{'role': 'system', 'content': instruct}, {'role': 'user', 'content': prompt + "\n#############"}]
        for instruct, prompt in zip(instruct_list, prompt_list)
    ]
    if check_proprietary_model(args.user_model):
        # Prepare messages for API calls
        
        responses = asyncio.run(main_async(messages_list))
    else:
        formatted_messages_list = args.user_tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            padding = True,
        )
            
        # Tokenize without padding first to find max length
        encodings = [args.user_tokenizer.encode(text) for text in formatted_messages_list]
        max_length = max(len(encoding) for encoding in encodings)
            
        input_ids = args.user_tokenizer(
            formatted_messages_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(f"cuda:{args.user_gpu_id}")

        outputs = args.user_LLM.generate(
            **input_ids,
            num_return_sequences = 1,
            max_new_tokens=128,
            eos_token_id=args.user_tokenizer.eos_token_id,
            temperature=1.0,
            early_stopping=False,
            min_length = -1, # 얘만 원래 없음.
            top_k = 0.0, # 얘만 다름.
            top_p = 1.0,
            do_sample = False,
            pad_token_id = args.user_tokenizer.eos_token_id,
        )

        responses = []
        for idx in range(outputs.shape[0]):
            response = args.user_tokenizer.decode(outputs[idx][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
            responses.append(response.strip())        

    # Run the asynchronous tasks and return the results
    return responses

def get_entity_data(args: argparse.Namespace) -> tuple[dict, list]:
    with open(f'{args.root_dir}/data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f: # TODO: 1) 이해하기 쉽게 수정 및 2) hard coding 없애기
        entity2id = json.load(f)
    
    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())

    return id2entity, entity_list

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
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must provide you next utterance based on given conversation. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must provide you next utterance based on given conversation. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''

    return recommender_instruction, seeker_instruction_template

def filter_seeker_text(seeker_text: str, goal_item_list: list[str], rec_success: bool) -> str:
    year_pattern = re.compile(r'\(\d+\)')
    goal_item_no_year_list = [year_pattern.sub('', rec_item).strip() for rec_item in goal_item_list]
    
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
        
    return seeker_response

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 사용하는 경우
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def convert_1d_to_2d(lst, b, k):
    return [lst[i:i + k] for i in range(0, b * k, k)]

def check_proprietary_model(model_name: str) -> bool:
    if 'gpt' in model_name:
        return True
    else:
        return False
    
def cosine_similarity(list1, list2):
    # Convert lists to numpy arrays for vector operations
    vector1 = np.array(list1)
    vector2 = np.array(list2)

    # Compute cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity

def batch_cosine_similarity(query_vector, item_matrix):
    # Convert query vector to numpy array
    query_vector = np.array(query_vector)
    
    # Compute dot products for all items at once
    dot_products = np.dot(item_matrix, query_vector)
    
    # Compute norms
    item_norms = np.linalg.norm(item_matrix, axis=1)  # Need norm for each item
    query_norm = np.linalg.norm(query_vector)         # Only one query norm needed
    
    # Compute similarities
    similarities = dot_products / (item_norms * query_norm)
    return similarities