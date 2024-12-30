import typing
import torch
import json
import os
import argparse

import nltk
import openai
import tiktoken
import numpy as np
import asyncio

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm

from .utils import LLM_model_load

def my_before_sleep(retry_state):
    logger.debug(f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


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

def annotate(args: argparse.Namespace, conv_str: str) -> str:
    request_timeout = 6
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.Embedding.create(
                model=args.embedding_model, input=conv_str, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)

    return response

def batch_annotate(args: argparse.Namespace, conv_str_list: list[str]) -> str:
    async def fetch_chat_completion(args, conv_str):
        request_timeout = 6
        for attempt in Retrying(
            reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
        ):
            with attempt:
                response = openai.Embedding.create(
                    model=args.embedding_model, input=conv_str, request_timeout=request_timeout
                )
            request_timeout = min(30, request_timeout * 2)

        return response
    
    async def main_async(prompts):
        tasks = [fetch_chat_completion(args, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

    # Execute all API calls concurrently
    responses = asyncio.run(main_async(conv_str_list))
        
    return responses

# def annotate_chat(args: argparse.Namespace, messages_list: list[list[dict]], logit_bias=None) -> str:
#     formatted_messages_list = self.tokenizer.apply_chat_template(
#         messages_list,
#         tokenize=False,
#         add_generation_prompt=True,
#         padding = True,
#     )
        
#     # Tokenize without padding first to find max length
#     encodings = [self.tokenizer.encode(text) for text in formatted_messages_list]
#     max_length = max(len(encoding) for encoding in encodings)
        
#     input_ids = self.tokenizer(
#         formatted_messages_list,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt",
#         return_attention_mask=True,
#         return_token_type_ids=False,
#     ).to(args.device)

#     outputs = self.model.generate(
#         **input_ids,
#         num_return_sequences = 1,
#         max_new_tokens=args.max_new_tokens,
#         eos_token_id=self.tokenizer.eos_token_id,
#         # top_k = 3, 
#         temperature=1.0,
#         # num_beams = 3,
#         early_stopping=False,
#         min_length = -1, # 얘만 원래 없음.
#         top_k = 0.0, # 얘만 다름.
#         top_p = 1.0,
#         do_sample = False,
#         pad_token_id = self.tokenizer.eos_token_id,
#     )

#     responses = []
#     for idx in range(outputs.shape[0]):
#         response = self.tokenizer.decode(outputs[idx][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
#         responses.append(response.strip())
    
#     return responses

class OPEN_MODEL():
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seed = args.seed
        self.debug = args.debug
        if self.seed is not None:
            set_seed(self.seed)
            
        base_generation_LLM = LLM_model_load(self.args)
        self.model = base_generation_LLM['model']
        self.tokenizer = base_generation_LLM['tokenizer']
        
        self.kg_dataset = args.kg_dataset
        
        self.kg_dataset_path = f"../data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
            
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        self.item_embedding_path = f"../save/embed/item/{self.kg_dataset}"
        
        item_emb_list = []
        id2item_id = []
        for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path)), desc="Processing item embeddings"):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        self.chat_recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation. The recommendation list must contain 10 items that are consistent with user preference. The recommendation list can contain items that the dialog mentioned before. The format of the recommendation list is: no. title. Don't mention anything other than the title of items in your recommendation list.'''

    def annotate_sample_chat(self, args: argparse.Namespace, messages: list[dict], logit_bias=None) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(args.device)

        # outputs = self.model.generate(
        #     input_ids,
        #     num_beams = args.beam_num,
        #     num_return_sequences = args.beam_num,
        #     max_new_tokens=args.resp_max_length,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     do_sample=False,
        #     temperature=args.temperature,
        #     top_p=1.0,
        #     early_stopping=False  # 모든 빔이 EOS에 도달하면 조기 종료
        # )
        
        outputs = self.model.generate(
            input_ids,
            num_return_sequences = args.beam_num,
            max_new_tokens=args.resp_max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            early_stopping=False  # 모든 빔이 EOS에 도달하면 조기 종료
        )
        
        real_ouputs = [outputs[idx][input_ids.shape[-1]:] for idx in range(args.beam_num)]
        responses = [self.tokenizer.decode(real_ouputs[idx], skip_special_tokens=True) for idx in range(args.beam_num)]
        
        return responses
    
    def annotate_batch_chat(self, args: argparse.Namespace, messages_list: list[list[dict]], logit_bias=None) -> str:
        formatted_messages_list = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            padding = True,
        )
            
        # Tokenize without padding first to find max length
        encodings = [self.tokenizer.encode(text) for text in formatted_messages_list]
        max_length = max(len(encoding) for encoding in encodings)
            
        input_ids = self.tokenizer(
            formatted_messages_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(args.device)

        outputs = self.model.generate(
            **input_ids,
            num_return_sequences = 1,
            max_new_tokens=args.resp_max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            # top_k = 3, 
            temperature=1.0,
            # num_beams = 3,
            early_stopping=False,
            min_length = -1, # 얘만 원래 없음.
            top_k = 0.0, # 얘만 다름.
            top_p = 1.0,
            do_sample = False,
            pad_token_id = self.tokenizer.eos_token_id,
        )

        responses = []
        for idx in range(outputs.shape[0]):
            response = self.tokenizer.decode(outputs[idx][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def annotate_sample_batch_chat(self, args: argparse.Namespace, messages_list: list[list[dict]], logit_bias=None) -> str:
        formatted_messages_list = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            padding = True,
        )
            
        # Tokenize without padding first to find max length
        encodings = [self.tokenizer.encode(text) for text in formatted_messages_list]
        max_length = max(len(encoding) for encoding in encodings)
            
        input_ids = self.tokenizer(
            formatted_messages_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(args.device)

        outputs = self.model.generate(
            **input_ids,
            num_return_sequences = args.beam_num,
            max_new_tokens=args.resp_max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = True,
            temperature=args.temperature,
            top_p = 1.0,
            early_stopping=False,
            min_length = -1, # 얘만 원래 없음.
            top_k = 0.0, # 얘만 다름.
        )

        responses = []
        for idx in range(outputs.shape[0]):
            response = self.tokenizer.decode(outputs[idx][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def get_rec(self, conv_dict):
        
        rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]
        
        context = conv_dict['context']
        context_list = [] # for model
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        conv_str = ""
        
        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "
            
        conv_embed = annotate(self.args, conv_str)['data'][0]['embedding']
        conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :self.args.topK]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        
        return item_rank_arr, rec_labels
    
    def get_batch_rec(self, conv_dict_list: list[dict]):
        
        item_rank_arr_list = []
        rec_labels_list = []
        conv_str_list = []
        
        for conv_dict in conv_dict_list:
            rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]
            
            context = conv_dict['context']
            context_list = [] # for model
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                context_list.append({
                    'role': role_str,
                    'content': text
                })
            
            conv_str = ""
            
            # for context in context_list[-2:]:
            for context in context_list:
                conv_str += f"{context['role']}: {context['content']} "
                
            conv_embed = annotate(self.args, conv_str)['data'][0]['embedding']
            conv_embed = np.asarray(conv_embed).reshape(1, -1)
            
            sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
            rank_arr = np.argsort(sim_mat, axis=-1).tolist()
            rank_arr = np.flip(rank_arr, axis=-1)[:, :self.args.topK]
            item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
            item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
            
            item_rank_arr_list.append(item_rank_arr)
            rec_labels_list.append(rec_labels)
        
        return item_rank_arr_list, rec_labels_list
    
    def get_sample_conv(self, conv_dict) -> list[str]:
        context = conv_dict['context']
        context_list = [] # for model
        context_list.append({
            'role': 'system',
            'content': self.chat_recommender_instruction
        })
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        gen_inputs = None
        gen_str_list = self.annotate_sample_chat(self.args, context_list)
    
        return gen_str_list

    # def get_conv(self, conv_dict_list) -> list[str]:
        
    #     context = conv_dict['context']
    #     context_list = [] # for model
    #     context_list.append({
    #         'role': 'system',
    #         'content': self.chat_recommender_instruction
    #     })
        
    #     for i, text in enumerate(context):
    #         if len(text) == 0:
    #             continue
    #         if i % 2 == 0:
    #             role_str = 'user'
    #         else:
    #             role_str = 'assistant'
    #         context_list.append({
    #             'role': role_str,
    #             'content': text
    #         })
        
    #     gen_inputs = None
    #     gen_str = annotate_chat(self.args, context_list)
        
    #     return gen_inputs, gen_str
    
    def get_batch_conv(self, conv_dict_list) -> list[str]:
        
        context_list_list = []
        for conv_dict in conv_dict_list:
            context = conv_dict['context']
            context_list = [] # for model
            context_list.append({
                'role': 'system',
                'content': self.chat_recommender_instruction
            })
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                context_list.append({
                    'role': role_str,
                    'content': text
                })
            
            gen_inputs = None
            context_list_list.append(context_list)
        gen_str_list = self.annotate_batch_chat(self.args, context_list_list)
        
        return gen_inputs, gen_str_list
    
    def get_sample_batch_conv(self, conv_dict_list) -> list[str]:
        
        context_list_list = []
        for conv_dict in conv_dict_list:
            context = conv_dict['context']
            context_list = [] # for model
            context_list.append({
                'role': 'system',
                'content': self.chat_recommender_instruction
            })
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                context_list.append({
                    'role': role_str,
                    'content': text
                })
            
            gen_inputs = None
            context_list_list.append(context_list)
        gen_str_list = self.annotate_sample_batch_chat(self.args, context_list_list)
        
        return gen_inputs, gen_str_list