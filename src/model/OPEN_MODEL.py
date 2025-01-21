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
import random

from sentence_transformers import SentenceTransformer, losses
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm

from .utils import LLM_model_load, get_embedding_model_path
from script.utils import my_stop_after_attempt, my_wait_exponential, my_before_sleep

def get_exist_item_set(embedding_path):
    exist_item_set = set()
    for file in os.listdir(embedding_path):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set

class OPEN_MODEL():
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seed = args.seed
        self.debug = args.debug
        if self.seed is not None:
            set_seed(self.seed)
            
        base_generation_LLM = LLM_model_load(self.args, self.args.rec_model, self.args.gpu_id, self.args.use_unsloth)
        self.model = base_generation_LLM['model']
        self.tokenizer = base_generation_LLM['tokenizer']
        
        self.kg_dataset = args.kg_dataset
        
        # embedding model API call
        if args.embedding_model.startswith('text-embedding'):
            pass
        else:
            embedding_model_path = get_embedding_model_path(self.args)
            embedding_model_id = embedding_model_path[args.embedding_model]
            self.embedding_model = SentenceTransformer(embedding_model_id, cache_folder = "/home/work/shchoi/.cache/huggingface/hub")
            
        self.kg_dataset_path = f"../data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
            
        self.name2id = {info['name']: id for id, info in self.id2info.items()}
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        self.item_embedding_path = f"../save/embed/item/{self.kg_dataset}/{args.embedding_model}"
        os.makedirs(self.item_embedding_path, exist_ok=True)
        item_emb_list = []
        id2item_id = []
        
        if len(self.id2info) == len(os.listdir(self.item_embedding_path)):
            print("Loading stored item embeddings...")
            for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path)), desc="Processing item embeddings"):
                item_id = os.path.splitext(file)[0]
                if item_id in self.id2entityid:
                    id2item_id.append(item_id)

                    with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                        embed = json.load(f)
                        item_emb_list.append(embed)
        else:
            """
            id2info file에서 item metatdata를 읽어와서 item embedding을 생성하고 저장하는 코드
            metadata를 text로 변환하고, 이를 embedding model에 넣어서 embedding을 생성한다. 
            생성된 embedding을 {id}.json 파일에 저장한다.
            
            The code reads item metadata from the id2info file, creates and saves item embeddings.
            Convert metadata to text and put it into the embedding model to create an embedding.
            Save the generated embedding in the {id}.json file.
            """
            print(f"Generating item embeddings using {args.embedding_model}...")
            if args.kg_dataset == 'redial': attr_list = ['genre', 'star', 'director']
            elif args.kg_dataset == 'opendialkg': attr_list = ['genre', 'actor', 'director', 'writer']
            else: raise ValueError(f"Invalid kg_dataset: {args.kg_dataset}")
            
            id2text = {}
            for item_id, info_dict in self.id2info.items():
                attr_str_list = [f'Title: {info_dict["name"]}']
                for attr in attr_list:
                    if attr not in info_dict:
                        continue
                    if isinstance(info_dict[attr], list):
                        value_str = ', '.join(info_dict[attr])
                    else:
                        value_str = info_dict[attr]
                    attr_str_list.append(f'{attr.capitalize()}: {value_str}')
                item_text = '; '.join(attr_str_list)
                id2text[item_id] = item_text
            
            item_ids = set(self.id2info.keys()) - get_exist_item_set(self.item_embedding_path)
            while len(item_ids) > 0:
                logger.info(len(item_ids))

                # redial
                if args.kg_dataset == 'redial':
                    batch_item_ids = random.sample(tuple(item_ids), min(args.batch_size, len(item_ids)))
                    batch_texts = [id2text[item_id] for item_id in batch_item_ids]

                # opendialkg
                if args.kg_dataset == 'opendialkg':
                    batch_item_ids = random.sample(tuple(item_ids), min(args.batch_size, len(item_ids)))
                    batch_texts = [id2text[item_id] for item_id in batch_item_ids]

                batch_embeds = self.annotate(args, batch_texts)
                for idx, embed in enumerate(batch_embeds):
                    item_id = batch_item_ids[idx]
                    with open(f'{self.item_embedding_path}/{item_id}.json', 'w', encoding='utf-8') as f:
                        json.dump(embed, f, ensure_ascii=False)
                    
                    if item_id in self.id2entityid:
                        id2item_id.append(item_id)
                        item_emb_list.append(embed)

                item_ids -= get_exist_item_set(self.item_embedding_path)
            
        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        self.chat_recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation. The recommendation list must contain 10 items that are consistent with user preference. The recommendation list can contain items that the dialog mentioned before. The format of the recommendation list is: no. title. Don't mention anything other than the title of items in your recommendation list.'''

    def annotate(self, args: argparse.Namespace, conv_str: str | list[str]):
        if hasattr(self, 'embedding_model'):
            response = self.embedding_model.encode(conv_str)
            if type(response) == np.ndarray:
                response = response.tolist()
            else:
                raise ValueError(f"Invalid response type: {type(response)}")
            if type(conv_str) == list:
                response = [item for item in response]
            else:
                response = response[0]
        else:
            request_timeout = 6
            for attempt in Retrying(
                reraise=True, retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
                wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
            ):
                with attempt:
                    response = args.openai_client.embeddings.create(
                        model = args.embedding_model, input=conv_str, timeout=request_timeout
                    )
                request_timeout = min(30, request_timeout * 2)
                
            response = response.data
            if len(response) == 1:
                response = response[0].embedding
            else:
                response = [item.embedding for item in response]
        return response

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
        
        if self.args.history == 'full':
            for context in context_list:
                conv_str += f"{context['role']}: {context['content']} "
        else:
            recent_turn = (-1) * int(self.args.history)
            for context in context_list[recent_turn:]:
                conv_str += f"{context['role']}: {context['content']} "
            
        conv_embed = self.annotate(self.args, conv_str)
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
            rec_labels_list.append(rec_labels)
            
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
            
            if self.args.history == 'full':
                for context in context_list:
                    conv_str += f"{context['role']}: {context['content']} "
            else:
                recent_turn = (-1) * int(self.args.history)
                for context in context_list[recent_turn:]:
                    conv_str += f"{context['role']}: {context['content']} "
            conv_str_list.append(conv_str)
        
        conv_embed_list = self.annotate(self.args, conv_str_list)
        
        for conv_embed in conv_embed_list:
            conv_embed = conv_embed
            conv_embed = np.asarray(conv_embed).reshape(1, -1)
            
            sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
            rank_arr = np.argsort(sim_mat, axis=-1).tolist()
            rank_arr = np.flip(rank_arr, axis=-1)[:, :self.args.topK]
            item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
            item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
            
            item_rank_arr_list.append(item_rank_arr)
        
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
    
    def get_batch_rewriting_rec(self, conv_dict_list):
        context_list_list = []
        for conv_dict in conv_dict_list:
            context = conv_dict['context']
            context_list = [] # for model
            context_list.append({
                'role': 'system',
                'content': '''You are a extractor to summarize user preference from dialogue. You must follow the instructions below during chat.
If you find features about user preference, you should include them into your summarization.
If there is repetitive contents or just chit-chat irrelevant to user preference, then you can ignore it.
'''
            })
            
            seeker_prompt = '''
You are role-playing as a extractor to summarize user preference from dialogue. Below is Conversation History
#############
        '''

            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    seeker_prompt += f'Seeker: {text}\n'
                else:
                    seeker_prompt += f'Recommender: {text}\n'
            seeker_prompt += "#############"
            context_list.append({
                'role': 'user',
                'content': seeker_prompt
            })
            
            gen_inputs = None
            context_list_list.append(context_list)
        rewritten_user_preference = self.annotate_batch_chat(self.args, context_list_list)
        
        item_rank_arr_list, rec_labels_list = self.get_batch_rec(conv_dict_list)
        
        return gen_inputs, rewritten_user_preference, item_rank_arr_list, rec_labels_list
        