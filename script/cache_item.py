import json
import os
import random
import typing
from argparse import ArgumentParser

import openai
from loguru import logger
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from utils import my_stop_after_attempt, my_wait_exponential, my_before_sleep

def annotate(item_text_list):
    request_timeout = 6
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.BadRequestError, openai.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.Embedding.create(
                model='text-embedding-ada-002', input=item_text_list, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)

    return response


def get_exist_item_set():
    exist_item_set = set()
    for file in os.listdir(save_dir):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set


if __name__ == '__main__':
    """
    id2info file에서 item metatdata를 읽어와서 item embedding을 생성하고 저장하는 코드
    metadata를 text로 변환하고, 이를 embedding model에 넣어서 embedding을 생성한다. 
    생성된 embedding을 {id}.json 파일에 저장한다.
    
    The code reads item metadata from the id2info file, creates and saves item embeddings.
    Convert metadata to text and put it into the embedding model to create an embedding.
    Save the generated embedding in the {id}.json file.
    """
    
    parser = ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', type=str, choices=['redial', 'opendialkg'])
    args = parser.parse_args()

    openai.api_key = args.api_key
    batch_size = args.batch_size
    dataset = args.dataset

    save_dir = f'../save/embed/item/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'../data/{dataset}/id2info.json', encoding='utf-8') as f:
        id2info = json.load(f)

    # redial
    if dataset == 'redial':
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Star: {', '.join(info['star'])}",
                f"Director: {', '.join(info['director'])}", f"Plot: {info['plot']}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'star', 'director']

    # opendialkg
    if dataset == 'opendialkg':
        item_texts = []
        for info_dict in id2info.values():
            item_attr_list = [f'Name: {info_dict["name"]}']
            for attr, value_list in info_dict.items():
                if attr != 'title':
                    item_attr_list.append(f'{attr.capitalize()}: ' + ', '.join(value_list))
            item_text = '; '.join(item_attr_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'actor', 'director', 'writer']

    id2text = {}
    for item_id, info_dict in id2info.items():
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

    item_ids = set(id2info.keys()) - get_exist_item_set()
    while len(item_ids) > 0:
        logger.info(len(item_ids))

        # redial
        if dataset == 'redial':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # opendialkg
        if dataset == 'opendialkg':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        batch_embeds = annotate(batch_texts)['data']
        for embed in batch_embeds:
            item_id = batch_item_ids[embed['index']]
            with open(f'{save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                json.dump(embed['embedding'], f, ensure_ascii=False)

        item_ids -= get_exist_item_set()
