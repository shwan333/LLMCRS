import argparse
import copy
import json
import sys
import random
import os
import numpy as np
from multiprocessing import Process
import time
import torch
sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER
from utils import annotate_completion, annotate_batch_completion, process_for_baselines, filter_seeker_text, convert_1d_to_2d, cosine_similarity, batch_cosine_similarity

def simulate_iEvaLM(dialog_id: str, data: dict, seeker_instruction_template: str, args: argparse.Namespace, recommender: RECOMMENDER, id2entity: dict, entity_list: list, save_dir: str):
    conv_dict = copy.deepcopy(data) # for model
    context = conv_dict['context']

    goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
    goal_item_str = ', '.join(goal_item_list)
    seeker_instruct = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
    seeker_prompt = args.seeker_prompt
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
        seeker_text = annotate_completion(args, seeker_instruct, seeker_prompt).strip()
        seeker_response = filter_seeker_text(seeker_text, goal_item_list, rec_success)
        seeker_prompt += f'Seeker: {seeker_response}\n'
        
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
        
def batch_simulate_iEvaLM(dialog_id_list: list[str], dialog_data_list: list[dict], seeker_instruction_template: str, args: argparse.Namespace, recommender: RECOMMENDER, id2entity: dict, entity_list: list, save_dir: str):
    """
    simulate function for multiple dialogs with iEvaLM model. It should be noted that this function can not be used for baselines.

    """
    conv_dict_list = []
    context_dict_list = []
    seeker_instruct_list = []
    rec_success_list = [False] * len(dialog_id_list)
    seeker_prompt_list = []
    goal_item_list_list = []
    import time
    start_time = time.time()
    for idx, dialog_id in enumerate(dialog_id_list):
        conv_dict = copy.deepcopy(dialog_data_list[idx]) # for model
        context = conv_dict['context']

        goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
        goal_item_str = ', '.join(goal_item_list)
        seeker_instruct = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
        seeker_prompt = args.seeker_prompt
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
        seeker_prompt_list.append(seeker_prompt)
        conv_dict_list.append(conv_dict)
        context_dict_list.append(context_dict)
        seeker_instruct_list.append(seeker_instruct)
        goal_item_list_list.append(goal_item_list)
        
    recommendation_template = "I would recommend the following items: {}:"
    print(f'initialization_time: {time.time() - start_time}s')
    start_time = time.time()
    for i in range(0, args.turn_num):
        print(f"Turn {i+1}")
        # Time for recommender 
        # recommendation
        print(f'Get recommendation')
        rec_items_list, rec_labels_list, item_names_list = recommender.get_batch_rec(conv_dict_list)
        
        # check if rec success
        for idx, rec_labels in enumerate(rec_labels_list):
            for rec_label in rec_labels:
                if rec_label in rec_items_list[idx][0]:
                    rec_success_list[idx] = True
                    break
        del rec_labels_list
        
        print(f'Rec time: {time.time() - start_time}s')
        start_time = time.time()
        
        # conversation
        print(f'Get conversation')
        _, recommender_text_list = recommender.get_batch_conv(conv_dict_list, item_names_list)
        
        # post-process for conversation
        for idx, rec_success in enumerate(rec_success_list):
            if rec_success == True or i == args.turn_num - 1:
                rec_items_str = ''
                for j, rec_item in enumerate(rec_items_list[idx][0][:50]):
                    rec_items_str += f"{j+1}: {id2entity[rec_item]}\n"
                recommendation_template = recommendation_template.format(rec_items_str)
                recommender_text_list[idx] = recommendation_template + recommender_text_list[idx]
        
            recommender_resp_entity = get_entity(recommender_text_list[idx], entity_list)
        
            conv_dict_list[idx]['context'].append(recommender_text_list[idx])
            conv_dict_list[idx]['entity'] += recommender_resp_entity
            conv_dict_list[idx]['entity'] = list(set(conv_dict_list[idx]['entity']))
        
            context_dict_list[idx].append({
                'role': 'assistant',
                'content': recommender_text_list[idx],
                'entity': recommender_resp_entity,
                'rec_items': rec_items_list[idx][0],
                'rec_success': rec_success,
                'item_names_list': item_names_list[idx]
            })
        
            seeker_prompt_list[idx] += f'Recommender: {recommender_text_list[idx]}\n'
        del recommender_text_list
        del rec_items_list
        
        print(f'Conv time: {time.time() - start_time}s')
        start_time = time.time()
        
        # Time for seeker
        print(f'Get seeker response')
        seeker_text_list = annotate_batch_completion(args, seeker_instruct_list, seeker_prompt_list) # TODO: utils.py에서 구현
        
        # post-process for seeker utterance
        for idx, seeker_text in enumerate(seeker_text_list):
            seeker_response = filter_seeker_text(seeker_text, goal_item_list_list[idx], rec_success_list[idx])
            seeker_prompt_list[idx] += f'Seeker: {seeker_response}\n'
        
            # public
            seeker_resp_entity = get_entity(seeker_text, entity_list)
        
            context_dict_list[idx].append({
                'role': 'user',
                'content': seeker_text,
                'entity': seeker_resp_entity,
            })
            
            conv_dict_list[idx]['context'].append(seeker_text)
            conv_dict_list[idx]['entity'] += seeker_resp_entity
            conv_dict_list[idx]['entity'] = list(set(conv_dict_list[idx]['entity']))
        
        print(f'Seeker time: {time.time() - start_time}s')
        start_time = time.time()
        
        # terminate if rec success
        removal_dialog_id_list = []
        for dialog_id in dialog_id_list:
            idx = dialog_id_list.index(dialog_id)
            if rec_success_list[idx]:
                #TODO: write the result
                # score persuativeness
                conv_dict_list[idx]['context'] = context_dict_list[idx]
                dialog_data_list[idx]['simulator_dialog'] = conv_dict_list[idx]

                # save
                with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f: 
                    json.dump(dialog_data_list[idx], f, ensure_ascii=False, indent=2)
                removal_dialog_id_list.append(dialog_id)
        
        #TODO: delete the relevant dialog element from the list
        for delete_dialog_id in removal_dialog_id_list:
            idx = dialog_id_list.index(delete_dialog_id)
            target_lists = [dialog_id_list, dialog_data_list, conv_dict_list, context_dict_list, seeker_instruct_list, rec_success_list, seeker_prompt_list, goal_item_list_list]
            for target_list in target_lists:
                target_list.pop(idx)
                
        assert len(dialog_id_list) == len(dialog_data_list) == len(conv_dict_list) == len(context_dict_list) \
            == len(seeker_instruct_list) == len(rec_success_list) == len(seeker_prompt_list) == len(goal_item_list_list)
        
        if len(dialog_id_list) == 0:
            break
        
        print(f'post-process time: {time.time() - start_time}s')
        start_time = time.time()
    
    if len(dialog_id_list) != 0:
        for dialog_id in dialog_id_list:
            idx = dialog_id_list.index(dialog_id)
            
            #TODO: write the result
            conv_dict_list[idx]['context'] = context_dict_list[idx]
            dialog_data_list[idx]['simulator_dialog'] = conv_dict_list[idx]

            # save
            with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f: 
                json.dump(dialog_data_list[idx], f, ensure_ascii=False, indent=2)
            removal_dialog_id_list.append(dialog_id)

def get_dialog_emb(dialog: dict, recommender: RECOMMENDER):
    conv_str = ""
    for utterance in dialog:
        conv_str += f"{utterance['role']}: {utterance['content']} "
    
    dialog_emb = recommender.crs_model.annotate(recommender.crs_model.args, conv_str)

    return dialog_emb

def get_dialog_list_emb(dialog_list: list[dict], recommender: RECOMMENDER):
    conv_str_list = []
    for dialog in dialog_list:
        conv_str = ""
        for utterance in dialog:
            conv_str += f"{utterance['role']}: {utterance['content']} "
        conv_str_list.append(conv_str)
    
    dialog_embs = recommender.crs_model.annotate(recommender.crs_model.args, conv_str_list)

    return dialog_embs
                
def save_dpo_data(args: argparse.Namespace, dialog_id_list: list[str], instruct_list: list[str], prompt_list: list[str], rec_model_prompt: str, save_dir: str, each_context_dict_list: list[dict], 
                  conv_dict_list: list[dict], item_embeddings: np.ndarray, idx: int, title2emb: dict, each_turn: int, base_dialog_emb, each_context_emb_list):
    
    # base dialog에 대여 score 계산
    target_item_title = conv_dict_list[idx]['rec'][0] # TODO: 여러 개의 target item을 고려할 수 있도록 수정
    target_item_emb = title2emb[target_item_title]
    all_similarities = batch_cosine_similarity(base_dialog_emb, item_embeddings)
    # print(all_similarities)
    mean_similarity = np.mean(all_similarities)
    base_sim = cosine_similarity(base_dialog_emb, target_item_emb) - mean_similarity

    # TODO: maxtrix multiplication 연산으로 바꾸어 speed 높이기.
    # 각 dialog에 대하여 score 계산
    diff_sim_list = []
    for dialog_emb in each_context_emb_list:
        all_similarities = batch_cosine_similarity(dialog_emb, item_embeddings)
        mean_similarity = np.mean(all_similarities)
        sim = cosine_similarity(dialog_emb, target_item_emb) - mean_similarity
        diff_sim = sim - base_sim
        diff_sim_list.append(diff_sim)
    
    max_idx = np.argmax(diff_sim_list)
    min_idx = np.argmin(diff_sim_list)
    
    pos_dialog = each_context_dict_list[max_idx]
    neg_dialog = each_context_dict_list[min_idx]
    pos_score = diff_sim_list[max_idx]
    neg_score = diff_sim_list[min_idx]
    
    pos_seeker_prompt = prompt_list[max_idx]
    neg_seeker_prompt = prompt_list[min_idx]
    pos_seeker_instruct = instruct_list[max_idx]
    neg_seeker_instruct = instruct_list[min_idx]
    
    assert pos_seeker_prompt == neg_seeker_prompt and pos_seeker_instruct == neg_seeker_instruct
    
    total_seeker_prompt = [{'role': 'system', 'content': pos_seeker_instruct}, {'role': 'user', 'content': pos_seeker_prompt + "\n#############"}]
    
    for dialog in [pos_dialog, neg_dialog]:
        for i, context_dict_element in enumerate(dialog):
            for key in list(context_dict_element.keys()):
                if key not in ['role', 'content']:
                    context_dict_element.pop(key)
    
    # save chosen and rejected
    save_data = {
        'chosen': pos_dialog,
        'rejected': neg_dialog,
        'seeker_prompt': total_seeker_prompt,
        'recommender_prompt': rec_model_prompt,
        'chosen_score': pos_score,
        'rejected_score': neg_score
    }
    
    # save
    with open(f'{save_dir}/{dialog_id_list[idx]}_{each_turn}.json', 'w', encoding='utf-8') as f: 
        json.dump(save_data, f, ensure_ascii=False, indent=2)
        
def cosine_similarity(A, B):
    dot_products = A @ B.T

    # 2) Compute row norms for A and B
    A_norms = A.norm(dim=1)       # shape (N,)
    B_norms = B.norm(dim=1)       # shape (M,)

    # 3) Divide elementwise to get the NxM matrix of cosine similarities
    # Reshape norms to broadcast across rows and columns
    cos_sim_matrix = dot_products / (A_norms.unsqueeze(1) * B_norms.unsqueeze(0))

    return cos_sim_matrix

def batch_construct_DPO_data(dialog_id_list: list[str], data_list: list[dict], seeker_instruction_template: str, args: argparse.Namespace, recommender: RECOMMENDER, save_dir: str):
    start_time = time.time()
    print("=====================================")
    print(dialog_id_list)
    start_time = time.time()
    # emb_file_path = f"{args.root_dir}/data/{args.dataset}/title2emb_{args.embedding_model}.json"
    # if os.path.exists(emb_file_path):
    #     with open(emb_file_path, "r") as f:
    #         title2emb = json.load(f)
    name2id = recommender.crs_model.name2id
    id2item_id = recommender.crs_model.id2item_id
    item_emb_list = recommender.crs_model.item_emb_list
    entity2id = recommender.crs_model.entity2id
    
    title2emb = {}
    for name, id in name2id.items():
        if name in entity2id:
            index = id2item_id.index(id)
            emb = item_emb_list[index]
            title2emb[name] = emb
    item_embeddings = np.stack(list(title2emb.values()))
    
    conv_dict_list = [] 
    context_dict_list = []
    rec_success_list = [False] * len(dialog_id_list)
    seeker_prompt_list = []
    
    goal_item_list_list = [] 
    seeker_instruct_list = []
    
    for idx, dialog_id in enumerate(dialog_id_list):
        conv_dict = copy.deepcopy(data_list[idx]) # for model
        context = conv_dict['context']

        goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
        goal_item_str = ', '.join(goal_item_list)
        seeker_instruct = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
        # seeker_prompt = ''' v0
        #     You are role-playing as a Seeker to only generate the Seeker's next response. You are not recommender, so do not generate responses or utterances on behalf of the recommender. Keep in mind that Your task is only to generate the User’s next response. Below is Conversation History
        #     #############
        # '''
        #v3
        seeker_prompt = args.seeker_prompt
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
        seeker_prompt_list.append(seeker_prompt)
        conv_dict_list.append(conv_dict)
        context_dict_list.append(context_dict)
        seeker_instruct_list.append(seeker_instruct)
        goal_item_list_list.append(goal_item_list)


    # 1. 매 턴에서의 chosen과 rejected를 구분하여 저장
    # 2. next convdict를 생성하는 것
    print(f'initialization_time: {(time.time() - start_time):.2f}s')
    start_time = time.time()
    for each_turn in range(0, args.turn_num):
        print(f"Turn {each_turn}")
        start_time = time.time()

        # for current turn DPO data construction
        current_turn_context_dict_list = []
        current_turn_seeker_prompt_list = []
        current_turn_seeker_instruct_list = []
        for i in range(len(context_dict_list)):
            for j in range(args.beam_num):
                current_turn_context_dict_list.append(copy.deepcopy(context_dict_list[i]))
                current_turn_seeker_prompt_list.append(copy.deepcopy(seeker_prompt_list[i]))
                current_turn_seeker_instruct_list.append(copy.deepcopy(seeker_instruct_list[i]))

        # Recommendation
        rec_items_list, rec_labels_list, item_names_list = recommender.get_batch_rec(conv_dict_list)
        print(f'recommendation time: {(time.time() - start_time):.2f}s')
        start_time = time.time()
        
        # check if rec success
        for idx, rec_labels in enumerate(rec_labels_list):
            for rec_label in rec_labels:
                if rec_label in rec_items_list[idx][0]:
                    rec_success_list[idx] = True
                    break
        del rec_labels_list
            
        # Conversation
        print(type(conv_dict_list))
        print(type(item_names_list))
        _, recommender_text_list, rec_prompt_list = recommender.get_sample_batch_conv(conv_dict_list, item_names_list) # length = batch_size * sample_size (beam_num)
        print(f'conversation time: {(time.time() - start_time):.2f}s')
        start_time = time.time()
        
        # post-process
        print(len(dialog_id_list), len(recommender_text_list))
        resized_recommender_text_list = convert_1d_to_2d(recommender_text_list, len(dialog_id_list), len(recommender_text_list) // len(dialog_id_list))
        random_index = random.choice(list(range(len(recommender_text_list) // len(dialog_id_list))))
        save_recommender_text_list = [resized_recommender_text_list[idx][random_index] for idx in range(len(resized_recommender_text_list))]
        
        for idx, rec_success in enumerate(rec_success_list):
            conv_dict_list[idx]['context'].append(save_recommender_text_list[idx])
            context_dict_list[idx].append({
                'role': 'assistant',
                'content': save_recommender_text_list[idx],
                'rec_items': rec_items_list[idx][0],
                'rec_success': rec_success
            })
            seeker_prompt_list[idx] += f'Recommender: {save_recommender_text_list[idx]}\n'
        del rec_items_list    
        del save_recommender_text_list
        print(f'post-proces time: {(time.time() - start_time):.2f}s')
        start_time = time.time()
        
        # Seeker Response
        seeker_text_list = annotate_batch_completion(args, seeker_instruct_list, seeker_prompt_list)
        print(f'seeker_response time: {(time.time() - start_time):.2f}s')
        start_time = time.time()
                
        # post-process
        for idx, seeker_text in enumerate(seeker_text_list):
            seeker_response = filter_seeker_text(seeker_text, goal_item_list_list[idx], rec_success_list[idx])
        
            context_dict_list[idx].append({
                'role': 'user',
                'content': seeker_text,
            })
            conv_dict_list[idx]['context'].append(seeker_text)
            seeker_prompt_list[idx] += f'Seeker: {seeker_response}\n'
        
        print(f'each_turn_time: {(time.time() - start_time):.2f}s')
        start_time = time.time()

        # DPO data construction
        assert len(current_turn_context_dict_list) == len(recommender_text_list)
        for idx in range(len(current_turn_context_dict_list)):
            current_turn_context_dict_list[idx].append({
                'role': 'assistant',
                'content': recommender_text_list[idx],
            })
            current_turn_seeker_prompt_list[idx] += f'Recommender: {recommender_text_list[idx]}\n'
            
        seeker_text_list = annotate_batch_completion(args, current_turn_seeker_instruct_list, current_turn_seeker_prompt_list)
        print(f'seeker_response for DPO: {(time.time() - start_time):.2f}s')
        start_time = time.time()
        
        for idx in range(len(current_turn_context_dict_list)):
            current_turn_context_dict_list[idx].append({
                'role': 'user',
                'content': seeker_text_list[idx],
            })
        
        processes = []
        rec_model_prompt = recommender.crs_model.chat_recommender_instruction
        base_context_dict_list = [context_dict[:-2] for context_dict in context_dict_list] # TODO: save_dpo_data 이전에 gpu 사용하는 task 모두 수행시켜두기.
        base_context_dict_list.extend(current_turn_context_dict_list)
        
        dialog_embs = get_dialog_list_emb(base_context_dict_list, recommender)
        
        dialog_embs = torch.tensor(dialog_embs, dtype=torch.float).to(args.device)
        item_embeddings = torch.tensor(item_embeddings, dtype=torch.float).to(args.device)
        
        cosine_similarity_matrix = cosine_similarity(dialog_embs, item_embeddings) # size: (batch_size * (beam_num+1), item_num)
        if args.reward_func_topK == -1:
            mean_similarties = torch.mean(cosine_similarity_matrix, dim=1) # size: (batch_size * (beam_num+1))
        else:
            topK_cosine_similarity_matrix = torch.topk(cosine_similarity_matrix, args.topK, dim=1)
            mean_similarties = torch.mean(topK_cosine_similarity_matrix.values, dim=1) # size: (batch_size * (beam_num+1))
        base_mean_similarities = mean_similarties[:len(context_dict_list)] # size: (batch_size)
        current_turn_mean_similarities = mean_similarties[len(context_dict_list):] # size: (batch_size * beam_num)
        
        target_items = []        
        for conv_dict in conv_dict_list:
            target_item_title_list = conv_dict['rec']
            for target_item_title in target_item_title_list:
                target_items.append(title2emb[target_item_title])
        
        target_items = torch.tensor(target_items).to(args.device)
        target_cosine_similarities = cosine_similarity(dialog_embs, target_items) # size: (batch_size * (beam_num+1), item_num)
        base_target_cosine_similarities = target_cosine_similarities[:len(context_dict_list)] # size: (batch_size , item_num)
        each_current_turn_target_cosine_similarities = target_cosine_similarities[len(context_dict_list):] # size: (batch_size * beam_num , item_num)
        
        prev = 0
        for idx in range(len(context_dict_list)):
            each_context_dict_list = current_turn_context_dict_list[args.beam_num * idx: args.beam_num * (idx + 1)]
            each_seeker_prompt_list = current_turn_seeker_prompt_list[args.beam_num * idx: args.beam_num * (idx + 1)]
            each_seeker_instruct_list = current_turn_seeker_instruct_list[args.beam_num * idx: args.beam_num * (idx + 1)]
            
            base_dialog_mean_similarity = base_mean_similarities[idx]
            each_current_turn_mean_similarity = current_turn_mean_similarities[args.beam_num * idx: args.beam_num * (idx + 1)]

            base_target_cosine_similarity = base_target_cosine_similarities[idx, prev: prev + len(conv_dict_list[idx]['rec'])]
            if base_target_cosine_similarity.size()[0] == 1: 
                pass
            else:
                base_target_cosine_similarity = torch.mean(base_target_cosine_similarity, dim=0)
                
            each_current_turn_target_cosine_similarity = each_current_turn_target_cosine_similarities[args.beam_num * idx: args.beam_num * (idx + 1), prev: prev + len(conv_dict_list[idx]['rec'])]
            if each_current_turn_target_cosine_similarity.dim() == 1:
                pass
            else:
                each_current_turn_target_cosine_similarity = torch.mean(each_current_turn_target_cosine_similarity, dim=1)
            
            prev = prev + len(conv_dict_list[idx]['rec'])
            
            base_score = base_target_cosine_similarity - base_dialog_mean_similarity
            each_current_turn_score = each_current_turn_target_cosine_similarity - each_current_turn_mean_similarity
            diff_score = each_current_turn_score - base_score
            
            max_idx = torch.argmax(diff_score)
            min_idx = torch.argmin(diff_score)
            
            if max_idx == min_idx:
                print(f'diff_score: {diff_score}')
                print(f'each_current_turn_score: {each_current_turn_score}')
                print(f'base_score: {base_score}')
                print("================")
            
            pos_dialog = each_context_dict_list[max_idx]
            neg_dialog = each_context_dict_list[min_idx]
            pos_score = diff_score[max_idx]
            neg_score = diff_score[min_idx]
            
            pos_seeker_prompt = each_seeker_prompt_list[max_idx]
            neg_seeker_prompt = each_seeker_prompt_list[min_idx]
            pos_seeker_instruct = each_seeker_instruct_list[max_idx]
            neg_seeker_instruct = each_seeker_instruct_list[min_idx]
            
            assert pos_seeker_instruct == neg_seeker_instruct
            
            total_seeker_prompt = [{'role': 'system', 'content': pos_seeker_instruct}, {'role': 'user', 'content': pos_seeker_prompt + "\n#############"}]
            
            for dialog in [pos_dialog, neg_dialog]:
                for i, context_dict_element in enumerate(dialog):
                    for key in list(context_dict_element.keys()):
                        if key not in ['role', 'content']:
                            context_dict_element.pop(key)
                            
            # save chosen and rejected
            save_data = {
                'chosen': pos_dialog,
                'rejected': neg_dialog,
                'seeker_prompt': total_seeker_prompt,
                'recommender_prompt': rec_prompt_list[idx],
                'chosen_score': pos_score.item(),
                'rejected_score': neg_score.item()
            }
            
            # save
            with open(f'{save_dir}/{dialog_id_list[idx]}_{each_turn}.json', 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        # for idx in range(len(context_dict_list)):
        #     base_dialog_emb = base_dialog_emb_list[idx]
        #     each_context_emb_list = current_turn_dialog_emb_list[args.beam_num * idx: args.beam_num * (idx + 1)]
        #     each_context_dict_list = current_turn_context_dict_list[args.beam_num * idx: args.beam_num * (idx + 1)]
        #     each_seeker_prompt_list = current_turn_seeker_prompt_list[args.beam_num * idx: args.beam_num * (idx + 1)]
        #     each_seeker_instruct_list = current_turn_seeker_instruct_list[args.beam_num * idx: args.beam_num * (idx + 1)]
            
        #     p = Process(target=save_dpo_data,
        #                 args=(args, dialog_id_list, each_seeker_instruct_list, each_seeker_prompt_list, rec_model_prompt, save_dir, each_context_dict_list, conv_dict_list, item_embeddings, idx, title2emb, each_turn, base_dialog_emb, each_context_emb_list))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        print(f'dpo_data_construction&save: {(time.time() - start_time):.2f}s')
            
        # Turn Stop
        if rec_success == True or each_turn == args.turn_num - 1:
            removal_dialog_id_list = []
            for dialog_id in dialog_id_list:
                idx = dialog_id_list.index(dialog_id)
                if rec_success_list[idx]:
                    removal_dialog_id_list.append(dialog_id)
            
            for delete_dialog_id in removal_dialog_id_list:
                idx = dialog_id_list.index(delete_dialog_id)
                target_lists = [dialog_id_list, data_list, conv_dict_list, context_dict_list, seeker_instruct_list, rec_success_list, seeker_prompt_list, goal_item_list_list]
                for target_list in target_lists:
                    target_list.pop(idx)
                    
            assert len(dialog_id_list) == len(data_list) == len(conv_dict_list) == len(context_dict_list) \
                == len(seeker_instruct_list) == len(rec_success_list) == len(seeker_prompt_list) == len(goal_item_list_list)
        
        if len(dialog_id_list) == 0:
            break