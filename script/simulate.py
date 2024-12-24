import argparse
import copy
import json
import sys
sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER
from utils import annotate_completion, annotate_batch_completion, process_for_baselines, filter_seeker_text

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
        seeker_text = annotate_completion(args, seeker_instruct, seeker_prompt).strip()
        seeker_response = filter_seeker_text(seeker_text, goal_item_list, rec_success)
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
    
    for idx, dialog_id in enumerate(dialog_id_list):
        conv_dict = copy.deepcopy(dialog_data_list[idx]) # for model
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
        seeker_prompt_list.append(seeker_prompt)
        conv_dict_list.append(conv_dict)
        context_dict_list.append(context_dict)
        seeker_instruct_list.append(seeker_instruct)
        goal_item_list_list.append(goal_item_list)
        
    recommendation_template = "I would recommend the following items: {}:"

    for i in range(0, args.turn_num):
        print(f"Turn {i+1}")
        # Time for recommender 
        # recommendation
        print(f'Get recommendation')
        rec_items_list, rec_labels_list = recommender.get_batch_rec(conv_dict_list) # TODO: OPEN_MODEL.py에서 구현
        
        # check if rec success
        for idx, rec_labels in enumerate(rec_labels_list):
            for rec_label in rec_labels:
                if rec_label in rec_items_list[idx][0]:
                    rec_success_list[idx] = True
                    break
        del rec_labels_list
                
        # conversation
        print(f'Get conversation')
        _, recommender_text_list = recommender.get_batch_conv(conv_dict_list) # TODO: OPEN_MODEL.py에서 구현
        
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
                'rec_success': rec_success
            })
        
            seeker_prompt_list[idx] += f'Recommender: {recommender_text_list[idx]}\n'
        del recommender_text_list
        del rec_items_list
        
        # Time for seeker
        print(f'Get seeker response')
        seeker_text_list = annotate_batch_completion(args, seeker_instruct_list, seeker_prompt_list) # TODO: utils.py에서 구현
        
        # post-process for seeker utterance
        for idx, seeker_text in enumerate(seeker_text_list):
            seeker_response = filter_seeker_text(seeker_text, goal_item_list_list[idx], rec_success_list[idx])
            seeker_prompt_list[idx] += f' {seeker_response}\n'
        
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
    
    if len(dialog_id_list) != 0:
        for dialog_id in dialog_id_list:
            idx = dialog_id_list.index(dialog_id)
            if rec_success_list[idx]:
                #TODO: write the result
                conv_dict_list[idx]['context'] = context_dict_list[idx]
                dialog_data_list[idx]['simulator_dialog'] = conv_dict_list[idx]

                # save
                with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f: 
                    json.dump(dialog_data_list[idx], f, ensure_ascii=False, indent=2)
                removal_dialog_id_list.append(dialog_id)
    
    # # score persuativeness
    # conv_dict['context'] = context_dict
    # data['simulator_dialog'] = conv_dict

    # # save
    # with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f: 
    #     json.dump(data, f, ensure_ascii=False, indent=2)