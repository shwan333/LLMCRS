import json
import argparse
import re
import os
from tqdm import tqdm

import sys
sys.path.append("..")

from src.model.metric import RecMetric

# datasets = ['redial_eval', 'opendialkg_eval']
# models = ['kbrd', 'barcor', 'unicrs', 'chatgpt']


# compute rec recall
def rec_eval(turn_num, mode):

    with open(f"../data/{args.dataset.split('_')[0]}/entity2id.json", 'r', encoding="utf-8") as f:
        entity2id = json.load(f)
    
    metric = RecMetric([1, 5, 10, 25, 50])
    # persuatiness = 0
    save_path = f'{args.root_dir}/save_{args.turn_num}/{args.mode}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}' 
    # save_path = f"../save_{turn_num}/{mode}/{model}/{dataset}" # data loaded path
    result_path = f"../save_{args.turn_num}/result/{args.mode}/{args.crs_model}_{args.rec_model}"
    os.makedirs(result_path, exist_ok=True)
    print(os.listdir(save_path))
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        path_list = os.listdir(save_path)
        print(f"turn_num: {turn_num}, mode: {mode} model: {args.crs_model} dataset: {args.dataset}", len(path_list))
        
        for path in tqdm(path_list):
            if os.path.isdir(f"{save_path}/{path}"):
                continue
            with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                data = json.load(f)
                # if mode == 'chat':
                #     persuasiveness_score = data['persuasiveness_score']
                #     persuatiness += float(persuasiveness_score)
                PE_dialog = data['simulator_dialog']
                rec_label = data['rec']
                rec_label = [entity2id[rec] for rec in rec_label if rec in entity2id]
                contexts = PE_dialog['context']
                for context in contexts[::-1]:
                    if 'rec_items' in context:
                        rec_items = context['rec_items']
                        metric.evaluate(rec_items, rec_label)
                        break
            
        report = metric.report()
        
        print('r1:', f"{report['recall@1']:.3f}", 'r5:', f"{report['recall@5']:.3f}", 'r10:', f"{report['recall@10']:.3f}", 'r25:', f"{report['recall@25']:.3f}", 'r50:', f"{report['recall@50']:.3f}", 'count:', report['count'])
        # if mode == 'chat':
        #     persuativeness_score = persuatiness / len(path_list)
        #     print(f"{persuativeness_score:.3f}")
        #     report['persuativeness'] = persuativeness_score
        
        with open(f"{result_path}/{args.dataset}_{args.eval_data_size}_{args.eval_strategy}.json", 'w', encoding="utf-8") as w:
            w.write(json.dumps(report))
                    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn_num', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'openmodel'])
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--rec_model', type=str, default = "gpt-4o-mini", choices=["gpt-4o-mini", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"])
    parser.add_argument('--topK', default = 10, type=int)
    parser.add_argument('--history', default = 'full', type=str)
    
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    
    rec_eval(args.turn_num, args.mode)