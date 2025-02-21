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
def rec_eval(turn_num, args):

    with open(f"../data/{args.dataset.split('_')[0]}/entity2id.json", 'r', encoding="utf-8") as f:
        entity2id = json.load(f)
    
    metric = RecMetric([1, 5, 10, 25, 50])
    
    if args.adapter is not None:
        save_path = f'{args.root_dir}/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_adapter_{args.adapter}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/{args.split}' 
        result_path = f"../save_{args.turn_num}/{args.prompt_ver}/result/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_adapter_{args.adapter}/{args.split}"
    else:
        save_path = f'{args.root_dir}/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/{args.split}' 
        result_path = f"../save_{args.turn_num}/{args.prompt_ver}/result/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}/{args.split}"
    
    if args.rewrite:
        save_path = f'{save_path}_rewrite'
        result_path = f"{result_path}_rewrite"
    
    print(save_path)
    os.makedirs(result_path, exist_ok=True)
    print(os.listdir(save_path))
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        path_list = os.listdir(save_path)
        print(f"turn_num: {turn_num}, model: {args.crs_model} dataset: {args.dataset}", len(path_list))
        
        for path in tqdm(path_list):
            if os.path.isdir(f"{save_path}/{path}"):
                continue
            with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                data = json.load(f)
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
        
        with open(f"{result_path}/{args.dataset}_{args.eval_data_size}_{args.eval_strategy}.json", 'w', encoding="utf-8") as w:
            w.write(json.dumps(report))
                    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'openmodel'])
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'], default='opendialkg_eval')
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--rec_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--user_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--embedding_model', type=str, default = "text-embedding-3-small")
    parser.add_argument('--topK', default = 10, type=int)
    parser.add_argument('--history', default = 'full', type=str)
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--split', type=str)
    parser.add_argument('--prompt_ver', type=str)
    
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    
    rec_eval(args.turn_num, args)