import glob
from tqdm import tqdm
from datasets import load_dataset
import json
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
import gzip
import os
from sentence_transformers import InputExample
from datasets import Dataset

import torch, argparse

def get_data(path, data_type, title2text):
    cnt = 0
    data_dict = {
        'anchor': [],
        'positive': []
    }
    results_path = glob.glob(path + '/*.json')
    for sample_result_path in tqdm(results_path):
        turn_num = int(sample_result_path.split('/')[-1].split('.')[0].split('_')[-1])
        sample_result = json.load(open(sample_result_path))
        target_item_title = sample_result['rec'][0]
        dialog_dict = sample_result['simulator_dialog']['context']
        # assert (len(dialog_dict) - turn_num) % 2 == 0.0, f"dialog_dict: {len(dialog_dict)}, turn_num: {turn_num}"
        if (len(dialog_dict) - turn_num) % 2 != 0.0: continue
        iter_num = (len(dialog_dict) - turn_num) // 2
        gt_item_list = sample_result['rec']
        for gt_item in gt_item_list:
            passage = title2text[gt_item]
            if data_type == 'total':
                for idx in range(1, iter_num+1):
                    if idx == 0:
                        query_dialog = dialog_dict[:]
                    else:
                        query_dialog = dialog_dict[: -2*idx]
                    query = ""
                    for context in query_dialog:
                        query += f"{context['role']}: {context['content']} "
                    data_dict['anchor'].append(query)
                    data_dict['positive'].append(passage)
            else:
                query_dialog = dialog_dict[: -2*iter_num]
                query = ""
                for context in query_dialog:
                    query += f"{context['role']}: {context['content']} "
                data_dict['anchor'].append(query)
                data_dict['positive'].append(passage)
            
    data = Dataset.from_dict(data_dict)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'], default = "opendialkg_eval", help="데이터셋 종류")
    parser.add_argument('--embedding_model', type=str, required=True, help="임베딩 모델 이름")
    parser.add_argument('--rec_model', type=str, default="Llama-3.2-1B-Instruct", help="추천 모델 이름")
    parser.add_argument('--adapter_model', type=str, default=None)
    parser.add_argument('--topK', type=int, default=10, help="상위 K 개의 결과를 사용할 때의 K 값")
    parser.add_argument('--history', type=str, default="full", help="히스토리 설정 (예: full, partial 등)")
    parser.add_argument('--split', type=str, default="train", help="데이터셋 split (train/valid/test)")
    parser.add_argument('--num_epochs', type=int, default=7, help="학습 에포크 수")
    parser.add_argument('--lr', type=float, default=5e-5, help="학습률")
    parser.add_argument('--data_type', type=str, default="total", help="데이터 타입")
    parser.add_argument('--user_model', type=str, default="Llama-3.1-8B-Instruct", help="유저 모델 이름")
    parser.add_argument('--gpu_id', type=int, default=0,)
    parser.add_argument('--prompt_ver', type=str, required=True, help="prompt 버전")
    parser.add_argument('--turn_num', type=int, required=True, help="저장 디렉토리")
    
    args = parser.parse_args()
    args.root_dir = os.path.dirname(os.getcwd())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Train the model
    if args.adapter_model == None:
        train_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/openmodel_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/full_non_repeated/train'
        eval_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/openmodel_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/full_non_repeated/valid' 
    else:
        train_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/openmodel_{args.rec_model}_adapter_{args.adapter_model}_top{args.topK}_{args.history}_history/{args.dataset}/full_non_repeated/train'
        eval_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/openmodel_{args.rec_model}_adapter_{args.adapter_model}_top{args.topK}_{args.history}_history/{args.dataset}/full_non_repeated/test' 
    
    output_dir = f'/data1/shchoi/output/{args.data_type}/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/openmodel_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/full_non_repeated/{args.split}'
    print(f'train_dir: {train_dir}')
    print(f'eval_dir: {eval_dir}')
    
    if args.dataset == 'redial_eval': attr_list = ['genre', 'star', 'director']
    elif args.dataset == 'opendialkg_eval': attr_list = ['genre', 'actor', 'director', 'writer']
    kg_dataset_path = os.path.join(args.root_dir, "data", args.dataset)
    
    with open(f"{kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
        id2info = json.load(f)
            
    title2text = {}
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
        title2text[info_dict["name"]] = item_text
        
    training_data = get_data(train_dir, args.data_type, title2text)
    eval_data = get_data(eval_dir, args.data_type, title2text)
    args.root_dir = os.path.dirname(os.getcwd())
    model_path = os.path.join(args.root_dir, 'model_path.json')
    with open(model_path, 'r') as f:
        model_path = json.load(f)
    embedding_model_path = model_path["Embedding_model"]
    
    # print(gt_dir)

    from torch.utils.data import DataLoader
    import wandb
    from transformers import EarlyStoppingCallback
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers
    model = SentenceTransformer(embedding_model_path[args.embedding_model], cache_folder = "/data1/shchoi/LLM_ckp/hub", device="cuda:0", trust_remote_code=True)
    args.embedding_model = args.embedding_model + f"-true-total-tuned-{args.prompt_ver}"
    # Initialize loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    early_stopper = EarlyStoppingCallback(
        early_stopping_patience=5, # you can change this value if needed
    )
    train_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        seed = 42,
        # Optional training parameters:
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=8, 
        warmup_ratio=0.1,
        fp16=True,  # Set to False if GPU can't handle FP16
        bf16=False,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        load_best_model_at_end=True,
        save_steps=5,
        save_total_limit=5,
        logging_steps=5,
        eval_on_start=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=training_data,
        eval_dataset=eval_data,
        loss=train_loss,
        # evaluator=evaluator,
        callbacks=[early_stopper]
    )

    trainer.train()
    trainer.save_model(f'{output_dir}/best_model')