# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig
import torch, json, os, argparse
from src.model.utils import get_model_path

import torch
from transformers import TrainingArguments
from trl import DPOTrainer

def load_dialog_data_list(dir: str) -> list:
    
    dialog_data_list = []
    for file in os.listdir(dir):
        try:
            file_path = os.path.join(dir, file)
            if os.path.isdir(file_path):
                continue
            dialog_data = json.load(open(file_path, 'r'))
            dialog_data_list.append(dialog_data)
        except json.decoder.JSONDecodeError as e:
            print(e)
            print(file)
        
    return dialog_data_list

def rec_extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return [messages[index]]
    else:
        return rec_extract_assistant_messages(messages, index-1)

def create_triplets_v0(example, tokenizer):
    """Create the triplets (prompt, chosen, rejected)"""
    # Extract the N-1 turns to form the prompt
    # Prepend a system message if the first message is not a system message
    
    chosen_prompt_messages = example["chosen"][:-2]
    rejected_prompt_messages = example["rejected"][:-2]
    
    assert chosen_prompt_messages == rejected_prompt_messages, "Prompt messages are different for chosen and rejected responses"
    
    prompt_messages = chosen_prompt_messages
    if prompt_messages[-1]['role'] == 'assistant':
        print(f'Prompt messages: {prompt_messages}')
    sys_instrunction = {
        'role': 'system',
        'content': example['recommender_prompt'],
    }
    prompt_messages.insert(0, sys_instrunction)
    
    # Now we extract the final assistant turn to define chosen/rejected responses
    chosen_messages = rec_extract_assistant_messages(example["chosen"])
    rejected_messages = rec_extract_assistant_messages(example["rejected"])
    
    # apply template to the messages and return the triplets
    # return {
    #     "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True,),
    #     "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=True,),
    #     "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=True,)
    # }
    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }
    
def create_triplets_v1(example, tokenizer):
    """Create the triplets (prompt, chosen, rejected)"""
    # Extract the N-1 turns to form the prompt
    # Prepend a system message if the first message is not a system message
    
    chosen_prompt_messages = example["chosen"][:-2]
    rejected_prompt_messages = example["rejected"][:-2]
    
    assert chosen_prompt_messages == rejected_prompt_messages, "Prompt messages are different for chosen and rejected responses"
    
    # prompt_messages = chosen_prompt_messages
    # if prompt_messages[-1]['role'] == 'assistant':
    #     print(f'Prompt messages: {prompt_messages}')
    prompt_messages = []
    sys_instrunction = {
        'role': 'system',
        'content': example['recommender_prompt'],
    }
    prompt_messages.append(sys_instrunction)
    
    # Now we extract the final assistant turn to define chosen/rejected responses
    chosen_messages = rec_extract_assistant_messages(example["chosen"])
    rejected_messages = rec_extract_assistant_messages(example["rejected"])
    
    # apply template to the messages and return the triplets
    # return {
    #     "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True,),
    #     "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=True,),
    #     "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=True,)
    # }
    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'], default='opendialkg_eval')
    parser.add_argument('--embedding_model', type=str, default = "text-embedding-3-small")
    parser.add_argument('--rec_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--user_model', type=str, default = "gpt-4o-mini")
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'openmodel'])
    parser.add_argument('--use_unsloth', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_num', type=int, default=8)
    parser.add_argument('--topK', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--history', type=str, default='full')
    parser.add_argument('--eval_strategy', type=str, default='non_repeated', choices=['repeated', 'non_repeated'])
    parser.add_argument('--eval_data_size', type=str, default='full', choices=['sample', 'full']) # "sample" means "sampling 100 dialogues"
    parser.add_argument('--reward_func_topK', type=int, default=10, choices =[-1, 1, 5, 10, 20, 30, 50, 100])
    parser.add_argument('--prompt_ver', type=str, default = "v0")
    parser.add_argument('--max_grad_norm', type=int, default = 1)
    parser.add_argument('--turn_num', type=int, default = 5)
    
    args = parser.parse_args()

    gradient_accumulation_steps = 64
    epochs = 3
    lr = 5e-5
    output_dir = f"/data1/shchoi/pref_tuned/{args.prompt_ver}/{args.rec_model}_{args.embedding_model}_{args.dataset}_rank_{args.rank}_grad_acc_{gradient_accumulation_steps}_lr_{lr}_epochs_{epochs}"

    if 'unsloth' in args.rec_model: args.use_unsloth = True
    else: args.use_unsloth = False
    if 'unsloth' in args.user_model: args.user_use_unsloth = True
    else: args.user_use_unsloth = False
    args.root_dir = os.getcwd()
    args.cache_dir = "/data1/shchoi/LLM_ckp/hub"
    print(f'root_dir: {args.root_dir}')
    # save_dir = f'/home/work/shchoi/iEvaLM-CRS/save_5/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_train_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}' 
    # print(f'save_dir: {save_dir}')
    # dialog_data_list = load_dialog_data_list(save_dir)
        
    train_data_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_train_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}_reward_func_topK{args.reward_func_topK}' 
    valid_data_dir = f'/home/shchoi/iEvaLM-CRS/save_{args.turn_num}/{args.prompt_ver}/user_{args.user_model}/emb_{args.embedding_model}/{args.crs_model}_{args.rec_model}_top{args.topK}_{args.history}_history/{args.dataset}/{args.eval_data_size}_{args.eval_strategy}/dpo_valid_data_temp{args.temperature}_sample_num{args.beam_num}_top{args.topK}_reward_func_topK{args.reward_func_topK}' 
    
    print(f'train_data_dir: {train_data_dir}')
    print(f'valid_data_dir: {valid_data_dir}')
    train_dialog_data_list = load_dialog_data_list(train_data_dir)
    valid_dialog_data_list = load_dialog_data_list(valid_data_dir)
    
    if args.use_unsloth:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        from unsloth import is_bfloat16_supported
        print(f'Using unsloth for training')
        PatchDPOTrainer()
        args.kg_dataset = args.dataset.split('_')[1]
        model_id = get_model_path()[args.rec_model]
        model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_id, load_in_4bit = False, cache_dir = args.cache_dir, device_map="auto", max_seq_length = args.max_seq_length)
        if "Llama" in args.rec_model:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset = Dataset.from_list(dialog_data_list)
        train_dataset = train_dataset.map(create_triplets, remove_columns=train_dataset.features, fn_kwargs={"tokenizer": tokenizer})

        # print size of train_dataset
        print(f"Number of dialogues: {len(train_dataset)}")

        model = FastLanguageModel.get_peft_model(
            model,
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = args.seed,
            max_seq_length = args.max_seq_length,
            r=args.rank,
            use_rslora=True,
            lora_alpha=4*args.rank,
            lora_dropout=0.0,
            target_modules=[
                'q_proj',
                'v_proj',
                'k_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            bias="none",
        )

        training_args = TrainingArguments(
            output_dir=output_dir, 
            logging_steps=10,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = gradient_accumulation_steps,
            num_train_epochs = epochs,
            learning_rate=lr,
            warmup_ratio = 0.1,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = args.seed,
        ) 
        dpo_trainer = DPOTrainer(
            model = model,
            args = training_args,
            processing_class=tokenizer,
            train_dataset = train_dataset,
            tokenizer = tokenizer,
            max_length = args.max_seq_length,
        )
        dpo_trainer.train()
    else:
        args.kg_dataset = args.dataset.split('_')[1]
        model_id = get_model_path(args)[args.rec_model]
        print(args.gpu_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            ).to(f'cuda:{args.gpu_id}')
        # import time
        # time.sleep(10)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir)
        if "Llama" in args.rec_model:
            tokenizer.pad_token = tokenizer.eos_token

        if args.prompt_ver == 'v0' or args.prompt_ver == 'v2':
            train_dataset = Dataset.from_list(train_dialog_data_list)
            train_dataset = train_dataset.map(create_triplets_v0, remove_columns=train_dataset.features, fn_kwargs={"tokenizer": tokenizer})
            
            valid_dataset = Dataset.from_list(valid_dialog_data_list)
            valid_dataset = valid_dataset.map(create_triplets_v0, remove_columns=valid_dataset.features, fn_kwargs={"tokenizer": tokenizer})
        elif args.prompt_ver == 'v1':
            train_dataset = Dataset.from_list(train_dialog_data_list)
            train_dataset = train_dataset.map(create_triplets_v1, remove_columns=train_dataset.features, fn_kwargs={"tokenizer": tokenizer})
            
            valid_dataset = Dataset.from_list(valid_dialog_data_list)
            valid_dataset = valid_dataset.map(create_triplets_v1, remove_columns=valid_dataset.features, fn_kwargs={"tokenizer": tokenizer})
            
        # print size of train_dataset
        print(f"Number of train dialogues: {len(train_dataset)}")
        print(f"Number of valid dialogues: {len(valid_dataset)}")

        # print size of train_dataset
        peft_config = LoraConfig(
            r=args.rank,
            use_rslora= True,
            lora_alpha=args.rank*4,
            lora_dropout=0.05,
            target_modules=[
                'q_proj',
                'v_proj',
                'k_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )


        training_args = DPOConfig(
            output_dir=output_dir, 
            logging_steps=10,
            per_device_train_batch_size=1,  # Start with small batch size
            gradient_accumulation_steps=gradient_accumulation_steps,  # Increase effective batch size 
            num_train_epochs = epochs,
            # max_steps=1000,
            # save_steps=100,
            # gradient_checkpointing=True,
            learning_rate=lr,
            bf16=True,
            warmup_ratio = 0.1,
            # remove_unused_columns=False,
            # run_name="dpo_llama2",
            eval_strategy="steps",
            eval_steps=30,
            save_strategy="steps",
            seed=42,
            load_best_model_at_end=True,
            save_steps=30,
            save_total_limit=5,
            eval_on_start=True,
            greater_is_better= True,
            metric_for_best_model="eval_rewards/accuracies"
        )
        trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, eval_dataset=valid_dataset, peft_config=peft_config, callbacks=[EarlyStoppingCallback(early_stopping_patience=360)])
        trainer.train()
        trainer.save_model(f'{output_dir}/best_model')
        