# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch, json, os, argparse
from src.model.utils import get_model_path

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported

import torch
from transformers import TrainingArguments
from trl import DPOTrainer

def rec_extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return [messages[index]]
    else:
        return rec_extract_assistant_messages(messages, index-1)

def create_triplets(example, tokenizer):
    """Create the triplets (prompt, chosen, rejected)"""
    # Extract the N-1 turns to form the prompt
    # Prepend a system message if the first message is not a system message
    
    chosen_prompt_messages = example["chosen"][:-2]
    rejected_prompt_messages = example["rejected"][:-2]
    
    assert chosen_prompt_messages == rejected_prompt_messages, "Prompt messages are different for chosen and rejected responses"
    
    prompt_messages = chosen_prompt_messages
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--use_unsloth', action='store_true')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--sample_num', type=int)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--rank', type=int, default=4)
    args = parser.parse_args()
    if args.use_unsloth:
        print(f'Using unsloth for training')
        PatchDPOTrainer()
        args.kg_dataset = args.dataset.split('_')[1]
        model_id = get_model_path()[args.rec_model]
        model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_id, cache_dir = "/home/work/shchoi/.cache/huggingface/hub", device_map="auto", max_seq_length = args.max_seq_length)
        if "Llama" in args.rec_model:
            tokenizer.pad_token = tokenizer.eos_token
        save_dir = f'/home/work/shchoi/iEvaLM-CRS/save_5/chat/openmodel_{args.rec_model}/opendialkg_eval/full_non_repeated/dpo_data_temp{args.temperature}_sample_num{args.sample_num}_top{args.top_k}' 
        dialog_data_list = []
        for file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file)
            if os.path.isdir(file_path):
                continue
            dialog_data = json.load(open(file_path, 'r'))
            dialog_data_list.append(dialog_data)
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
        gradient_accumulation_steps = 32
        epochs = 3
        lr = 5e-5
        training_args = TrainingArguments(
            output_dir=f"full_{args.rec_model}_rank_{args.rank}_grad_acc_{gradient_accumulation_steps}_lr_{lr}_epochs_{epochs}", 
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
        model_id = get_model_path()[args.rec_model]
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            cache_dir="/home/work/shchoi/.cache/huggingface/hub",
            torch_dtype=torch.bfloat16,
            device_map={"": 1},
            ).to('cuda:0')
        # import time
        # time.sleep(10)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/work/shchoi/.cache/huggingface/hub")
        if "Llama" in args.rec_model:
            tokenizer.pad_token = tokenizer.eos_token
        save_dir = f'/home/work/shchoi/iEvaLM-CRS/save_5/chat/openmodel_{args.rec_model}/opendialkg_eval/full_non_repeated/dpo_data' 
        dialog_data_list = []
        for file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file)
            if os.path.isdir(file_path):
                continue
            dialog_data = json.load(open(file_path, 'r'))
            dialog_data_list.append(dialog_data)
        train_dataset = Dataset.from_list(dialog_data_list)

        # print size of train_dataset
        print(f"Number of dialogues: {len(train_dataset)}")

        peft_config = LoraConfig(
            r=args.rank,
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
        gradient_accumulation_steps = 32
        epochs = 3
        lr = 5e-5

        training_args = DPOConfig(
            output_dir=f"full_{args.rec_model}_grad_acc_{gradient_accumulation_steps}_lr_{lr}_epochs_{epochs}", 
            logging_steps=10,
            per_device_train_batch_size=1,  # Start with small batch size
            gradient_accumulation_steps=gradient_accumulation_steps,  # Increase effective batch size 
            num_train_epochs = epochs,
            # max_steps=1000,
            # save_steps=100,
            # gradient_checkpointing=True,
            learning_rate=lr,
            bf16=True,
            # remove_unused_columns=False,
            # run_name="dpo_llama2",
            seed=42,
        )
        trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, peft_config=peft_config)
        trainer.train()