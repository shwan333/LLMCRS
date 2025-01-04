# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch, json, os, argparse
from src.model.utils import get_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--rec_model', type=str)
    args = parser.parse_args()
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
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
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