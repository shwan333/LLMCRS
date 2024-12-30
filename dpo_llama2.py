# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch, json, os

model_id = "/home/work/shchoi/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir="/home/work/shchoi/.cache/huggingface/hub",
    torch_dtype=torch.bfloat16,
    device_map={"": 1},
    ).to('cuda:0')
# import time
# time.sleep(10)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/work/shchoi/.cache/huggingface/hub")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.chat_template = """
#     {% if messages[0]['role'] == 'system' %}
#         {% set offset = 1 %}
#     {% else %}
#         {% set offset = 0 %}
#     {% endif %}

#     {{ bos_token }}
#     {% for message in messages %}
#         {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
#             {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
#         {% endif %}

#         {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
#     {% endfor %}

#     {% if add_generation_prompt %}
#         {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
#     {% endif %}"""
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test", cache_dir="/home/work/shchoi/.cache/huggingface/hub")
save_dir = f'/home/work/shchoi/iEvaLM-CRS/save_5/chat/openmodel_Llama-3.2-1B-Instruct/opendialkg_eval/full_non_repeated/dpo_data_static' 
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
        # "k_proj",
        # "out_proj",
        # "fc_in",
        # "fc_out",
        # "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
gradient_accumulation_steps = 32
epochs = 5
lr = 5e-5

training_args = DPOConfig(
    output_dir=f"Llama-3.2-1B-Instruct_grad_acc_{gradient_accumulation_steps}_lr_{lr}_epochs_{epochs}", 
    logging_steps=10,
    per_device_train_batch_size=1,  # Start with small batch size
    gradient_accumulation_steps=32,  # Increase effective batch size 
    num_train_epochs = 5,
    # max_steps=1000,
    # save_steps=100,
    # gradient_checkpointing=True,
    learning_rate=5e-5,
    bf16=True,
    # remove_unused_columns=False,
    # run_name="dpo_llama2",
    seed=42,
)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, peft_config=peft_config)
trainer.train()