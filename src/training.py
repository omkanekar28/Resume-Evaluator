"""
COMMANDS TO RUN BEFORE TRAINING:

pip install unsloth "xformers==0.0.28.post2"
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"


NOTES:

1. On T4 GPU, if we dont manually reduce 'per_device_train_batch_size' 
from 2 to 1, we run a risk of the notebook crashing while training is 
going on.

2. Once the training is over, the script automatically stores both the 
adapters as well as gguf_version of the finetuned model. So make sure 
sufficient space is available.
"""

import os
from datetime import datetime
import training_config as config
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from utils import fancy_print

fancy_print("Unsloth Training")

# MODEL LOADING
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.MODEL_NAME,
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype = config.DTYPE,
    load_in_4bit = config.LOAD_IN_4BIT
)

# PEFT MODEL LOADING FOR TRAINING
model = FastLanguageModel.get_peft_model(
    model,
    r=config.R,
    target_modules=config.TARGET_MODULES,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    bias=config.BIAS,
    use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    random_state=config.RANDOM_STATE,
    max_seq_length=config.MAX_SEQ_LENGTH,
    use_rslora=config.USE_RSLORA,
    loftq_config=config.LOFTQ_CONFIG,
)

# DATASET LOADING AND PREPROCESSING
tokenizer = get_chat_template(
    tokenizer,
    chat_template = config.CHAT_TEMPLATE,
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("json", data_files=config.DATASET_PATH, split='train')
dataset = dataset.shuffle(seed=config.DATASET_SHUFFLE_SEED)
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset = dataset.train_test_split(test_size=config.VALIDATION_SPLIT_SIZE, shuffle=True, seed=config.SPLIT_SHUFFLE_SEED)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# CREATING DIRECTORY STRUCTURE TO STORE ALL MODEL VERSIONS
model_store_dir = os.path.join(config.MODEL_STORE_DIR, str(datetime.now()).replace(' ', '-'))
merged_16bit_dir = os.path.join(model_store_dir, "merged_16bit")
q4_k_m_dir = os.path.join(model_store_dir, "q4_k_m")
quantized_dir = os.path.join(model_store_dir, "quantized")
os.makedirs(merged_16bit_dir)
os.makedirs(q4_k_m_dir)
os.makedirs(quantized_dir)

# INITIALISING TRAINING PARAMETERS (IMPORTANT)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=config.MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        # max_steps=config.MAX_STEPS,    # DISABLE IF USING 'num_train_epochs' OTHERWISE THIS WILL OVERWRITE IT.
        learning_rate=config.LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim=config.OPTIM,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        seed=config.TRAINING_SEED,
        output_dir=model_store_dir,
        report_to=config.REPORT_TO,
        logging_strategy=config.LOGGING_STRATEGY,
        logging_steps=config.LOGGING_STEPS,
        save_strategy=config.SAVE_STRATEGY,
        save_steps=config.SAVE_STEPS
    ),
)

# TRAINING
trainer_stats = trainer.train()    # ADAPTERS ARE SAVED DURING THIS STEP

# SAVING FULL MODEL (16_bits)
if config.SAVE_16BITS:
    model.save_pretrained_merged(merged_16bit_dir, tokenizer, save_method="merged_16bit",)

# SAVING 'Q4_K_M' VERSION
if config.SAVE_Q4_K_M:
    model.save_pretrained_gguf(q4_k_m_dir, tokenizer, quantization_method="q4_k_m")

# SAVING 'QUANTIZED' VERSION
if config.SAVE_QUANTIZED:
    model.save_pretrained_gguf(quantized_dir, tokenizer, quantization_method="quantized")

fancy_print("The End")
