# MODEL PARAMETERS
"""
List of models available for training: 
https://DOCS.unsloth.ai/get-started/all-our-models
"""
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"        # Must be available on unsloth
MAX_SEQ_LENGTH = 4500                               # Choose any! We auto support RoPE Scaling internally!
DTYPE = None                                        # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

# PEFT MODEL PARAMETERS
R=16                                                # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
TARGET_MODULES=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
LORA_ALPHA=16
LORA_DROPOUT=0                                      # Supports any, but = 0 is optimized
BIAS="none"                                         # Supports any, but = "none" is optimized
USE_GRADIENT_CHECKPOINTING="unsloth"                # True or "unsloth" for very long context
RANDOM_STATE=3407
USE_RSLORA=False                                    # We support rank stabilized LoRA
LOFTQ_CONFIG=None                                   # And LoftQ

# DATASET
CHAT_TEMPLATE = "qwen-2.5"
DATASET_PATH = "dataset.json"
DATASET_SHUFFLE_SEED = 65
SPLIT_SHUFFLE_SEED = 42
VALIDATION_SPLIT_SIZE = 0.2

# TRAINING PARAMETERS
PER_DEVICE_TRAIN_BATCH_SIZE = 1                     # Number of training examples processed by each device (GPU/CPU) in one forward/backward pass.
PER_DEVICE_EVAL_BATCH_SIZE = 1                      # Number of evaluation examples processed per device in one forward pass. Lower values reduce memory usage during evaluation.
GRADIENT_ACCUMULATION_STEPS = 4                     # Effective batch size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
EVAL_ACCUMULATION_STEPS = 1                         # Number of steps to accumulate evaluation data before processing. Helps manage memory during evaluation without affecting training.
WARMUP_STEPS = 5                                    # Number of steps to gradually increase the learning rate from 0 to the initial value (LEARNING_RATE).
NUM_TRAIN_EPOCHS = 5                                # Total number of passes over the entire dataset.
# MAX_STEPS = 3                                     # Disable if using 'num_train_epochs' otherwise this will overwrite it.
LEARNING_RATE = 5e-4                                # Initial step size for updating model parameters. Smaller values ensure slower but stable convergence.
OPTIM = "adamw_8bit"                                # Optimizer type for updating model weights. "adamw_8bit" usually works for most cases.
WEIGHT_DECAY = 0.01                                 # Regularization parameter to prevent overfitting by penalizing large weights.
LR_SCHEDULER_TYPE = "linear"                        # Type of learning rate schedule. "linear" means the learning rate decreases linearly from its initial value to 0 over the course of training.
TRAINING_SEED = 3407                                # Random seed for reproducibility. Ensures consistent results when re-running the training script.
REPORT_TO = "none"                                  # Reporting framework for logging metrics (e.g., "wandb", "tensorboard"). "none" disables reporting.
LOGGING_STRATEGY = 'steps'                          # 'epoch' or 'steps'.
LOGGING_STEPS = 15                                   # Frequency (in steps/epoch) of logging training metrics like loss.
SAVE_STRATEGY = 'epoch'                             # 'epoch' or 'steps'.
SAVE_STEPS = 1                                      # Frequency (in steps/epoch) of saving the merged model.

# ONLY ENABLE IF YOU ARE SURE THAT GPU WILL BE USED 24/7 FOR INFERENCE!!!
LOAD_IN_4BIT = False                                # Use 4bit quantization to reduce memory usage. Can be False.

# OUTPUT
MODEL_STORE_DIR = "outputs"                         # Store all model versions here.
SAVE_16BITS = True                                  # Merged model saved in float16.
SAVE_Q4_K_M = True                                  # First quantised version. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K".
SAVE_QUANTIZED = True                               # Second quantised version. Recommended. Slow conversion. Fast inference, small files.
