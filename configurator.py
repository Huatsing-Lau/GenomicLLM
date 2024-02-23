# -----------------------------------------------------------------------------
# I/O
data_dir = "./data"
out_dir = f"./model"  # model out dir
eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
resume_ckpt_path = "./model/ckpt_20231212_len512_Alldata_balance_best.pt"

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# data
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 512

# model
dim = 768
n_layers = 12
n_heads = 12
n_kv_heads = 12
multiple_of = 768
dropout = 0.1

# adamw optimizer
gradient_accumulation_steps = 4 #2 # used to simulate larger batch sizes
learning_rate = 6e-4  #6e-4  # max learning rate
max_iters = 5e5  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 500  # how many steps to warm up for

# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 1e-12 #3e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
