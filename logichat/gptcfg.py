# based on shakespeare_char model

cfgname = "large"
dsname = "openwebtext"

out_dir = f'out-logichat-{cfgname}'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = f'logichat-{cfgname}'
wandb_run_name = 'mini-gpt'

dataset = f'openwebtext/{cfgname}'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on some systems also add
device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# shrink some training parms further
eval_interval = 150
eval_iters = 20
batch_size = 32

# run train with --init_from=scratch to start fresh
init_from = 'resume'
