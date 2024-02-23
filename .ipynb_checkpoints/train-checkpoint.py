#!/usr/bin/env python
# coding: utf-8
# %%

"""
This training script can be run on a single gpu in debug mode.
Set parameters in the configurator.py before run this script.

To run on a single GPU small debug run, example:
CUDA_VISIBLE_DEVICES=0 python train.py
"""


# %%
import shutup 
shutup.please()

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from data_utils import *


# %%
from datetime import datetime, timedelta
import pytz
local_time = datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).date().strftime("%Y%m%d")
task_name = f'{local_time}_len512_Alldata_balance'


# %%
# configuration
exec(open("configurator.py").read())  # overrides from command line or config file
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# make out_dir
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
# logging init
import logging
logging.basicConfig(
    filename=os.path.join(out_dir,f'log_{task_name}.log'), 
    level=logging.DEBUG, 
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
logger_blocklist = ["torch"]
for module in logger_blocklist:
    logging.getLogger(module).setLevel(logging.CRITICAL)
logging.info(config)


# %%
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    logging.info(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")


# %%
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype))


# %%
# data loader
SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
vocab_size = SPMtokenizer.vocab_size()

logging.info('Initializing dataset obejcts...')
GenomTrainSet = GenomicData(
    Becky_txt_file=os.path.join(data_dir,'GenomicLLM_GRCh38/trainset_20230814_gene_info_res.txt'),
    BeckyGRCh38_data_name=['gene biotype','nt2aa','splice site','enhancer', 'orf'],
    GUE_path=os.path.join(data_dir,"GUE"),
    mode="train",
    text_max_length=(512-100)*3, 
    ids_max_length=512, 
    tokenizer=SPMtokenizer
    )
trainloader = GenomTrainSet.get_DataLoader(batch_size=batch_size, mode='dataset balance', num_workers=2)

GenomValSet = GenomicData(
    Becky_txt_file=os.path.join(data_dir,'GenomicLLM_GRCh38/valset_20230814_gene_info_res.txt'),
    BeckyGRCh38_data_name=['gene biotype','nt2aa','splice site','enhancer', 'orf'],
    GUE_path=os.path.join(data_dir,"GUE"),
    mode="dev", 
    text_max_length=(512-100)*3, 
    ids_max_length=512, 
    tokenizer=SPMtokenizer,
)
valloader = GenomValSet.get_DataLoader(batch_size=batch_size, mode='dataset balance', num_workers=4)

logging.info('Train dataset...') 
logging.info(f"Number of samples in GenomTrainSet: {GenomTrainSet.__len__()}")
logging.info(f"Number of samples in GenomTrainSet.BeckyGRCh38Dset: {GenomTrainSet.BeckyGRCh38Dset_len}")
logging.info(f"Number of samples in GenomTrainSet.BeckyDset: {GenomTrainSet.BeckyDset_len}")
logging.info(f"Number of samples in GenomTrainSet.GUEDset: {GenomTrainSet.GUEDset_len}")
logging.info(f"Number of samples in GenomTrainSet.HyenaDset: {GenomTrainSet.HyenaDset_len}")
logging.info('Val dataset...')  
logging.info(f"Number of samples in GenomValSet: {GenomValSet.__len__()}")   ###改
logging.info(f"Number of samples in GenomValSet.BeckyGRCh38Dset: {GenomValSet.BeckyGRCh38Dset_len}")
logging.info(f"Number of samples in GenomValSet.BeckyDset: {GenomValSet.BeckyDset_len}")
logging.info(f"Number of samples in GenomValSet.GUEDset: {GenomValSet.GUEDset_len}")
logging.info(f"Number of samples in GenomValSet.HyenaDset: {GenomValSet.HyenaDset_len}")
logging.info('Finish dataset obejcts...')

# %%
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    # resume training from a checkpoint.
    logging.info(f"Resuming training from {resume_ckpt_path}")
    
    checkpoint = torch.load(resume_ckpt_path, map_location=device)
    checkpoint["best_val_loss"] = np.inf
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory


# %%
# compile the model
if compile:
    logging.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# %%
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split,loader in zip(['train', 'val'], [trainloader, valloader]):
        data_iter = iter(loader)
        num_iters = min([eval_iters,len(data_iter)])
        losses = torch.zeros(num_iters)  # keep on CPU
        for k in range(num_iters):
            X, Y, LossMask = next(data_iter)  #, AttMask
            X = X.to(device)
            Y = Y.to(device)
            LossMask = LossMask.to(device)
#             AttMask = AttMask.to(device)
            with ctx:
                logits, loss = model(X, Y, LossMask)#, AttMask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# %%
# training loop
from tqdm import tqdm
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
logging.info("Start training...")
print(f"\n ########################## Start training ###################")
while True:
    # termination conditions
    if iter_num > max_iters:
        break
        
    micro_step = 0
    for X, Y, LossMask in trainloader:  
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        LossMask = LossMask.to(device, non_blocking=True) 
        
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits, loss = model(X, Y, LossMask) #AttMask
            # loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
            
        if micro_step==gradient_accumulation_steps-1:
            iter_num += 1
            local_iter_num += 1
        
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            
            micro_step = 0
            
            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process:
                losses = estimate_loss()
                print(f"Step {iter_num}: train loss {losses['train']:.4f} ,val loss {losses['val']:.4f}")
                logging.info(f"Step {iter_num}: train loss {losses['train']:.4f} ,val loss {losses['val']:.4f}")
                
                if wandb_log:
                    try:
                        wandb.log(
                            {
                                "iter": iter_num,
                                "tokens": iter_num * tokens_per_iter,
                                "loss/train": losses["train"],
                                "loss/val": losses["val"],
                                "lr": lr,
                                "mfu": running_mfu * 100,  # convert to percentage
                            }
                        )
                    except Exception as e:
                        logging.info(f"logging to wandb failed: {e}")
                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        }
                        print(f"saving checkpoint to ckpt_{task_name}_best.pt")
                        torch.save(checkpoint, os.path.join(out_dir, f"ckpt_{task_name}_best.pt"))
                        logging.info(f"saving checkpoint to ckpt_{task_name}_best.pt")
                elif iter_num % 20000 == 0:
                    loss = losses["val"]
                    print(f"saving checkpoint to ckpt_{task_name}_iter={iter_num}_loss={loss}.pt")
                    logging.info(f"saving checkpoint to ckpt_{task_name}_iter={iter_num}_loss={loss}.pt")
                    torch.save(checkpoint, os.path.join(out_dir, f"ckpt_{task_name}_iter={iter_num}_loss={loss}.pt"))

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                logging.info(f"iter {iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
                print(f"iter {iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
                
        micro_step += 1

        # termination conditions
        if iter_num > max_iters:
            break


# %%
if ddp:
    destroy_process_group()

