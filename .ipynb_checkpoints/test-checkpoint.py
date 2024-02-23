"""
This testing script can be run on a single gpu.

To run this testing script, example:
CUDA_VISIBLE_DEVICES=0 python test.py

Data loader:
If custom=False, mean using the dataset from GenomicLLM.
GenomicLLM contains BeckyGRCh38Data, BeckyData, GUEData, HyenaData.
The BeckyGRCh38Data/BeckyData/GUEData/HyenaData_num_samples is the number of samples. The value -1 mean all samples.
Example: CUDA_VISIBLE_DEVICES=0 python test.py --custom=False --BeckyGRCh38Data_num_samples=-1 --GUEData_num_samples=-1

BeckyGRCh38Data contains optional tasks: 'gene biotype','nt2aa','splice site','enhancer', 'orf'.
Example: CUDA_VISIBLE_DEVICES=0 python test.py --custom=False --BeckyGRCh38Data_num_samples=-1 --BeckyGRCh38_data_name=['enhancer', 'splice site']
    
Custom data loader：
If custom=True, mean custom dataset, should input file_name.
GenomicBenchmarks enhancers ensembl test data: './data/Genomic/custom_data/human_enhancers_ensembl_test.txt',
GenomicBenchmarks enhancers cohn test data: './data/Genomic/custom_data/human_enhancers_cohn_test.txt',
Example: CUDA_VISIBLE_DEVICES=0 python test.py --custom=True --file_name='./data/Genomic/custom_data/human_enhancers_cohn_test.txt'
"""

import torch
from model import ModelArgs, Transformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import requests
import subprocess
import glob
from functools import partial
from data_utils import *

# +
# parameters
out_dir = "out"
save_name = "your_generated_text_name"
data_dir = "data"
SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
vocab_size = SPMtokenizer.vocab_size()
checkpoint = './model/ckpt_20240102_len512_Alldata_balance_best.pt'
device = 'cuda'
batch_size = 16
max_seq_len = 512
vocab_source = "llama2" 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# data
custom=True
file_name='./data/custom_data/human_enhancers_cohn_test.txt'
BeckyGRCh38Data_num_samples=0
BeckyGRCh38_data_name=['gene biotype','nt2aa','splice site','enhancer', 'orf']
BeckyData_num_samples=0
GUEData_num_samples=0
HyenaData_num_samples=0

# configuration
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print("config: ", config)


# -

# test dataloader
def test_dataloader(
    custom:bool=False,
    file_name:str='./data/custom_data/human_enhancers_cohn_test.txt',
    BeckyGRCh38Data_num_samples:int=0, 
    BeckyGRCh38_data_name:List[str]=['gene biotype','nt2aa','splice site','enhancer', 'orf'],
    BeckyData_num_samples:int=0, 
    GUEData_num_samples:int=0,
    HyenaData_num_samples:int=0
):
    print('Test dataset...') 
    if custom:
        GenomTestSet = TestData(
            file_name=file_name, 
            tokenizer=SPMtokenizer, 
            text_max_length=1024,  
            ids_max_length=512, 
        )
        testloader = DataLoader(
            GenomTestSet, 
            batch_size = 16,
            shuffle=False,
            collate_fn = collate_fn,
            num_workers = 16,
            pin_memory = True,
        )
        print(f"Number of samples in GenomTestSet: {GenomTestSet.__len__()}")
    else:
        # 所有测试集
        GenomTestSet = GenomicData(
            Becky_txt_file=os.path.join(data_dir,'GenomicLLM_GRCh38/valset_20230814_gene_info_res.txt'),
            BeckyGRCh38_data_name=BeckyGRCh38_data_name, #optional list
            GUE_path=os.path.join(data_dir,"GUE"),
            mode="test", 
            text_max_length=(512-100)*3, 
            ids_max_length=512,
            tokenizer=SPMtokenizer,
            BeckyGRCh38Data_num_samples=BeckyGRCh38Data_num_samples,
            BeckyData_num_samples=BeckyData_num_samples,
            GUEData_num_samples=GUEData_num_samples,
            HyenaData_num_samples=HyenaData_num_samples,
        )
        testloader = GenomTestSet.get_DataLoader(batch_size=batch_size, mode='naive', num_workers=16)
        print(f"Number of samples in GenomTestSet: {GenomTestSet.__len__()}")   ###改
        print(f"Number of samples in GenomTestSet.BeckyGRCh38Dset: {GenomTestSet.BeckyGRCh38Dset_len}")
        print(f"Number of samples in GenomTestSet.BeckyDset: {GenomTestSet.BeckyDset_len}")
        print(f"Number of samples in GenomTestSet.GUEDset: {GenomTestSet.GUEDset_len}")
        print(f"Number of samples in GenomTestSet.HyenaDset: {GenomTestSet.HyenaDset_len}")
    print('Finish dataset obejcts...')
    return testloader, GenomTestSet.equal_id


if __name__ == '__main__':
    # data loader
    testloader, equal_id = test_dataloader(
        custom=config['custom'],
        file_name=config['file_name'],
        BeckyGRCh38Data_num_samples=config['BeckyGRCh38Data_num_samples'],
        BeckyGRCh38_data_name=BeckyGRCh38_data_name, 
        BeckyData_num_samples=config['BeckyData_num_samples'], 
        GUEData_num_samples=config['GUEData_num_samples'], 
        HyenaData_num_samples=config['HyenaData_num_samples'],
    )  

    # load model
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(dtype=torch.bfloat16)
    model.to(device)
    
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # run model    
    TEXT_input, TEXT_true_output, TEXT_pred = [], [], []
    TEXT_true_output_piece, TEXT_pred_piece = [], []
    with torch.no_grad():#torch.inference_mode():
         for ind,(X, Y, LossMask) in enumerate(tqdm(testloader,desc='Generating...')):  #, AttMask
            # 输入给模型的prompt是最后一个‘=’及其之前的内容       
            prompt_tokens = []
            true_out_tokens = []
            idx_eql = []
            for x in X:
                x = x.numpy().tolist()
                idx = find_last_index(x, equal_id) 
                idx_eql.append(idx)
                prompt_tokens.append( x[:idx+1] )
                true_out_tokens.append( x[idx+1:] )
               
            gt_lens = torch.where(Y==SPMtokenizer.eos_id())[1].numpy() - np.array(idx_eql)
            max_gen_len = max(gt_lens)+10 #256

            temperature = 0.0
            top_k = 10
            logprobs = False
            echo = False

            out_tokens, _ = model.generate(
                prompt_tokens = prompt_tokens,
                max_gen_len = max_gen_len, # 生成的句子的最大长度，不包含提示部分。
                temperature = temperature,
                top_k = top_k,
                logprobs = logprobs,
                echo = echo,# False表示函数返回值不包含prompt部分
                tokenizer = SPMtokenizer, 
                device = device,
            )

            TEXT_input = TEXT_input + [SPMtokenizer.decode(tokens) for tokens in prompt_tokens]
            TEXT_true_output = TEXT_true_output + [SPMtokenizer.decode(tokens) for tokens in true_out_tokens]
            TEXT_pred = TEXT_pred + [SPMtokenizer.decode(tokens) for tokens in out_tokens]
            # for BLUE
            TEXT_true_output_piece = TEXT_true_output_piece + [SPMtokenizer.id_to_piece(tokens) for tokens in true_out_tokens]
            TEXT_pred_piece = TEXT_pred_piece + [SPMtokenizer.id_to_piece(tokens) for tokens in out_tokens]
    
    # save
    df = pd.DataFrame(columns=['input text','true output text','generated text', 'true output piece', 'generated piece'])
    df['input text'] = TEXT_input
    df['true output text'] = TEXT_true_output
    df['generated text'] = TEXT_pred

    TEXT_true_output_piece = [tokens[:tokens.index('</s>')] if '</s>' in tokens else tokens for tokens in TEXT_true_output_piece]
    df['true output piece'] = TEXT_true_output_piece
    df['generated piece'] =  TEXT_pred_piece
    
    df.to_csv(os.path.join(out_dir,f'{save_name}.csv') )
