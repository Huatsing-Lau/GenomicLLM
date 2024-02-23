# 碱基序列片段在前，标签在后。根据序列和标签提示符，生成标签取值。

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
import random
import sentencepiece as spm
import torch
import random
from typing import List, Tuple, Literal
import zipfile


# +
def reverse_complement(sequence:str):
    """
    用于将剪辑序列中的简并碱基替换为别的字符。
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 
                  'B':'N', 'D':'N', 'E':'N', 'F':'N', 'H':'N', 'I':'N', 'J':'N', 'K':'N', 'L':'N', 'M':'N', 'N':'N',
                  'O':'N', 'P':'N', 'Q':'N', 'R':'N', 'S':'N', 'U':'N', 'V':'N', 'W':'N', 'X':'N', 'Y':'N', 'Z':'N',
                 }
    reverse_sequence = sequence[::-1]
    reverse_complement_sequence = ''.join(complement[base] for base in reverse_sequence)
    return reverse_complement_sequence

def complement(sequence:str):
    """
    互补碱基。
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 
                  'B':'N', 'D':'N', 'E':'N', 'F':'N', 'H':'N', 'I':'N', 'J':'N', 'K':'N', 'L':'N', 'M':'N', 'N':'N',
                  'O':'N', 'P':'N', 'Q':'N', 'R':'N', 'S':'N', 'U':'N', 'V':'N', 'W':'N', 'X':'N', 'Y':'N', 'Z':'N',
                 }
    sorted(set([nt for nt in sequence]))
    complement_sequence = ''.join(complement[base] for base in sequence)
    return complement_sequence

def reverse(sequence:str):
    reverse_sequence = sequence[::-1]
    return reverse_sequence

def random_sample(lst, k):
    return random.sample(lst, k)

def dictflatten(src_dict:dict,prefix=None):
    flatten_linedict = dict([])
    for key,v in src_dict.items():
        if type(v)==dict:
            flatten_linedict.update( dictflatten(v,prefix=key) )
        else:
            if prefix is not None:
                flatten_linedict[f'{prefix}:{key}'] = v
            else:
                flatten_linedict[f'{key}'] = v
    return flatten_linedict

def find_last_index(lst, value):
    try:
        reversed_list = lst[::-1]  # 反转列表
        first_index = reversed_list.index(value)  # 获取反转列表中值第一次出现位置
        last_index = len(lst) - 1 - first_index  # 计算反转后的位置在原始列表中的位置
        return last_index
    except ValueError:
        return -1  # 如果值不存在于列表中，则返回-1
    
def get_GCcontent(seq:str):
    GC = sum([1 for nt in seq if nt in ['G','C']]) / len(seq)
    return GC

def sample_dict(dictionary, sample_size):
    keys = list(dictionary.keys())
    sampled_keys = random.sample(keys, sample_size)
    sampled_dict = {key: dictionary[key] for key in sampled_keys}
    return sampled_dict

def rename_dict_keys_to_serial_numbers(dictionary):
    """
    重命名字典的key为0,1,2,3,...这样的序号.
    """
    new_dict = {i: dictionary[key] for i, key in enumerate(dictionary)}
    return new_dict

def rename_dict_keys(my_dict:dict,keys:dict):
    for old_key,new_key in keys.items():
        if old_key in my_dict:
            my_dict[new_key] = my_dict.pop(old_key)
    return my_dict


import random
def sample_with_probabilities(lst, probabilities):
    sampled_element = random.choices(lst, probabilities)[0]
    return sampled_element


# +
class SPMprocess(object):
    def __init__(self,sp,max_length=None,padding=False):
        super().__init__()
        self.sp = sp
        self.max_length = max_length
        self.padding = padding
    
    def truncate(self,pieces,max_length:int):
        if len(pieces) > max_length:
            pieces = pieces[:max_length]  # 截断
        return pieces   
    
    def pad(self,pieces,max_length:int):
        if len(pieces) < max_length:
            padding_length = max_length - len(pieces)
            pieces += ['</s>'] + ['<pad>']*(padding_length-1)  # 填充 # eos_token = sp.IdToPiece(sp.eos_id())
        return pieces

    
    def __call__(self,text):
        if (self.max_length is None) and (not self.padding):
            ids = SPMtokenizer.encode_as_ids(text)
            ids.append(self.sp.eos_id())# add the end of text token
        else:
            pieces = self.sp.EncodeAsPieces(text)
            if self.max_length > 0:
                pieces = self.truncate(pieces, self.max_length)
            if self.padding:
                pieces = self.pad(pieces, self.max_length)
            ids = self.sp.piece_to_id(pieces)
        return ids
    
from Bio.SeqUtils import seq3
def convert_single_to_three_letter(sequence):
    """ 
    使用 Biopython 将单字母的氨基酸序列转换为三字母的氨基酸序列。
    """
    three_letter_sequence = seq3(sequence)
    return three_letter_sequence


# -

def collate_fn(items:List[dict]):
    """
    items: GenomicData.__getitem__提供。
    """
    X = torch.stack( [torch.tensor(item['ids'][:-1],dtype=torch.int64) for item in items] )
    Y = torch.stack( [torch.tensor(item['ids'][1:],dtype=torch.int64) for item in items] )
    Mask = torch.stack( [torch.tensor(item['mask'][1:],dtype=torch.int64) for item in items] )
    # 根据Mask裁剪掉多余的pad
    rows, cols = torch.where(Mask==1)
    clip_to = cols.max().item()
    X = X[:,:clip_to]
    Y = Y[:,:clip_to]
    Mask = Mask[:,:clip_to]
    return X, Y, Mask


# +
class BeckyData(torch.utils.data.Dataset):
    """
    定义数据集对象.
    seq: 序列列表,列表内每个元素是一个字符串,表示碱基序列.
    label: 标签列表. 列表内的每个元素都是一个数值,表示序列的类型或者分数.
    encoder: 编码器,调用该编码器的__call__方法,可以将序列字符串转为数组.
    transform: defalut=None, 表示数据扩增操作对象,此处主要是同义密码子转换.
    """
    def __init__(self, txt_file, text_max_length=512, ids_max_length=256, tokenizer=None, transform=None, num_samples:int=-1):
        super().__init__()
        self.txt_file = txt_file
        self.text_max_length = text_max_length
        self.ids_max_length = ids_max_length
        self.tokenizer = tokenizer
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.dsetdict = self.lines2dsetdict(self.read_txt_file(self.txt_file))
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id()
        if num_samples>0:
            self.dsetdict = sample_dict( self.dsetdict, num_samples )
            # 重命名key,从0开始
            self.dsetdict = rename_dict_keys_to_serial_numbers( self.dsetdict )
        elif num_samples==0:
            self.dsetdict = {}
    
    def read_txt_file(self,txt_file):
        # 读取文件内容
        with open( txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines
    
    def lines2dsetdict(self,lines):
        dsetdict = {}
        for i in range(1,len(lines)):
            linedict = self.line2dict(lines[i])
            dsetdict[i-1] = linedict
        return dsetdict
    
    def line2dict(self,line):
        """
        将1行样本text转为字典。
        """
        linedict = {
            'chromosome_id': None, 
            'feature_type': None, 
            'symbol': None, 
            'id': None, 
            'start': None, 
            'end': None, 
            'sequence direction': None, 
            'annotation': None, 
            'seq': None }

        for key,value in  zip(linedict.keys(), line.split('\t')):
            key = key.strip()
            if key=='seq':
                split_list = value.split('N'*20)
                value = list(filter(lambda x: x != '', split_list))[0]
            if key=='sequence direction':
                value = 'forward' if value=='+' else 'reverse'
            if key=='chromosome_id':
                if value.split('.')[0] in [f'NC_0000{k:02}' for k in range(1,24+1)]:
                    # NC_000001,..., NC_000024共24条染色体
                    value = f"chromosome{int(value.split('.')[0][-2:])}" 
                else:
                    # NT, NW
                    value = value.split('_')[0]# 只需要类别号，不需要数字编号
                    
            linedict[key] = value
        linedict['annotation'] = dict(item.split("=") for item in linedict['annotation'].split(';'))
        # 更改名称
        linedict['annotation'] = rename_dict_keys(linedict['annotation'], keys={'gene_biotype': 'gene biotype'})
        # gene biotype的类别改为'protein coding gene' 和 'other'(在get_a_random_text_from_linedict)
        if 'gene biotype' in linedict['annotation']:
            if linedict['annotation']['gene biotype'] == 'protein_coding':
                linedict['annotation']['gene biotype'] = 'protein coding gene'
    
        linedict['annotation']['Dbxref'] = dict(item.split(":",maxsplit=1) for item in linedict['annotation']['Dbxref'].split(','))
        return linedict
    
    def get_a_random_text_from_linedict(self, linedict:dict, max_length:int):
        flatten_linedict = dictflatten(linedict)   
        # 分离出属于QA的key,value;以及属于已知信息的key,value
        Qkeys = [ 'sequence direction', 'annotation:gene biotype', 'annotation:description']
        QA_linedict = dict([])
        for key in Qkeys:
            try:
                QA_linedict.update( {key:flatten_linedict.pop(key)} )
            except:
                pass

        seq = flatten_linedict.pop('seq')

        # 随机挑选若干作为已知信息
        known_info = []
        length = 0 
        for key in random_sample( list(flatten_linedict.keys()), k=random.randint(1,min(4,len(flatten_linedict))) ):
            known_info.append( f'{key}={flatten_linedict[key]}' ) 
            length += len(f'{key}={flatten_linedict[key]}')
            # 如果已知信息文本长度 >60，不再添加
            if length > 60: 
                break       
        # 根据已知信息长度，修剪seq长度，需预留一定数量的token（默认50个）给QA。
        max_length = max_length-length-50
        ix = random.randint(0,max(0,len(seq)-max_length))
        seq = seq[ix:ix+max_length]        
            
        # 最后随机挑选一个作为问题和答案：
        key = random.choice( list(QA_linedict.keys())+['GC content', 'reading comprehension'] )
        if key == 'sequence direction':
            # forward(+): 5'到3', 
            # reverse(-): 3'到5'
            value = 'forward'# 原始序列，其正确标签都是+ 
            if random.randint(0,1)==1:
                # 随机反向
                value = 'reverse' if value=='forward' else 'forward'# 'reverse' if v=='forward' else 'forward'
                seq = reverse(seq)
            else:
                pass
        elif key == 'GC content': 
            value = f'{get_GCcontent(seq):0.3f}'
        elif key == 'reading comprehension':
            # 阅读理解
            # 从known_info中随机挑选一个QA出来
            known_QA = random.choice(known_info)
            known_key, known_value = known_QA.split('=')
            known_key, known_value = known_key.strip(), known_value.strip()
            key = f'reading comprehension: {known_key}'
            value = known_value
        elif key == 'annotation:gene biotype':
            value = QA_linedict[key] if QA_linedict[key]=='protein coding gene' else 'other'
        else:
            value = QA_linedict[key]
        QA_text = f'{key}={value}'
            
        # 随机放置known_info和seq
        known_info = ', '.join(known_info)
        if random.randint(0,1)==0:
            text = f'{seq}, {known_info}, {QA_text}'
        else:
            text = f'{known_info}, {seq}, {QA_text}'
        return text

    
    def text2ids(self,text):
        ids = self.SPMprocess(text)
        return ids
            
    def get_mask(self,ids:List[int],text):
        """mask最后一个Q=A前的部分"""
        # 寻找最后一个等号
        equal_idx = find_last_index(ids, self.equal_id)
        # 寻找eos位置
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)-1
        # mask：
        # 序列和第5个之后的<pad>取值0
        pad5_idx = min(len(ids)-1, eos_idx+5)
        mask = [0]*(equal_idx+1) + [1]*(pad5_idx-equal_idx) + [0]*(len(ids)-1-pad5_idx)
        return mask
        
    def __len__(self):
        return len(self.dsetdict)
    
    def __getitem__(self, i):
        text = self.get_a_random_text_from_linedict(self.dsetdict[i], max_length=self.text_max_length)
        ids = self.text2ids(text)
        # get mask
        mask = self.get_mask(ids,text)
        item = {'text':text, 'ids':ids, 'mask':mask}    
        return item
    
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    BeckyDset = BeckyData('./data/GenomicLLM_GRCh38/valset_20230814_gene_info_res.txt',text_max_length=(512-100)*3, ids_max_length=512, tokenizer=SPMtokenizer) 
    print( BeckyDset.__len__() )
    it = BeckyDset.__getitem__(0)
    print( it )
# -



class BeckyGRCh38Data(torch.utils.data.Dataset):
    def __init__(self, data_name:List[str],mode:str="train", text_max_length=512, ids_max_length=512, tokenizer=None, num_samples:int=-1):
        super().__init__()
        self.data_name = data_name
        self.mode = mode
        self.ids_max_length = ids_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id()
        self.dsetdict = self.read_file(data_name=self.data_name, mode=self.mode)
        self.num_samples = num_samples
        if num_samples>0:
            self.dsetdict = sample_dict( self.dsetdict, num_samples )
            # 重命名key,从0开始
            self.dsetdict = rename_dict_keys_to_serial_numbers( self.dsetdict )
        elif num_samples==0:
            self.dsetdict = {}
            
        
    def read_file(self,data_name:List[str]=None, mode:str="train"):  
        #get data file list
        data_file_dict = {    
            'splice site': f'./data/GenomicLLM_GRCh38/splice_400_{mode}.csv',
            'gene biotype':f'./data/GenomicLLM_GRCh38/20230906_gene_res_{mode}.csv',# 考虑是否起别的名字
            'nt2aa': f'./data/GenomicLLM_GRCh38/20230906_cds_res_nt2aa_{mode}.csv',
            'aa2nt': f'./data/GenomicLLM_GRCh38/20230906_cds_res_aa2nt_{mode}.csv', 
            'enhancer': f'./data/GenomicLLM_GRCh38/20230906_enhancer_res_{mode}.csv',
            'orf': f'./data/GenomicLLM_GRCh38/20230905_3orf_res_{mode}.csv',
        }
        
        if data_name == None:
            data_file_list = list(data_file_dict.values()) 
        else:
            data_file_list = []
            for name in data_name:
                assert name in data_file_dict.keys(), f"{name} not in {data_file_dict.keys}."
                data_file_list.append( data_file_dict[name] )
            
        #read data to dict    
        dsetdict = {}
        for file_name in data_file_list:
            df_data = pd.read_csv(file_name)
            df_data = df_data[~df_data.seq.str.contains('NNNNNNNNNN')].reset_index(drop=True)
            df_data.index = df_data.index + len(dsetdict)
            dsetdict = {**dsetdict, **df_data.to_dict(orient='index')}
        return dsetdict
    
    def get_text_from_linedict(self,linedict:dict,max_length:int=512):
        rdnum = random.random()
        if rdnum<0.2:
            # 碱基序列的反向序列、互补序列、 反向互补序列
            # 随机截片段
            if 'nucleotide seq' in linedict:# aa2nt数据集
                seq = linedict['nucleotide seq']
            else:
                seq = linedict['seq']
            rdlen = random.randint( 1,min([300,len(seq),max_length//2]) )
            rdst = random.randint( 0,max(0,len(seq)-rdlen) )
            end = rdst+rdlen
            seq = seq[rdst:end]
            key = random.choice(['reverse sequence', 'complementary sequence', 'reverse complementary sequence'])
            if key=='reverse sequence':
                value = reverse(seq)
            elif key=='complementary sequence':
                value = complement(seq)
            elif key=='reverse complementary sequence':
                value = reverse_complement(seq)
            text = f"{seq}, {key} = {value}"     
        else:
            keys = list(linedict.keys())
            keys.remove('seq')
            if 'protein seq' in list(linedict.keys()):# nt2aa任务数据集
                aa_seq = linedict['protein seq']
                if rdnum>=0.2 and rdnum<0.5 and len(aa_seq)>50:
                    # 氨基酸序列完型填空
                    min_len = 50
                    rdlen = random.randint( min_len, min(max_length//3-20,len(aa_seq)) )
                    rdst = random.randint(0, len(aa_seq)-rdlen)
                    end = rdst+rdlen
                    aa_seq = aa_seq[rdst:end]
                    # 随机抠出一段氨基酸片段
                    comp_len = random.randint(1,3)# random.randint(1,5)# 完型填空的长度
                    rdst = random.randint( 0, len(aa_seq)-comp_len )
                    # 答案，# 转化为氨基酸3字母表示
                    value = aa_seq[rdst:rdst+comp_len]
                    value = convert_single_to_three_letter(value)
                    # 序列，# 转化为氨基酸3字母表示
                    aa_seq = convert_single_to_three_letter(aa_seq)
                    aa_seq = aa_seq[:rdst*3] + '_'*comp_len + aa_seq[rdst*3+comp_len*3:]
                    # text
                    text = f"{aa_seq}, complete the blank = {value}"
                else:
                    # 碱基翻译成氨基酸
                    # nt 序列 在3的倍数上，随机选取start 和 len 
                    rdlen = random.choice( range(3, min([360,len(linedict['seq']),max_length//4*3-30]), 3) )
                    rdst = random.choice( range(0, len(linedict['seq'])-rdlen, 3) )  
                    end = min(rdst+rdlen, len(linedict['seq']))
                    nt_seq = linedict['seq'][rdst: end]
                    aa_seq = linedict['protein seq'][int(rdst/3):int(end/3)]
                    aa_seq = convert_single_to_three_letter(aa_seq)
                    text = f"{nt_seq}, protein sequence = {aa_seq}"
            elif 'nucleotide seq' in linedict:# aa2nt数据集
                aa_seq = linedict['seq']
                if rdnum>=0.2 and rdnum<0.5 and len(aa_seq)>50:
                    # 氨基酸序列完型填空
                    min_len = 50
                    rdlen = random.randint( min_len, min(max_length//3-20,len(aa_seq)) )
                    rdst = random.randint(0, len(aa_seq)-rdlen)
                    end = rdst+rdlen
                    aa_seq = aa_seq[rdst:end]
                    # 随机抠出一段氨基酸片段
                    comp_len = random.randint(1,3)# random.randint(1,5)# 完型填空的长度
                    rdst = random.randint( 0, len(aa_seq)-comp_len )
                    # 答案，# 转化为氨基酸3字母表示
                    value = aa_seq[rdst:rdst+comp_len]
                    value = convert_single_to_three_letter(value)
                    # 序列，# 转化为氨基酸3字母表示
                    aa_seq = convert_single_to_three_letter(aa_seq)
                    aa_seq = aa_seq[:rdst*3] + '_'*comp_len + aa_seq[rdst*3+comp_len*3:]
                    # text
                    text = f"{aa_seq}, complete the blank = {value}"
                else:
                    # 氨基酸翻译成碱基
                    min_len = 20
                    rdlen = random.choice( range(min_len, min([60,len(linedict['seq']),max_length//4-30])) )
                    rdst = random.choice([num for num in range(0, len(linedict['seq'])-rdlen)]) 
                    end = min(rdst+rdlen, len(linedict['seq']))
                    aa_seq = linedict['seq'][rdst:end]
                    aa_seq = convert_single_to_three_letter(aa_seq)
                    nt_seq = linedict['nucleotide seq'][rdst*3:end*3]
                    text = f"{aa_seq}, nucleotide sequence = {nt_seq}"
            else:
                # 随机截片段
                rdst = random.randint(0,max(0,len(linedict['seq'])-max_length))# 只有当linedict['seq']长度比max_length长，才会截片段
                seq = linedict['seq'][rdst:rdst+max_length]
                key = keys[0]
                value = linedict[key] 
                value = 'other' if value=='pseudogene' else value #  'pseudogene' 改为 'other'
                text = f"{seq}, {key} = {value}"
        return text
    
    def text2ids(self,text):
        ids = self.SPMprocess(text)
        return ids
    
    def get_mask(self,ids:List[int],text):
        """mask最后一个Q=A前的部分"""
        # 寻找最后一个等号
        equal_idx = find_last_index(ids, self.equal_id)
        # 寻找eos位置
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)-1
        # mask：
        # 序列和第5个之后的<pad>取值0
        pad5_idx = min(len(ids)-1, eos_idx+5)
        mask = [0]*(equal_idx+1) + [1]*(pad5_idx-equal_idx) + [0]*(len(ids)-1-pad5_idx)
        return mask
    
    def __len__(self):
        return len(self.dsetdict)
    
    def __getitem__(self, i): 
        text = self.get_text_from_linedict(self.dsetdict[i],self.text_max_length)
        ids = self.text2ids(text)
        # get mask
        mask = self.get_mask(ids,text)
        # item
        item = {'text':text, 'ids':ids, 'mask':mask}    
        return item
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    #data_name:gene biotype/splice/nt2aa/aa2nt/enhancer，可列表['nt2aa','aa2nt']  mode="train"/"test"
    data_name=['splice site']#['splice site']#['splice']#=['gene biotype','nt2aa','splice','enhancer']
    BeckyGRCh38Dset = BeckyGRCh38Data(data_name=data_name,mode="test",tokenizer=SPMtokenizer)
    print(BeckyGRCh38Dset.__getitem__(10))



# +
class GUEData(torch.utils.data.Dataset):
    def __init__(self, path:str="GUE", mode:str="train", ids_max_length=256, tokenizer=None, num_samples:int=-1):
        super().__init__()
        self.mode = mode
        self.ids_max_length = ids_max_length
        self.tokenizer = tokenizer
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id()
        self.path = path
        self.dsetdict = self.read_file(path=path, mode=self.mode)
        self.num_samples = num_samples
        if num_samples>0:
            self.dsetdict = sample_dict( self.dsetdict, num_samples )
            # 重命名key,从0开始
            self.dsetdict = rename_dict_keys_to_serial_numbers( self.dsetdict )
        elif num_samples==0:
            self.dsetdict = {}
            
    def read_file(self,path="GUE",mode="train"):
        # 读取文件
        file_list = [os.path.join(path, file) for file in os.listdir(path) if 'csv' in file]
        file_list.sort()
        
        #写入字典
        dsetdict={}
        for name in [name for name in file_list if mode in name]:
            df_data = pd.read_csv(name)
            df_data = df_data[~df_data.seq.str.contains('NNNNNNNNNN')].reset_index(drop=True)
            df_data.index = df_data.index + len(dsetdict)
            dsetdict = {**dsetdict, **df_data.to_dict(orient='index')}

        return dsetdict
    
    def get_text_from_linedict(self,linedict,max_length=512):  
        if random.random()<0.1 and self.mode=="train":
            # 碱基序列的反向序列、互补序列、 反向互补序列
            seq = linedict['seq']
            # 随机截片段  
            rdlen = random.randint( 1,min([300,len(seq),max_length//2]) )
            rdst = random.randint( 0,max(0,len(seq)-rdlen) )
            end = rdst+rdlen
            seq = seq[rdst:end]
            key = random.choice(['reverse sequence', 'complementary sequence', 'reverse complementary sequence'])
            if key=='reverse sequence':
                value = reverse(seq)
            elif key=='complementary sequence':
                value = complement(seq)
            elif key=='reverse complementary sequence':
                value = reverse_complement(seq)
            text = f"{seq}, {key} = {value}" 
        else:    
            keys = list(linedict.keys())
            keys.remove('seq')
            text = f"{linedict['seq']}"
            for key in keys:
                text = text if pd.isna(linedict[key]) else text + f", {key} = {linedict[key]}" 
        return text
    
    def text2ids(self,text):
        ids = self.SPMprocess(text)
        return ids
    
    def get_loss_mask(self,ids:List[int],text):
        """mask最后一个Q=A前的部分"""
        # 寻找最后一个等号
        equal_idx = find_last_index(ids, self.equal_id)
        # 寻找eos位置
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)-1
        # mask：
        # 序列和第5个之后的<pad>取值0
        pad5_idx = min(len(ids)-1, eos_idx+5)
        mask = [0]*(equal_idx+1) + [1]*(pad5_idx-equal_idx) + [0]*(len(ids)-1-pad5_idx)
        return mask
    
    def __len__(self):
        return len(self.dsetdict)
    
    def __getitem__(self, i): 
        text = self.get_text_from_linedict(self.dsetdict[i],40*3)#序列转换的长度不要太长，否则可能比较困难 ，且测试也比较困难#,self.ids_max_length*3)
        ids = self.text2ids(text)
        mask = self.get_loss_mask(ids,text)
        item = {'text':text, 'ids':ids, 'mask':mask}    
        return item
    
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    GUEDset = GUEData(path="./data/GUE",mode="test",tokenizer=SPMtokenizer)
    print( GUEDset.__len__() )
    print( GUEDset.__getitem__(1) )
# -



# +
class HyenaData(torch.utils.data.Dataset):
    def __init__(self, data_name:List[str],mode:str="train", text_max_length=512, ids_max_length=512, tokenizer=None, num_samples:int=-1):
        super().__init__()
        self.data_name = data_name
        self.mode = mode
        self.ids_max_length = ids_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id()
        self.dsetdict, self.weights = self.read_file(data_name=self.data_name, mode=self.mode)
        self.num_samples = num_samples
        if num_samples>0:
            self.dsetdict = sample_dict( self.dsetdict, num_samples )
            # 重命名key,从0开始
            self.dsetdict = rename_dict_keys_to_serial_numbers( self.dsetdict )
        elif num_samples==0:
            self.dsetdict = {}            
        
    def read_file(self,data_name:List[str]=None, mode:str="train"):  
        #get data file list
        data_file_dict = {           
                'enhancer all': f'./data/GenomicBenchmarks/human_enhancers_all_{mode}.txt',
                'regulatory ensembl': f'./data/GenomicBenchmarks/human_ensembl_regulatory_{mode}.txt',
                'promoter notata': f'./data/GenomicBenchmarks/human_nontata_promoters_{mode}.txt',
                'ocr ensembl': f'./data/GenomicBenchmarks/human_ocr_ensembl_{mode}.txt', #Chromatin Open Regions
        }
            
        #read data to dict    
        dsetdict = {}
        weights = []
        dsetsizedict = {}
        for key, file_name in data_file_dict.items():
            df_data = pd.read_csv(file_name)
            df_data = df_data[df_data.seq.apply(lambda x: len(x) >=50 and len(x) <=600)]
            df_data = df_data[~df_data.seq.str.contains('NNNNNNNNNN')]
            df_data = df_data.reset_index(drop=True)
    
            df_data.rename(columns={"Region":"chromosome"}, inplace=True)
            df_data.index = df_data.index + len(dsetdict)
            dsetdict = {**dsetdict, **df_data.to_dict(orient='index')}
            dsetsizedict[key]  = len(df_data)

        return dsetdict, weights
    
    def get_text_from_linedict(self,linedict:dict,max_length:int=512):       
        keys = list(linedict.keys())
        keys.remove('seq')
        text = linedict['seq']
        for key in keys:
            text = text if pd.isna(linedict[key]) else f"{text}, {key} = {linedict[key]}"
        return text
    
    def text2ids(self,text):
        ids = self.SPMprocess(text)
        return ids
    
    def get_loss_mask(self,ids:List[int]):
        """mask最后一个Q=A前的部分"""
        # 寻找最后一个等号
        equal_idx = find_last_index(ids, self.equal_id)
        # 寻找eos位置
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)-1
        # mask：
        # 序列和第5个之后的<pad>取值0
        pad5_idx = min(len(ids)-1, eos_idx+5)
        loss_mask = [0]*(equal_idx+1) + [1]*(pad5_idx-equal_idx) + [0]*(len(ids)-1-pad5_idx)
        return loss_mask
    
    def __len__(self):
        return len(self.dsetdict)
    
    def __getitem__(self, i): 
        text = self.get_text_from_linedict(self.dsetdict[i],self.text_max_length)
        ids = self.text2ids(text)
        # get loss mask
        loss_mask = self.get_loss_mask(ids)
        # item
        item = {'text':text, 'ids':ids, 'mask':loss_mask }
        return item
    
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    data_name = None
    HyenaDset = HyenaData(data_name=data_name,mode="test",tokenizer=SPMtokenizer, text_max_length=256, ids_max_length=256)
    print( HyenaDset.__len__() )
    print( HyenaDset.__getitem__(1000))
    
# -



# +
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
class GenomicData():
    """
    定义数据集对象.
    seq: 序列列表,列表内每个元素是一个字符串,表示碱基序列.
    label: 标签列表. 列表内的每个元素都是一个数值,表示序列的类型或者分数.
    encoder: 编码器,调用该编码器的__call__方法,可以将序列字符串转为数组.
    transform: defalut=None, 表示数据扩增操作对象,此处主要是同义密码子转换.
    """
    def __init__(self, 
                 Becky_txt_file:str, 
                 BeckyGRCh38_data_name:Tuple[str],
                 GUE_path:str, 
                 mode:str='train', 
                 text_max_length=512, 
                 ids_max_length=256, 
                 tokenizer=None, 
                 transform=None,
                 GUEData_num_samples:int=-1,
                 BeckyGRCh38Data_num_samples:int=-1,
                 HyenaData_num_samples:int=-1,
                 BeckyData_num_samples:int=-1,
                ):
        super().__init__()
        
        self.Becky_txt_file = Becky_txt_file
        self.BeckyGRCh38_data_name = BeckyGRCh38_data_name
        self.GUEData_num_samples = GUEData_num_samples
        self.HyenaData_num_samples = HyenaData_num_samples
        self.BeckyData_num_samples = BeckyData_num_samples
        
        self.text_max_length = text_max_length
        self.ids_max_length = ids_max_length
        self.tokenizer = tokenizer
        self.mode = mode
        
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id
        
        # Becky数据集
        self.BeckyDset = BeckyData(
            Becky_txt_file, 
            tokenizer=tokenizer,
            text_max_length=text_max_length, 
            ids_max_length=ids_max_length, 
            num_samples=BeckyData_num_samples,
        )
        self.BeckyDset_len = self.BeckyDset.__len__()
        
        # BeckyGRCh38数据集
        self.BeckyGRCh38_data_name = BeckyGRCh38_data_name
        self.BeckyGRCh38Dset = BeckyGRCh38Data(
            data_name=BeckyGRCh38_data_name, 
            mode=self.mode, 
            tokenizer=tokenizer,
            text_max_length=text_max_length, 
            ids_max_length=ids_max_length, 
            num_samples=BeckyGRCh38Data_num_samples,
        )
        self.BeckyGRCh38Dset_len = self.BeckyGRCh38Dset.__len__()
        
        # GUE数据集
        self.GUEDset = GUEData(
            path=GUE_path,
            mode=self.mode,
            tokenizer=tokenizer, 
            ids_max_length=ids_max_length, 
            num_samples=GUEData_num_samples,
        )
        self.GUEDset_len = self.GUEDset.__len__()
        
        # Hyena 数据集
        self.HyenaDset =HyenaData(
            data_name=None, 
            mode=self.mode, 
            tokenizer=tokenizer, 
            ids_max_length=ids_max_length,
            num_samples=self.HyenaData_num_samples,
        )
        self.HyenaDset_len = self.HyenaDset.__len__()
        
        # 将子数据集合并成一个大的数据集
        self.ConcatDset= ConcatDataset([self.BeckyDset, self.BeckyGRCh38Dset, self.GUEDset, self.HyenaDset])
        
    
    def __len__(self):
        return len(self.ConcatDset)
    
    
    def get_DataLoader(
        self, 
        batch_size=2, 
        mode:Literal['naive', 'sample balance', 'dataset balance', 'user defined sampler'] = 'naive', 
        sampler=None, 
        shuffle=False, 
        num_workers:int=0):
        
        if mode=='dataset balance':
            """
            若是balance模式，则子数据集加权均衡。
            """
            # 计算每个子数据集中每个样本的采样权重
            weights = []
            for dset in self.ConcatDset.datasets:
                weights += [self.__len__()/dset.__len__()] * dset.__len__()
                
            # 创建带有权重的随机采样器
            sampler = WeightedRandomSampler(weights, num_samples=self.__len__(), replacement=True)

            # 创建父级DataLoader，包含多个子DataLoader
            GenomicDataloader = DataLoader(
                self.ConcatDset, 
                batch_size = batch_size,
                sampler = sampler,
                collate_fn = collate_fn,
                num_workers = num_workers,
                pin_memory = True,
            )
        
        elif mode in ['naive','sample balance']:
            """
            该模式下，所有样本均衡。
            """
            GenomicDataloader = DataLoader(
                self.ConcatDset, 
                batch_size = batch_size,
                shuffle = shuffle,
                collate_fn = collate_fn,
                num_workers = num_workers
            ) 
                
        elif mode=='user defined sampler':
            """
            用户自定义sampler
            """
            assert sampler is not None, "sampler must be provided when mode=='user defined sampler'"
            GenomicDataloader = DataLoader(
                self.ConcatDset, 
                batch_size = batch_size,
                sampler = sampler,
                collate_fn = collate_fn,
                num_workers = num_workers
            ) 
                
        return GenomicDataloader
    
def user_defined_weights(dset,weights:list,task:list,ratio:float):
    """
    Input Arguments:
        dset: dataset对象
        weights： 原始的weights
    """
    mask = []
    for task_ in task:
        task_mask = [ind for ind, val in enumerate(dset.dsetdict.values()) if (task_ in val)]
        mask += task_mask
        print( f"{task_}样本量占比： {len(task_mask)/dset.__len__()}" )
    new_weight = sum(weights)*ratio/len(mask) # 占比ratio%
    new_other_weight = (sum(weights)-new_weight*len(mask)) / (len(weights)-len(mask))
    weights = [new_other_weight]*len(weights)
    for ind in mask:
        weights[ind] = new_weight
    print( f"修改后{task}采样总概率：{sum([weights[ind] for ind in mask])/sum(weights)}")
    return weights
        
        
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    GenomicDset = GenomicData(
        Becky_txt_file='./data/GenomicLLM_GRCh38/valset_20230814_gene_info_res.txt',
        BeckyGRCh38_data_name=['gene biotype','nt2aa','splice site','enhancer'],#['gene biotype','nt2aa','splice','enhancer'],
        GUE_path= "./data/GUE/",  # "./GUE/GUE.zip",
        mode="test", 
        text_max_length=(512-100)*3, 
        ids_max_length=512, 
        tokenizer=SPMtokenizer,
    )
    # 'dataset balance'
    dataloader = GenomicDset.get_DataLoader(batch_size=12, mode='dataset balance', num_workers=2)
    
    """# user defined sampler
    # 计算每个子数据集中每个样本的采样权重。对于BeckyGRCh38Data，提高其中'protein seq'任务的采样概率
    weights = []
    for dset in GenomicDset.ConcatDset.datasets:
        sub_weights = [GenomicDset.__len__()/dset.__len__()] * dset.__len__()
        print(dset.__len__(), sub_weights[0])
        if isinstance(dset, BeckyGRCh38Data):
            sub_weights = user_defined_weights(dset,sub_weights,task=['gene biotype', 'enhancer'],ratio=0.2)
        elif isinstance(dset, GEUData):
            sub_weights = user_defined_weights(dset,sub_weights,task=['proximal tata promoter', 'core tata promoter'],ratio=0.2)
        elif isinstance(dset, HyenaData):
            sub_weights = user_defined_weights(dset,sub_weights,task=['promoter'],ratio=0.15)
        weights += sub_weights
    # 创建带有权重的随机采样器
    sampler = WeightedRandomSampler(weights, num_samples=GenomicDset.__len__(), replacement=True)
    dataloader = GenomicDset.get_DataLoader(batch_size=12, mode='user defined sampler', sampler=sampler, num_workers=2)
    """
# -




# +
class TestData(torch.utils.data.Dataset):
    def __init__(self, 
                 file_name:str="./data/GenomicLLM_GRCh38/20230906_enhancer_res_test.csv", 
                 text_max_length=512, 
                 ids_max_length=512, 
                 tokenizer=None, 
                 num_samples:int=-1
                ):
        super().__init__()
        self.ids_max_length = ids_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        self.SPMprocess = SPMprocess(sp=tokenizer,max_length=ids_max_length,padding=True)
        self.equal_id = self.tokenizer.piece_to_id('=')
        self.eos_id = self.tokenizer.eos_id()
        self.dsetdict = self.read_file(file_name=file_name)
        self.randint_list = list(range(text_max_length))
        self.num_samples = num_samples
        if num_samples>0:
            self.dsetdict = sample_dict( self.dsetdict, num_samples )
            # 重命名key,从0开始
            self.dsetdict = rename_dict_keys_to_serial_numbers( self.dsetdict )
            
        
    def read_file(self, file_name:str):  
        df_data = pd.read_csv(file_name)
        df_data = df_data.reset_index(drop=True)
        dsetdict = df_data.to_dict(orient='index')
            
        return dsetdict
    
    def get_text_from_linedict(self,linedict:dict,max_length:int=512):       
        keys = list(linedict.keys())
        keys.remove('seq')

        if len(linedict['seq'])> max_length:
            rdst = random.choice(self.randint_list[0: len(linedict['seq'])-max_length])
            seq = linedict['seq'][rdst:rdst+max_length]
            text = seq
        else:
            text = linedict['seq']
            
        for key in keys:
            text = text if pd.isna(linedict[key]) else text + f", {key} = {linedict[key]}"
      
        return text
    
    def text2ids(self,text):
        ids = self.SPMprocess(text)
        return ids
    
    def get_loss_mask(self,ids:List[int]):
        """mask最后一个Q=A前的部分"""
        # 寻找最后一个等号
        equal_idx = find_last_index(ids, self.equal_id)
        # 寻找eos位置
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)-1
        # mask：
        # 序列和第5个之后的<pad>取值0
        pad5_idx = min(len(ids)-1, eos_idx+5)
        loss_mask = [0]*(equal_idx+1) + [1]*(pad5_idx-equal_idx) + [0]*(len(ids)-1-pad5_idx)
        return loss_mask
    
    def __len__(self):
        return len(self.dsetdict)
    
    def __getitem__(self, i): 
        text = self.get_text_from_linedict(self.dsetdict[i],self.text_max_length)
        ids = self.text2ids(text)
        # get loss mask
        loss_mask = self.get_loss_mask(ids)
        # item
        item = {'text':text, 'ids':ids, 'mask':loss_mask }
        return item
    
if __name__ == '__main__':
    SPMtokenizer = spm.SentencePieceProcessor(model_file="./model/SPMtokenizer_GenomicLlama_vocabsize512_20231117.model")
    TestDset = TestData(
        file_name="./data/GenomicLLM_GRCh38/20230906_enhancer_res_test.csv",
        tokenizer=SPMtokenizer, 
        text_max_length=1024, 
        ids_max_length=256
    )
    print( TestDset.__len__() )
    print( TestDset.__getitem__(14) )
# -






