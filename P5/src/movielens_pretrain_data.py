from torch.utils.data import Dataset, DataLoader,Sampler
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import torch
import gzip
import random
from multiprocessing import pool
import pickle

import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5ForConditionalGeneration,T5TokenizerFast
from tokenization import P5Tokenizer

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
        

class P5_movielens_Dataset(Dataset):
    
    def __init__(self,task_templates,task_list,tokenizer,args,sample_numbers,mode='train',split='movielens',rating_augment=False,sample_type ='random'):
        self.task_templates = task_templates
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.mode = mode
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        data_dir = Path('../JulianMcAuley/movielens/ml-1m/sequential_recommendation_data.txt')
        self.sequential_data = ReadLineFromFile(data_dir)
        item_count = defaultdict(int)
        user_items = defaultdict()
        
        for line in self.sequential_data:
            user,items = line.strip().split(' ',1)
            items = items.split(' ')
            items =[int(i) for i in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
            self.all_item = list(item_count.keys())
            count = list(item_count.values())
            sum_value = np.sum([x for x in count])
            self.probability = [x/sum_value for x in count]
            self.user_items = user_items
            
            self.total_length =0
            self.datum_info = []
            self.compute_datum_info()
            
    def compute_datum_info(self):
        curr  =0
        for key in list(self.task_list.keys()):
            if key =='sequential':
                if sum([0<int(ind.split('_')[1])<=6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data)*self.sample_numbers[key][0]
                    for i in range(self.total_length-curr):
                        self.datum_info.append((i+curr,key,i//self.sample_numbers[key][0]))
                    curr = self.total_length
                    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self,index):
        out_dict = {}
        out_dict['args']= deepcopy(self.args)
        loss_weight =1.0
        datum_info_idx = self.datum_info[index]
        assert index == datum_info_idx[0]
        
        if len(datum_info_idx) == 3:
            task = datum_info_idx[1]
            task_idx = datum_info_idx[2]
            
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx= datum_info_idx[2]
            task_idx = datum_info_idx[3]
                    
        else:
            raise NotImplementedError
        
        if task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            
            if self.mode =='train':
                end_candidates =[_ for _ in range(max(2,len(sequence)-6),len(sequence)-3)]
                end_index = random.randint(0,len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(11,min(4,end_pos))]
                start_index = random.randint(0,len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
                
            elif self.mode == 'valid':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
                
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
                
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0,len(task_candidates)-1)
            task_template = self.task_templates['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id']=='2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id,' , '.join(purchase_history))
                    
                else:
                    source_text = task_template['source'].format(user_id,' -> '.join(purchase_history))
                    
                target_text = task_template['target'].format(target_item)
                
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        input_ids = self.tokenizer.encode(source_text,padding=True,truncation=True,max_length= self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text,input_ids)
        assert len(input_ids) == len(whole_word_ids)
        
        target_ids = self.tokenizer.encode(target_text,padding=True,truncation=True,max_length=self.args.max_text_length)
        
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['loss_weight'] = loss_weight
        out_dict['input_length'] = len(input_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['source_text'] = source_text
        out_dict['target_text'] = target_text
        out_dict['task'] = task_template['task']
        out_dict['tokenized_text'] = tokenized_text
        
        return out_dict
    
    def calculate_whole_word_ids(self,tokenized_text,input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                whole_word_ids.append(i)
                curr+=1
            else:
                whole_word_ids.append(curr)
                
        return whole_word_ids[:len(input_ids)-1]+[0]
    
    def collate_fn(self, batch):
        
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry
    
    
    
def get_loader(args,task_list,sample_numbers,split='movielens',mode='train',batch_size =16, workers=4, distributed = False):
    
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(args.backbone,max_length=args.max_text_length,do_lower_case=args.do_lower_case)
        
    if split=='movielens':
        
        from all_movielens_templates import all_tasks as task_templates
        dataset = P5_movielens_Dataset(task_templates,task_list,tokenizer,args,sample_numbers,mode=mode,split=split)
        
    else:
        raise NotImplementedError
    
    if distributed:
        sampler = DistributedSampler(dataset)
        
    else:
        sampler = None
        
    if mode=='train':
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=(sampler is None),num_workers=workers,sampler=sampler,collate_fn=dataset.collate_fn)
        
    else:
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,sampler=sampler,collate_fn=dataset.collate_fn)
        
    return loader