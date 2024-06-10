
# def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
#                batch_size=16, workers=4, distributed=False):

#     if 't5' in args.backbone:
#         tokenizer = P5Tokenizer.from_pretrained(
#             args.backbone, 
#             max_length=args.max_text_length, 
#             do_lower_case=args.do_lower_case)

#     if split == 'yelp':
#         from all_yelp_templates import all_tasks as task_templates
        
#         dataset = P5_Yelp_Dataset(
#             task_templates,
#             task_list,
#             tokenizer,
#             args,
#             sample_numbers,
#             mode=mode,
#             split=split,
#             rating_augment=False
#         )
#     else:
#         from all_amazon_templates import all_tasks as task_templates

#         dataset = P5_Amazon_Dataset(
#             task_templates,
#             task_list,
#             tokenizer,
#             args,
#             sample_numbers,
#             mode=mode,
#             split=split,
#             rating_augment=False
#         )

#     if distributed:
#         sampler = DistributedSampler(dataset)
#     else:
#         sampler = None

#     if mode == 'train':
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=(sampler is None),
#             num_workers=workers, pin_memory=True, sampler=sampler,
#             collate_fn=dataset.collate_fn)
#     else:
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             num_workers=workers, pin_memory=True,
#             sampler=sampler,
#             shuffle=None if (sampler is not None) else False,
#             collate_fn=dataset.collate_fn,
#             drop_last=False)
        
#     return loader

# Help me in creating customized dataloader for the new dataset called bookreads

# Now help me in creating P5_Bookreads_Dataset class in the file

from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast


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


    

# create like above 

class P5_Bookreads_Dataset(Dataset):
    
    def __init__(self, task_templates, task_list, tokenizer, args, sample_numbers, mode='train', split='bookreads', rating_augment=False,sample_type = 'random'):
        
        self.task_templates = task_templates
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.mode = mode
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        # data directory is '../../JulianMcAuley/good_reads/goodreads_interactions.pkl'
        
        # print the current working directory
        print(os.getcwd())
        data_dir = Path('../JulianMcAuley/good_reads/goodreads_interactions.pkl')
        if self.mode == 'train':
            self.data = load_pickle(data_dir)['train']
        elif self.mode == 'val':
            self.data = load_pickle(data_dir)['val']
        elif self.mode == 'test':
            self.data = load_pickle(data_dir)['test']
            
        self.total_length = 0
        self.datum_info= []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr =0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                
                self.total_length += len(self.data)*self.sample_numbers[key]
                for i in range(self.total_length-curr):
                    
                    self.datum_info.append((i+curr,key,i//self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError
            
    def gaussian_sampling(self, datum):
        if self.mode =='train':
            if datum['rating'] == 0:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((0+0.4)/2)), std=torch.tensor((0.4-0.0)/4)).item(),1)
            elif datum['rating'] == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((0.5+1.4)/2)), std=torch.tensor((1.4-0.5)/4)).item(),1)
                
            elif datum['rating'] == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((1.5+2.4)/2)), std=torch.tensor((2.4-1.5)/4)).item(),1)
                
            elif datum['rating'] == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((2.5+3.4)/2)), std=torch.tensor((3.4-2.5)/4)).item(),1)
                
            elif datum['rating'] == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((3.5+4.4)/2)), std=torch.tensor((4.4-3.5)/4)).item(),1)
                
            elif datum['rating'] == 5:
                sampled_rating = round(torch.normal(mean=torch.tensor(mean=torch.tensor((4.5+5.0)/2)), std=torch.tensor((5.0-4.5)/4)).item(),1)
                
            if sampled_rating > 5:
                sampled_rating = 5
            
            if sampled_rating < 0:
                sampled_rating = 0
                
            return str(sampled_rating)
        else:
            return int(datum['rating']) 
            
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert idx == datum_info_idx[0]
        
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
        
        if task_name == 'rating':
            datum = self.data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.task_templates[task_name][task_candidates[task_idx]]
            assert task_template['task'] == task_name
            
            if task_template['id']=='1-1':
                source_text = task_template['source'].format(datum['user_id'],datum['book_id'])
                target_text = task_template['target'].format(self.gaussian_sampling(datum))
            else:
                raise NotImplementedError
        
        else:
            raise NotImplementedError
        
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(input_ids) == len(whole_word_ids)
        
        target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        
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
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] # [0] for </s>
    
    def collate_fn(self,batch):
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

def customized_bookreads_dataloader(args, task_list, sample_numbers, split='bookreads', mode='train', 
               batch_size=16, workers=4, distributed=False):
    
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)
        
    if split == 'bookreads':
        from all_bookreads_templates import all_tasks as task_templates
        
        dataset = P5_Bookreads_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
        
    else:
        from all_amazon_templates import all_tasks as task_templates

        dataset = P5_Amazon_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )   
        
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
        
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
        
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader


    