# %%
import sys
sys.path.append('../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
from packaging import version
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# src is in sub folder of another folder(P5) in parent parent directory 



from src.param import parse_args 
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining
from src.customized_pretrain_data import customized_bookreads_dataloader

from torch.utils.data import DataLoader, Dataset
from src.pretrain_data import get_loader

from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all
from all_bookreads_templates import all_tasks as task_templates

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from src.trainer_base import TrainerBase

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path): # This function is used to read lines from a file and return a list of lines
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):  # This function is used to parse the data file into a list of dictionaries for each review text
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

# %%
class DotDict(dict):  # This class is used to convert a dictionary into a class with attributes. This allows for easy access to the dictionary values using the dot notation (e.g. dict.key instead of dict['key'])
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
args = DotDict()

args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = "bookreads"
args.valid = "bookreads"
args.test = "bookreads"
args.batch_size = 11264
args.optim = 'adamw' 
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.losses = 'rating,sequential,explanation,review,traditional'
args.backbone = 't5-small' # small or base
args.output = 'snap/bookreads-small'
args.epoch = 1
args.local_rank = 0

args.comment = ''
args.train_topk = -1
args.valid_topk = -1
args.dropout = 0.1

args.tokenizer = 'p5'
args.max_text_length = 512
args.do_lower_case = False
args.word_mask_rate = 0.15
args.gen_max_length = 64

args.weight_decay = 0.01
args.adam_eps = 1e-6
args.gradient_accumulation_steps = 1

'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

'''
Whole word embedding
'''
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss') # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 0 # Change GPU ID
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

comments = []
dsets = []
if 'toys' in args.train:
    dsets.append('toys')
if 'beauty' in args.train:
    dsets.append('beauty')
if 'sports' in args.train:
    dsets.append('sports')

if 'bookreads' in args.train:
    dsets.append('bookreads')
    
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)

if args.local_rank in [0, -1]:
    print(args)

# %%
def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone
    
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)
    
    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model

# %%
config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone
    
tokenizer = create_tokenizer(args)

model_class = P5Pretraining
model = create_model(model_class, config)

model = model.cuda()

if 'p5' in args.tokenizer:
    model.resize_token_embeddings(tokenizer.vocab_size)
    
model.tokenizer = tokenizer

# I wat to see the model structure
print(model)

# %%
args.load = "../snap/beauty-small.pth"

# Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)

ckpt_path = args.load
load_checkpoint(ckpt_path)

from src.all_amazon_templates import all_tasks as task_templates

# %%
data_splits = load_pickle('../../JulianMcAuley/good_reads/goodreads_interactions.pkl')
train_data = data_splits['train']
valid_data = data_splits['val']
test_data = data_splits['test']


# %%
print(len(train_data))

# %%
print(test_data[0])
# print user_id type
print(type(test_data[0]['user_id']))

# %%
from src.customized_pretrain_data import customized_bookreads_dataloader

# %%
test_task_list = {'rating': ['1-1'] # or '1-6'
}
test_sample_numbers = {'rating': 3, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

# zeroshot_test_loader = get_loader(
#         args,
#         test_task_list,
#         test_sample_numbers,
#         split=args.test, 
#         mode='test', 
#         batch_size=args.batch_size,
#         workers=args.num_workers,
#         distributed=args.distributed
# )

zeroshot_test_loader = customized_bookreads_dataloader(args,test_task_list,test_sample_numbers,split=args.test,mode='test',batch_size=args.batch_size,workers=args.num_workers,distributed=args.distributed)
print(len(zeroshot_test_loader))  # noof datasamples/batch size which is 19850/16 which is  equal to 1241



# %%
gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)): # How to know how much percentage of data is processed 
    model.eval()
    
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
        if i<10:    
                
            print(f"result for batch {i} is {results} and ground truth is {batch['target_text']}")
            
        # save the resukt in /results/bookreads folder save  for every 500 iterations
        if i % 500 == 0:
            save_pickle((gt_ratings, pred_ratings), f'results/bookreads/{i}.pkl')
            
       
            
        
        
        
        
predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if p in [str(i/10.0) for i in list(range(10, 50))]]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))

# %%
torch.cuda.empty_cache()

# %%
print(len(zeroshot_test_loader))

# %%



