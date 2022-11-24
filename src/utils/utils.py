import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.linalg import norm
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from transformers import TrainerCallback
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy import sparse
from scipy import spatial
import random
import seaborn as sns

import os
import sys

import multiprocessing as mp

from datasets import DatasetDict

import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.utils import logging

from collections import namedtuple

logging.set_verbosity(transformers.logging.ERROR) 
logging.disable_progress_bar() 

SeenUnseenSplit = namedtuple("SeenUnseenSplit", "seen unseen")

def get_seen_unseen_split(train_df, test_df, label_col):
    seen_labels = set(train_df[label_col])    
    seen = test_df.filter(lambda x: x[label_col] in seen_labels)
    unseen = test_df.filter(lambda x: x[label_col] not in seen_labels)
    return SeenUnseenSplit(seen, unseen)

def get_seen(train_df, test_df, label_col):
    seen_pts = set(train_df[label_col].unique())
    return seen_pts

def get_unseen(train_df, test_df, label_col):
    seen_pts = set(train_df[label_col].unique())    
    a = test_df[~test_df[label_col].isin(seen_pts)]
    return test_df[~test_df[label_col].isin(seen_pts)]

class StoreLosses(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.eval_loss = []
        self.top1 = []
        self.top5 = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'loss' in logs.keys():
                self.train_loss.append(logs['loss'])
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero:
            if 'eval_loss' in metrics.keys():
                self.eval_loss.append(metrics['eval_loss'])
                self.top1.append(metrics['eval_top1'])
                self.top5.append(metrics['eval_top5'])

'''
excluded_classes will be the classes that will never be put in the train
dataset (will be unseen)
'''
def train_test_split(dataset, test_size=0.2, excluded_labels=[], label_col='label'):

    labels = dataset[label_col]

    # Indices of a class that is in excluded classes
    forbidden_indices = {i for i in range(0, len(dataset)) if labels[i] in excluded_labels}

    all_indices = set(range(len(dataset)))
    permitted_indices = list(all_indices - forbidden_indices)
    random.shuffle(permitted_indices)
    # Select test from permitted indices
    test_size_index = int(test_size * len(dataset))
    test_indices = list(forbidden_indices) + permitted_indices[0:test_size_index]
    train_indices = permitted_indices[test_size_index:]

    train_split = dataset.select(train_indices)
    test_split = dataset.select(test_indices)

    return DatasetDict({
        'train': train_split,
        'test': test_split
    })        