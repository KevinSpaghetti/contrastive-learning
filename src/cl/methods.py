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

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy import sparse
from scipy import spatial

import seaborn as sns

import os
import sys

import multiprocessing as mp

import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.utils import logging
from src.utils.utils import batch_encode, batch_tokenize

logging.set_verbosity(transformers.logging.ERROR) 
logging.disable_progress_bar() 

p = os.path.abspath('../../')
sys.path.insert(1, p)

def pick_triplets_with_model_np_cols(model, df, anchor_col_name, positive_col_name,
                                n_examples=1, random=False, 
                                device='cpu', batch_size=128, mat_size=None,
                                disable_output=False):
    
    if mat_size is None:
        mat_size = batch_size
        
    anchor_labels = df[anchor_col_name].astype('category').cat.codes.to_numpy()
    
    if not random:
        tokenizer = model.get_tokenizer()    
    
        anchor_encoded = batch_encode(df[anchor_col_name].tolist(), model, 
                                      batch_size=batch_size, disable_output=disable_output, device=device)
        anchor_embs = np.vstack(anchor_encoded)

        positive_encoded = batch_encode(df[positive_col_name].tolist(), model, 
                                        batch_size=batch_size, disable_output=disable_output, device=device)
        positive_embs = np.vstack(positive_encoded)

    positive_indices = np.zeros((len(df), n_examples), dtype=np.int32)
    negative_indices = np.zeros((len(df), n_examples), dtype=np.int32)
    
    for i in tqdm(range(0, len(df), batch_size), disable=disable_output):

        batch_start = i
        batch_end = min(i+mat_size, len(df))

        if random:
            
            mask = anchor_labels[batch_start:batch_end, None] == anchor_labels[None, :]
        
            for i, mask_row in enumerate(mask):
                pos_ex_idx = np.random.choice(mask_row.nonzero()[0], size=n_examples)
                neg_ex_idx = np.random.choice(np.invert(mask_row).nonzero()[0], size=n_examples)
                
                positive_indices[batch_start + i:batch_end] = pos_ex_idx
                negative_indices[batch_start + i:batch_end] = neg_ex_idx
                
        else:      
            mask = anchor_labels[batch_start:batch_end, None] != anchor_labels[None, :]

            distance_mat = 1.0 - cosine_similarity(anchor_embs[batch_start:batch_end], positive_embs)

            positive_masked = np.ma.masked_array(distance_mat, mask)
            negative_masked = np.ma.masked_array(distance_mat, np.invert(mask))
            
            positive_indices[batch_start:batch_end] = positive_masked.argmax(axis=1, keepdims=True)
            negative_indices[batch_start:batch_end] = negative_masked.argmin(axis=1, keepdims=True)
            
    results = pd.DataFrame({
        'anchor': df[anchor_col_name].tolist() * n_examples,
        'positive': df.iloc[positive_indices.T.reshape(-1), :][positive_col_name].tolist(),
        'negative': df.iloc[negative_indices.T.reshape(-1), :][positive_col_name].tolist(),
        'negative_anchor': df.iloc[negative_indices.T.reshape(-1), :][anchor_col_name].tolist()
    })
            
    return results
    