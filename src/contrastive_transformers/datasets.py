from datasets import Dataset
import pandas as pd
import random

from collections.abc import Sequence, Iterable, Hashable
from collections import defaultdict

class AutoAugmentDataset(Dataset):
    '''
    A dataset that uses the labels of the classes to construct the 
    (anchor, positive) pairs used in contrastive learning.
    Can also return the (anchor, positive, negative) pair.
    Returns a dict that is the pair with the required keys:
    { 
        'label': ..., 
        'anchor': ..., 
        'positive': ..., 
        'label_negative': ...,
        'negative': ...,  }
    '''

    def __init__(self, examples: Sequence, labels: Sequence, return_negative: bool = True, return_labels: bool = True):
        """ Inits the dataset from a pandas dataframe, positives will be drawn
        from positives with the same anchor value, negatives will be drawn from 
        examples with other anchor values

        Args:
            examples (Sequence): the values
            anchors (Sequence): the example column labels
            return_negative (bool): whether to return negatives in the pair
            return_labels (bool): whether to return the label associated with the positive and negative examples            
        """
       
        self.examples = list(examples)
        self.labels = list(labels)

        self.return_negative = return_negative
        self.return_labels = return_labels

        # to speed up positive search collect indices in examples list
        # speeds up positive and negative sampling 
        self._positive_indices_for_label = defaultdict(list)
        for index, (positive, label) in enumerate(zip(self.examples, self.labels)):
            self._positive_indices_for_label[label].append(index)

    def __len__(self):
        return len(self.examples)

    def get_positive(self, anchor):
        '''Get positive for an anchor'''
        pos_idx = random.choice(self._positive_indices_for_label[anchor])
        return self.examples[pos_idx]

    def get_negative(self, anchor):
        '''Get a negative example for an anchor, also returns the negative_anchor'''
        # Rejection sampling for negative sampling
        neg_idx = random.randrange(0, len(self))
        while neg_idx in self.positive_indices_for_label[anchor]:
            neg_idx = random.randrange(0, len(self))
        
        return (self.labels[neg_idx], self.examples[neg_idx]) 

    def __getitem__(self, idx):
        anchor, label = self.examples[idx], self.labels[idx]
        
        result = {
            'anchor': anchor,
            'positive': self.get_positive(label)
        }
        
        if self.return_labels: result['label'] = label

        if not self.return_negative:
            return result

        neg_label, neg_example = self.get_negative(anchor)

        result['negative'] = neg_example
        if self.return_labels: result['label_negative'] = neg_label

        return result

class DatasetWithPositives(Dataset):

    def __init__(self, examples: Sequence, labels: Sequence, positives: Sequence, return_labels: bool = True):
 
        self.examples = list(examples)
        self.positives = list(positives)
        self.labels = list(labels)

        self.return_labels = return_labels

        # to speed up positive search collect indices in examples list
        # speeds up positive and negative sampling 
        self.positives_for_example = defaultdict(list)
        for index, (example, positive) in enumerate(zip(self.examples, self.positives)):
            self.positives_for_example[example].append(positive)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        anchor, label = self.examples[idx], self.labels[idx]
        positive = random.choice(self.positives_for_example[anchor])
        
        result = {
            'anchor': anchor,
            'positive': positive
        }
        
        if self.return_labels: result['label'] = label
        return result

