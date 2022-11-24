import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
sns.set_theme(color_codes=True)
from PIL import Image
import os
import sys

import transformers
from transformers import AutoModelForImageClassification, AutoTokenizer, AutoConfig, AutoFeatureExtractor
from transformers.utils import logging
from transformers import DefaultDataCollator
import scipy.spatial.distance as distance
from transformers import TrainerCallback

logging.set_verbosity(transformers.logging.ERROR) 
logging.disable_progress_bar() 

p = os.path.abspath('../')
sys.path.insert(1, p)

from torch.utils.data import DataLoader 
from functools import partial 
from torchtext.vocab import build_vocab_from_iterator

from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, Dataset

from torchvision.io import read_image
from torchvision.transforms import RandomResizedCrop, CenterCrop, Compose, Normalize, ToTensor

import evaluate

from transformers import TrainingArguments, Trainer
from src.Loss.ContrastiveLoss import SupConLoss
from src.utils.utils import *
from src.transforms.transforms import Noise

import math
import copy

from collections import defaultdict

import random
import torchvision
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms

from datasets import Image

from src.contrastive_transformers.datasets import AutoAugmentDataset
from src.contrastive_transformers.collators import ImageCollator
from src.contrastive_transformers.trainers import ContrastiveImageTrainer
from src.utils.utils import StoreLosses
from src.wordnet_ontology.wordnet_ontology import WordnetOntology

import os
from datasets import load_dataset

torch.hub.set_dir('./cache')

seed=2783
random.seed(seed) # To sample different immages every epoch
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preparing datasets
mapping_filename = './data/external/imagenet/LOC_synset_mapping.txt'
wn = WordnetOntology(mapping_filename)
vocab = torch.load('./models/vocab.pt')

sketch = load_dataset("imagenet_sketch", split='train', cache_dir='./cache/')

imagenet_classes_folder = './data/external/imagenet/ILSVRC/Data/CLS-LOC/train'

image_labels = [] 
image_paths = []

N_IMAGENET_EXAMPLES = 50
imagenet_classes = sorted(os.listdir(imagenet_classes_folder))
for img_class in imagenet_classes:
    all_imgs = os.listdir(f"{imagenet_classes_folder}/{img_class}/")
    img_names = [random.choice(all_imgs) for _ in range(0, N_IMAGENET_EXAMPLES)]
                              
    image_paths.extend([f"{imagenet_classes_folder}/{img_class}/{name}" for name in img_names])
    image_labels.extend([img_class] * len(img_names))

sketch = load_dataset("imagenet_sketch", split='train', cache_dir='./cache/')
def get_hclass(x):
    _class = wn.class_for_index[x['label']] 
    return { 
        'label': vocab[wn.hypernym(_class)] 
    }

sketch = sketch.map(get_hclass)

n_excluded_classes = int(556 * 0.05)
_classes = list(set(sketch['label']))
excluded_classes = [random.choice(_classes) for i in range(n_excluded_classes)]
dt = train_test_split(sketch, excluded_labels=excluded_classes)
train, test = dt['train'], dt['test']

tr = train.cast_column('image', Image(decode=False))
train_data = pd.concat([
    pd.DataFrame({'image': [p['path'] for p in tr['image']], 'label': tr['label']}), 
    pd.DataFrame({'image': image_paths, 'label': [vocab[wn.hypernym(cl)] for cl in image_labels]})
], axis=0).reset_index(drop=True)

torch.hub.set_dir('../cache')
# Prepare model and data collators
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
train_transforms = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((feature_extractor.size, feature_extractor.size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(180),
    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
)

augmented_transforms = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((feature_extractor.size, feature_extractor.size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(180),
    Noise(p=0.2),
    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
)

train = AutoAugmentDataset(train_data['image'], train_data['label'], return_negative=False)
data_collator = ImageCollator(train_transforms, augmented_transform=augmented_transforms)

cb = StoreLosses()

model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
# Train the model
training_args = TrainingArguments(
    output_dir=f"./models/contrastive-pretraining-{seed}",
    resume_from_checkpoint=True,
    disable_tqdm=False,
    save_strategy='epoch',
    save_total_limit=2,
    num_train_epochs=16,
    learning_rate=2e-4,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    weight_decay=0.01,
    remove_unused_columns=False,
    logging_steps=100,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    fp16=True,
    fp16_opt_level='03',
    report_to="wandb",
    optim="adamw_torch"
)

contrastive_head = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(),
            nn.Linear(768 // 2, 768 // 4),
            nn.ReLU(),
            nn.Linear(768 // 4, 768 // 8),
)

loss = SupConLoss(0.2)
def loss_adapter(anchor_encodings, 
                 positive_encodings, 
                 negative_encodings, 
                 labels, 
                 negative_labels):
    return (
        loss(anchor_encodings, positive_encodings, labels) + 
        loss(positive_encodings, anchor_encodings, labels)
    )

trainer = ContrastiveImageTrainer(
    contrastive_loss=loss_adapter,
    projection_head=contrastive_head,
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    tokenizer=feature_extractor,
    callbacks=[cb]
)

trainer.train()

print(cb.train_loss)
