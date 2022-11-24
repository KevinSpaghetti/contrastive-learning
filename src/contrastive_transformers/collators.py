import functools
import torch
from torchvision.io import read_image, ImageReadMode

class TextCollator:
    def __init__(self, tokenizer, max_length=32, vocab=None):
        self.tokenizer = tokenizer
        self.max_len = max_length

        self.vocab = vocab

        self.tokenize_fun = functools.partial(self.tokenizer, 
            truncation=True, 
            padding='max_length', max_length=self.max_len,
            return_tensors='pt'
        )

    def __call__(self, examples):
        anchors = [example['anchor'] for example in examples]
        positives = [example['positive'] for example in examples]
        
        tokenized_dict = {
            "anchor": self.tokenize_fun(anchors),
            "positive": self.tokenize_fun(positives),
        }

        if 'negative' in examples:
            negatives = [example['negative'] for example in examples]
            tokenized_dict["negative"] =  self.tokenize_fun(negatives),
        
        if "label" in examples[0]:
            labels = [example['label'] for example in examples]
            if self.vocab:
                # Get the numeric labels from the vocab if a vocab is passed
                tokenized_dict['label']=torch.tensor([self.vocab[lbl] for lbl in labels], dtype=torch.long)
            else:
                # Pass the labels as they are
                tokenized_dict['label']=torch.tensor(labels, dtype=torch.long)
            
        if "label_negative" in examples:
            labels_negative = [example['label_negative'] for example in examples]
            if self.vocab:
                # Get the numeric labels from the vocab if a vocab is passed
                tokenized_dict['label_negative']=torch.tensor([self.vocab[lbl] for lbl in labels_negative], dtype=torch.long)
            else:
                # Pass the labels as they are
                tokenized_dict['label_negative']=torch.tensor(labels_negative, dtype=torch.long)
        
        
        return tokenized_dict    

class ImageCollator:
    def __init__(self, train_transform, vocab=None, augmented_transform=None):
        self.train_transform = train_transform
        self.vocab = vocab

        self.augmented_transform = augmented_transform if augmented_transform is not None else self.train_transform 

    def read_image(self, path: str):
        return read_image(path, ImageReadMode.RGB)
        
    def __call__(self, examples):
        anchors = torch.stack([self.train_transform(self.read_image(example['anchor'])) for example in examples])
        positives = torch.stack([self.augmented_transform(self.read_image(example['positive'])) for example in examples])

        result = {
            "anchor": {'pixel_values': anchors},
            "positive": {'pixel_values': positives}
        }

        if 'negative' in examples[0]:
            negatives = torch.stack([self.augmented_transform(self.read_image(example['negative'])) for example in examples])
            result['negative'] = {'pixel_values': negatives},
        
        if 'label' in examples[0]:
            labels = [example['label'] for example in examples]
            if self.vocab:
                result['label'] = torch.tensor([vocab[lbl] for lbl in labels], dtype=torch.long)
            else:
                result['label'] = torch.tensor([lbl for lbl in labels], dtype=torch.long)
        
        return result