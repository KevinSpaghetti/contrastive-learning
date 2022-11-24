---
title: accuracyk
datasets:
-  
tags:
- evaluate
- metric
- accuracy

description: "computes the accuracy at k for a set of predictions as labels"
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
---

# accuracyk

## Metric Description
Computes the accuracy at k for a set of predictions. The accuracy at k is the number of instances where the real label is in the set of the k most probable
classes.
The parameter k is inferred from the shape of the array passed. If you want the accuracy at 5 the shape needs to be (N, 5) where N is the number of examples.

## How to Use
```
predictions = np.array([
    [0, 7, 1, 3, 5],
    [0, 2, 9, 8, 4],
    [8, 4, 0, 1, 3],
])
references = np.array([
    3, 
    5, 
    0
])
results = accuracyk.compute(predictions=predictions, references=references)
# 2/3 of the labels are in the corresponding rows
# the shape of the array predictions is (3, 5) so accuracy at 5 has been computed
# { accuracy: 0.6 } 
```

### Inputs
- **predictions**: An array of shape (N, K) where N is the number of examples and K is the desired k (5 for accuracy at 5)
- **references**: An array of the true labels for the examples

### Output Values
The metric returns outputs between 0 and 1. With 0 being that no value is in its corresponding row and 1 being that every value occurs in its row (higher is better).

### Examples
```python
>>> accuracyk = evaluate.load("KevinSpaghetti/accuracyk")
>>> # with numpy arrays
>>> predictions = np.array([
>>>     [0, 7, 1, 3, 5],
>>>     [0, 2, 9, 8, 4],
>>>     [8, 4, 0, 1, 3],
>>> ])
>>> references = np.array([
>>>     3, 
>>>     4, 
>>>     0
>>> ])
>>> results = accuracyk.compute(predictions=predictions, references=references)
{ accuracy: 1 } # every label is in its row 

>>> # With lists
>>> predictions = [
>>>     [0, 7, 1, 3, 5],
>>>     [0, 2, 9, 8, 4],
>>>     [8, 4, 0, 1, 3],
>>> ]
>>> references = [
>>>     3, 
>>>     5, 
>>>     0
>>> ]
>>> results = accuracyk.compute(predictions=predictions, references=references)
{ accuracy: 0.6 } 
>>> # 3 is in the first row, 
>>> # 5 is not in the second row,
>>> # 0 is in the third row
    
>>> # with numpy for a batch of examples
>>> k=5
>>> # get the 5 highest probabilities
>>> top5_probs = np.argpartition(logits, -k, axis=-1)[:, -k:]
>>> results = accuracyk.compute(references=top5_probs, predictions=labels)

>>> # computing the accuracy at 1
>>> predictions = np.array([ 3, 8, 1 ])
>>> references = np.array([ 3, 4, 0 ])
>>> results = accuracyk.compute(predictions=np.expand_dims(predictions, axis=1), references=references)
```