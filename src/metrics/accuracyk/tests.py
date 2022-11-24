import evaluate
import numpy as np

accuracyk = evaluate.load("./accuracyk.py", use_auth_token=True)

predictions = np.array([
    [0, 7, 1, 3, 5],
    [0, 2, 9, 8, 4],
    [8, 4, 0, 1, 3],
])

references = np.array([
    3, 
    4, 
    0
])
results = accuracyk.compute(predictions=predictions, references=references)
print(results)

predictions = [
    [0, 7, 1, 3, 5],
    [0, 2, 9, 8, 4],
    [8, 4, 0, 1, 3],
]
references = [3, 5, 0]
results = accuracyk.compute(predictions=predictions, references=references)

predictions = np.array([ 3, 8, 1 ])
references = np.array([ 3, 4, 0 ])

results = accuracyk.compute(predictions=np.expand_dims(predictions, axis=1), references=references)
print(results)

