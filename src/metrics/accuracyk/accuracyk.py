"""Computes the accuracy at k for a set of labels"""

import evaluate
import datasets
import typing

_CITATION = ""

_DESCRIPTION = """\
Computes the accuracy at k for a set of predictions. The accuracy at k is the \
number of instances where the real label is in the set of the k most probable 
classes.
The parameter k is inferred from the shape of the array passed. If you want the accuracy \
at 5 the shape needs to be (N, 5) where N is the number of examples.
"""

# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Args:
    predictions: An array of shape (N, K) where N is the number of examples
                 and K is the desired k (5 for accuracy at 5)
    references: An array of the true labels for the examples
Returns:
    accuracy: the accuracy at k for the inputs
Examples:
    
    >>> accuracyk = evaluate.load("KevinSpaghetti/accuracyk")

    >>> #with numpy arrays
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
    >>> #\{ accuracy: 1 \} # every label is in its row 

    >>> #With lists
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
    >>> #\{ accuracy: 0.6 \} 
    >>> # 3 is in the first row, 
    >>> # 5 is not in the second row,
    >>> # 0 is in the third row
    
    >>> #with numpy for a batch of examples
    >>> k=5
    >>> # get the 5 highest probabilities
    >>> top5_probs = np.argpartition(logits, -k, axis=-1)[:, -k:]
    >>> results = accuracyk.compute(references=top5_probs, predictions=labels)

    >>> # computing the accuracy at 1
    >>> predictions = np.array([ 3, 8, 1 ])
    >>> references = np.array([ 3, 4, 0 ])
    >>> results = accuracyk.compute(predictions=np.expand_dims(predictions, axis=1), references=references)
    >>> print(results)

"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class accuracyk(evaluate.Metric):
    """Computes the accuracy at k for an array of shape (N, k) and correct labels"""

    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value("int64")),
                'references': datasets.Value('int64'),
            }),
            codebase_urls=[],
            reference_urls=[]
        )

    def _download_and_prepare(self, dl_manager):
        ...

    def _compute(self, predictions, references):
        """Returns the accuracy at k"""
        if isinstance(predictions, list):
            accuracyk = sum(
                [reference in kpredictions for kpredictions, reference in zip(predictions, references)]
            ) / len(references)    
        else:
            accuracyk = (
                references[:, None] == predictions[:, :]
            ).any(axis=1).sum() / len(references)
        return dict(accuracy=accuracyk)