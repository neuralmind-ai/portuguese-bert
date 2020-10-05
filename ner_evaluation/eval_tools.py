import contextlib
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from preprocessing import Example, InputSpan


TAG_SEQUENCE = Union[List[int], List[str]]
METRIC_FN = Callable[[List[TAG_SEQUENCE], List[TAG_SEQUENCE]], Any]


def flatten(list_: List[Any]) -> List[Any]:
    """Flattens a nested list of tag predictions."""
    result = []

    for sub in list_:
        if sub and isinstance(sub, list) and isinstance(sub[0], list):
            result.extend(flatten(sub))
        elif isinstance(sub, list):
            result.extend(sub)
        else:
            result.append(sub)

    return result


def confusion_matrix_nested(y_true: List[TAG_SEQUENCE],
                            y_pred: List[TAG_SEQUENCE]) -> str:
    """Shortcut to Sklearn Confusion Matrix accepting nested lists of
    gold labels and predictions instead of flats lists."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(flatten(y_true), flatten(y_pred))


def filtered(metric_fn: METRIC_FN,
             ner_tags,
             **kwargs: Any,
             ) -> METRIC_FN:
    """Wraps a metric function with invalid tag decoding filtering (removal of
    invalid tag predictions for the tag scheme).

    Args:
        metric_fn: a metric function.
        ner_tags: a NERLabelEncoder instance. Used to perform valid tag
            decoding.
        kwargs: extra arguments to be passed to `metric_fn`.
    """
    def metric(y_true: List[TAG_SEQUENCE], y_pred: List[TAG_SEQUENCE]) -> Any:
        y_pred = [ner_tags.decode_valid(y) for y in y_pred]
        return metric_fn(y_true, y_pred, **kwargs)
    return metric


def pad_max_context_array(max_context_mask, max_length=512):
    """Right pad max_context with zeros to the size of prediction_mask"""
    right_pad = max_length - len(max_context_mask)
    max_context_mask = np.pad(max_context_mask, (0, right_pad),
                              mode='constant', constant_values=(0, 0))

    return max_context_mask.astype(np.bool)


def postprocess_span_output(y_pred: TAG_SEQUENCE, span_features: InputSpan):
    """Postprocess the span output to consider only tokens of max context and
    not masked.

    The problem:
    The network is spitting span outputs. An example almost always have
    more than one span, and we have to combine all the spans to get the
    final output.

    Args:
        y_pred(List[int]): predicted class ids for one example span.
        span_features(InputFeatures): features of the span input.
    """

    out_cls_ids = []
    last_token_ix = -1

    # Get output classes skipping subtokens, the first [CLS] and masked tokens
    for tok_ix, cls_id in enumerate(y_pred[1:], start=1):

        is_considered = span_features.input_mask[tok_ix]
        pred_mask = span_features.prediction_mask[tok_ix]
        if is_considered and pred_mask:
            orig_token_ix = span_features.token_to_orig_map[tok_ix]
            is_max_context = span_features.token_is_max_context[tok_ix - 1]

            if orig_token_ix > last_token_ix:
                last_token_ix = orig_token_ix

                if is_max_context:
                    out_cls_ids.append(cls_id)

    return out_cls_ids


class SequentialSpanPostProcessor(object):
    """BERT (without CRF) Span post-processing class.
    This class handles network postprocessing after each batch.
    This class expects that the example order is NOT randomized, i.e., the
    DataLoader uses a SequentialSampler.
    """

    def __init__(self, features: List[InputSpan]):
        self.features = features
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def __call__(self,
                 y_true: TAG_SEQUENCE,
                 y_pred: TAG_SEQUENCE,
                 ) -> Tuple[int, TAG_SEQUENCE, TAG_SEQUENCE]:
        """Performs max-context token selection for a single span."""

        span_features = self.features[self._index]
        y_true = postprocess_span_output(y_true, span_features)
        y_pred = postprocess_span_output(y_pred, span_features)
        self._index += 1

        return span_features.example_index, y_true, y_pred


class CRFSpanPostProcessor(object):
    """Post-processes the output of the BERT-CRF network.

    The CRF layer outputs a list of lists of label ids of variable size.
    Each sequence has a variable length, defined by the feature output mask.
    Besides the prediction mask, we must select only the max context tokens of
    each document span to reconstruct the example text.
    """

    def __init__(self, features: List[InputSpan]):
        self.features = features
        # _index is the example index.
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def __call__(self, y_true: TAG_SEQUENCE, y_pred: TAG_SEQUENCE):
        span_features = self.features[self._index]

        max_context_mask = pad_max_context_array(
            span_features.token_is_max_context,
            len(span_features.input_ids))

        output_mask = np.asarray(span_features.prediction_mask, dtype=np.uint)
        partial_example_mask = max_context_mask[output_mask]

        y_true = [y for y, mask in zip(y_true, partial_example_mask) if mask]
        y_pred = [y for y, mask in zip(y_pred, partial_example_mask) if mask]

        assert len(y_true) == len(y_pred), \
            "y_true and y_pred should be of same length"

        self._index += 1

        return span_features.example_index, y_true, y_pred


class SequenceMetrics(object):
    """Calculates sequence metrics and keeps history of metric values.

    NOTE: Methods `get_best` and `get_best_epoch` assumes a **higher value**
        is better.
    """

    def __init__(self, metrics: List[Tuple[str, METRIC_FN]]):
        self.metrics = {}
        self.history = {}

        for metric_name, metric_fn in metrics:
            self.add_metric(metric_name, metric_fn)

    def add_metric(self, metric_name: str, metric_fn: METRIC_FN) -> None:
        self.metrics[metric_name] = metric_fn
        self.history[metric_name] = []

    def clear_history(self) -> None:
        self.history = {
            k: [] for k in self.history.keys()
        }

    def get_best(self, metric_name: str) -> Any:
        """Returns the maximum value of the given metric by name."""
        return max(self.history[metric_name])

    def get_best_epoch(self, metric_name: str) -> int:
        """Returns the epoch number for which the metric has its highest
        value."""
        return int(np.argmax(self.history[metric_name]) + 1)

    def get_value(self, metric_name: str, epoch: Optional[int] = None) -> Any:
        """Returns the value of a metric at a given epoch (defaults to last
        epoch)."""
        if epoch is None:
            epoch = -1
        else:
            epoch = epoch - 1
        return self.history[metric_name][epoch]

    def calculate_metrics(self,
                          y_true: List[TAG_SEQUENCE],
                          y_pred: List[TAG_SEQUENCE],
                          ) -> Dict[str, Any]:
        """Calculates all registered metrics for the gold and predicted tag
        sequences.

        Args:
            y_true: a list of gold tag sequences.
            y_pred: a list of predicted tag sequences.

        Returns:
            A dict of metric names to calculated metric values.
        """
        values = {}

        for name, metric_fn in self.metrics.items():
            metric_value = metric_fn(y_true, y_pred)
            values[name] = metric_value
            self.history[name].append(metric_value)

        return values


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def write_conll_prediction_file(
        out_file: str,
        examples: List[Example],
        y_preds: List[TAG_SEQUENCE]) -> None:
    """Writes a text output with predictions for a collection of Examples in
    CoNLL evaluation format, one token per line:

    TOKEN GOLD-TAG PRED-TAG

    Distinct example outputs are separated by a blank line.

    Args:
        out_file: the path of the output CoNLL prediction file.
        examples: list of Example instances with associated tokens and gold
            tag labels.
        y_preds: list of predicted tag sequences for each example.

    Raises:
        AssertionError: if (a) the lengths of y_preds and examples are not
            equal, or (b) there is a mismatch in length of tokens, labels or
            predicted tags for any example.
    """
    assert len(y_preds) == len(examples)

    with smart_open(out_file) as fd:
        for example, pred_tag in zip(examples, y_preds):

            tokens = example.doc_tokens
            labels = example.labels

            assert len(tokens) == len(labels)
            assert len(labels) == len(pred_tag)

            for token, label, pred in zip(tokens, labels, pred_tag):
                fd.write('{} {} {}\n'.format(str(token.text), label, pred))

            # Separate examples by line break
            fd.write('\n')


def write_outputs_to_json(out_file: str,
                          examples: List[Example],
                          y_preds: List[TAG_SEQUENCE]) -> None:
    """Writes a JSON with prediction outputs.

    Args:
        out_file: path to an output file or '-' to use stdout.
        examples: list of Example instances with associated tokens.
        y_preds: list of predicted tag sequences for each example.
    """
    output = []
    for example, y_pred in zip(examples, y_preds):
        predicted_entities = []

        for entity in get_entities(y_pred):
            entity_class, start_token_ix, end_token_ix = entity
            start_char = example.doc_tokens[start_token_ix].offset
            end_token = example.doc_tokens[end_token_ix]
            end_char = end_token.offset + len(end_token)

            predicted_entities.append({
                'class': entity_class,
                'start_char': start_char,
                'end_char': end_char,
                'text': example.orig_text[start_char:end_char],
            })
        output.append({
            'doc_id': example.doc_id,
            'text': example.orig_text,
            'entities': predicted_entities,
        })

    with smart_open(out_file) as fd:
        json.dump(output, fd)
