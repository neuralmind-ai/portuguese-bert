from collections import defaultdict

import numpy as np
import torch


def select_max_context_tokens(y_pred, prediction_mask, token_is_max_context):
    """Selects y_pred elements masked by prediction_mask &
    token_is_max_context.
    `y_pred` can be the output of any BERT model, and hence does not have a
    fixed expected length nor type.

    Shapes:
    -------
    y_pred: [seq_length] or [sum(prediction_mask)]. Shape depends on whether the
        BERT model has a CRF layer.
    prediction_mask: [seq_length]
    token_is_max_context: Variable length. Ranges from [doc_stride] up to 
        [seg_length - 1].
    """
    # Remove [CLS] token from prediction_mask
    prediction_mask = np.asarray(prediction_mask[1:], dtype=np.bool)
    max_context_mask = np.asarray(token_is_max_context, dtype=np.bool)

    if len(max_context_mask) < len(prediction_mask):
        # Right pad max_context with zeros to the size of prediction_mask
        right_pad = len(prediction_mask) - len(max_context_mask)
        max_context_mask = np.pad(max_context_mask, (0, right_pad),
                                  mode='constant', constant_values=(0, 0))

    # 1st case: y_pred is output of CRF layer
    if isinstance(y_pred, list):
        # y_pred is output of CRF layer (already masked by prediction_mask)
        # So we have to index max_context_mask by prediction_mask
        assert len(y_pred) == sum(prediction_mask)
        out_mask = max_context_mask[prediction_mask]

    else:
        y_pred = y_pred[1:]  # Remove [CLS] token

        if len(y_pred) == len(prediction_mask):
            # 2nd case: output of BERT model
            out_mask = prediction_mask & max_context_mask

        else:
            # y_pred is output of BERT-LSTM, that outputs arrays of variable
            # length (same size as non-masked input, i.e. sum(input_mask).
            # We just need to adjust the masks to have the same length as the
            # output.
            assert prediction_mask[len(y_pred):].sum() == 0
            assert max_context_mask[len(y_pred):].sum() == 0
            prediction_mask = prediction_mask[:len(y_pred)]
            max_context_mask = max_context_mask[:len(y_pred)]

            out_mask = prediction_mask & max_context_mask

    return np.asarray(y_pred)[out_mask]


def concatenate(list_tensors):
    """Concatenates a list of arrays/tensors/list."""

    if isinstance(list_tensors[0], np.ndarray):
        return np.concatenate(list_tensors)

    if isinstance(list_tensors[0], torch.Tensor):
        return torch.cat(list_tensors)

    if isinstance(list_tensors[0], list):
        output = []
        for tensor in list_tensors:
            output.extend(tensor)
        return output

    raise TypeError(f"Received invalid type: {type(list_tensors[0])}")


class MissingPartialOutputError(Exception):
    pass


class OutputComposer:
    """Combines the output of split examples using the max context tokens of
    each span."""

    def __init__(self, examples, features, output_transform_fn=None):
        self.examples = examples
        self.features = features
        self.ix2feature = defaultdict(dict)
        for feat in features:
            self.ix2feature[feat.example_index][feat.doc_span_index] = feat

        self.output_transform_fn = output_transform_fn
        self.reset()

    def reset(self):
        """Clear all partial outputs."""
        self.partial_outputs = {i: {} for i in range(len(self.examples))}

    def insert_partial_output(self, example_ix, doc_span_ix, output):
        """Selects max context tokens from partial output."""
        feature = self.ix2feature[example_ix][doc_span_ix]
        output = select_max_context_tokens(output,
                                           feature.prediction_mask,
                                           feature.token_is_max_context)
        self.partial_outputs[example_ix][doc_span_ix] = output

    def insert_batch(self, example_ixs, doc_span_ixs, batch_output):
        """Insert a batch of partial predictions."""
        for output, example_ix, doc_span_ix in zip(batch_output,
                                                   example_ixs,
                                                   doc_span_ixs):
            self.insert_partial_output(
                example_ix.item(), doc_span_ix.item(), output)

    def get_example_output(self, example_ix):
        """Returns the final output of an example."""
        N_spans = len(self.ix2feature[example_ix])
        try:
            example_partial_outputs = [
                self.partial_outputs[example_ix].get(j, []) for j in range(N_spans)
            ]
        except KeyError as err:
            span_ix = err.args[0]
            msg = (f"Missing partial output for example {example_ix}, span "
                   f"{span_ix}.")
            raise MissingPartialOutputError(msg) from None

        complete_output = concatenate(example_partial_outputs)
        assert len(complete_output) == len(
            self.examples[example_ix].doc_tokens)

        if self.output_transform_fn is not None:
            transformed_output = self.output_transform_fn(complete_output)
            return transformed_output

        return complete_output

    def get_outputs(self):
        """Returns a list of max-context-combined outputs of all examples."""
        outputs = []
        for example_ix in range(len(self.examples)):
            example_output = self.get_example_output(example_ix)
            outputs.append(example_output)

        return outputs
