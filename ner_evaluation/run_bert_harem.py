"""Training and evaluation entry point for HAREM experiments.

This file simply defines a function that loads input data into Example
instances for training/evaluation and defines evaluation metrics for each
dataset split set.

Since `load_and_cache_examples` function below uses
`preprocessing.read_examples` to read the JSON dataset files. See its docstring
for a description of the JSON structure.
"""

import logging
from argparse import Namespace
from typing import List, Tuple

import torch
from pytorch_transformers import BertTokenizer
from seqeval.metrics import (classification_report,
                             f1_score,
                             precision_score,
                             recall_score)
from torch.utils.data import Dataset

from dataset import get_dataset
from eval_tools import confusion_matrix_nested, filtered, SequenceMetrics
from preprocessing import (Example, InputSpan, get_features_from_examples,
                           read_examples)
from tag_encoder import NERTagsEncoder
from trainer import main


logger = logging.getLogger(__name__)


def load_and_cache_examples(
    args: Namespace,
    tokenizer: BertTokenizer,
    tag_encoder: NERTagsEncoder,
    mode: str,
) -> Tuple[Dataset, List[Example], List[InputSpan]]:
    """Preprocesses an input JSON file with raw training/evaluation
    examples and to BERT format according to the provided args (tokenizer,
    tag_encoder/scheme, max sequence length, doc stride, etc)."""
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache.
        # TODO: Verify if this is working as expected.
        torch.distributed.barrier()

    if mode == 'train':
        input_file = args.train_file
    elif mode == 'valid':
        input_file = args.valid_file
    else:
        assert mode == 'eval', f"Invalid mode: {mode}"
        input_file = args.eval_file

    # HAREM dataset specific sanity checks
    # Assert all files use the same scenario (selective or total).
    scenario = 'selective' if 'selective' in input_file else 'total'
    assert scenario in args.labels_file

    examples = read_examples(
        input_file=input_file,
        is_training=True,
        classes=tag_encoder.classes,
        scheme=args.scheme)
    features = get_features_from_examples(
        examples,
        tag_encoder,
        tokenizer,
        args,
        mode=mode,
        unique_id_start=1000000000,
        verbose=args.verbose_logging)

    if mode != 'eval':
        if args.few_samples != -1:
            logger.info('Limiting dataset to %d examples.',
                        args.few_samples)
            examples = examples[:args.few_samples]
            features = list(filter(
                lambda f: f.example_index < args.few_samples, features))
            logger.info('Final features: %d', len(features))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        # TODO: Verify if this is working as expected.
        torch.distributed.barrier()

    dataset = get_dataset(features)

    return dataset, examples, features


def get_train_metrics_fn(tag_encoder) -> SequenceMetrics:
    """Get SequenceMetrics instance for evaluating on the train data."""
    metrics = [
        ('f1_score', f1_score)
    ]
    return SequenceMetrics(metrics)


def get_eval_metrics_fn(tag_encoder) -> SequenceMetrics:
    """Get SequenceMetrics instance for evaluating on the evaluation data.
    """
    metrics = [
        ('f1_score', filtered(f1_score, tag_encoder)),
        ('precision', filtered(
            precision_score, tag_encoder)),
        ('recall', filtered(
            recall_score, tag_encoder)),
        ('classification_report',
            filtered(classification_report, tag_encoder, digits=4)),
        ('confusion_matrix', confusion_matrix_nested),
    ]

    return SequenceMetrics(metrics)


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    main(load_and_cache_examples,
         get_train_metrics_fn=get_train_metrics_fn,
         get_valid_metrics_fn=get_eval_metrics_fn,  # same as evaluation
         get_eval_metrics_fn=get_eval_metrics_fn,
         )
