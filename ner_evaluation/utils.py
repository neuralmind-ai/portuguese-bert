import logging
import os
from argparse import Namespace
from typing import Type, Union

import torch
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from model import get_model_and_kwargs_for_args


logger = logging.getLogger(__name__)


def save_model(model: Type[torch.nn.Module], args: Namespace) -> None:
    """Save a trained model and the associated configuration to output dir."""
    model.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


def load_model(args: Namespace,
               model_path: str,
               training: bool = True,
               ) -> torch.nn.Module:
    """Instantiates a pretrained model from parsed argument values.

    Args:
        args: parsed arguments from argv.
        model_path: name of model checkpoint or path to a checkpoint directory.
        training: if True, loads a model with training-specific parameters.
    """

    model_class, model_kwargs = get_model_and_kwargs_for_args(
        args, training=training)
    logger.info('model: {}, kwargs: {}'.format(
        model_class.__name__, model_kwargs))

    cache_dir = os.path.join(
        PYTORCH_PRETRAINED_BERT_CACHE,
        'distributed_{}'.format(args.local_rank))
    model = model_class.from_pretrained(
        model_path,
        num_labels=args.num_labels,
        cache_dir=cache_dir,
        output_hidden_states=True,  # Ensure all hidden states are returned
        **model_kwargs)

    return model


class ExponentialAccumulator:
    """Exponential moving average train loss tracker."""

    def __init__(self, beta: float = 0.99):
        self._accum = None
        self.beta = beta

    def insert_value(self, value: float) -> float:
        if self._accum is None:
            self._accum = value
        else:
            self._accum = self.beta * self._accum + (1 - self.beta) * value

        return self._accum


class RunningAccumulator:
    """Loss value running accumulator."""

    def __init__(self):
        self.total = 0
        self.num_values = 0

    def accumulate(self, value: Union[torch.Tensor, float]):
        if torch.is_tensor(value):
            with torch.no_grad():
                self.total += value.item()
        else:
            self.total += value

        self.num_values += 1

    def mean(self) -> float:
        return self.total / self.num_values
