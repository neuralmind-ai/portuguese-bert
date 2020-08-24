# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file defines the `main` function that handles BERT, BERT-CRF,
BERT-LSTM and BERT-LSTM-CRF training and evaluation on NER task.

The `main` function should be imported and called by another script that passes
functions to 1) load and preprocess input data and 2) define metrics evaluate
the model during training or testing phases.

For further information, see `main` function docstring and the ArgumentParser
arguments.

The code was inspired by Huggingface Tranformers' script for training and
evaluating BERT on SQuAD dataset.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from dataset import get_bert_encoded_dataset
from eval_tools import SequenceMetrics, write_conll_prediction_file
from postprocessing import OutputComposer
from preprocessing import Example, InputSpan
from results_writer import compile_results, write_jsonl_results
from tag_encoder import NERTagsEncoder
from utils import RunningAccumulator, load_model, save_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    args: Namespace,
    train_dataset: Dataset,
    valid_dataset: Optional[Dataset] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Instantiates the train, train evaluation and validation dataloaders (if
    needed)."""
    # Instantiate Dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    train_eval_sampler = SequentialSampler(train_dataset)
    train_eval_dataloader = DataLoader(
        train_dataset,
        sampler=train_eval_sampler,
        batch_size=args.train_batch_size)

    valid_dataloader = None
    if valid_dataset:
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=args.train_batch_size)

    # Logs
    logger.info("  Num examples = %d", len(train_dataset))
    if valid_dataset:
        logger.info("  Num valid examples = %d", len(valid_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        (args.train_batch_size * args.gradient_accumulation_steps *
         (torch.distributed.get_world_size()
          if args.local_rank != -1 else 1)))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)

    return train_dataloader, train_eval_dataloader, valid_dataloader


def prepare_optimizer_and_scheduler(args: Namespace,
                                    model: nn.Module,
                                    num_batches: int,
                                    ) -> Tuple[AdamW, WarmupLinearSchedule]:
    """Configures BERT's AdamW optimizer and WarmupLinearSchedule learning rate
    scheduler. Divides parameters into two learning rate groups, with higher
    learning rate for non-BERT parameters (classifier model)."""
    t_total = (num_batches // args.gradient_accumulation_steps *
               args.num_train_epochs)

    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    logger.info("  Total optimization steps = %d", t_total)

    # Prepare optimizer
    param_optimizer = list(
        filter(lambda p: p[1].requires_grad, model.named_parameters()))

    no_decay = ['bias', 'LayerNorm.weight']
    higher_lr = ['classifier', 'crf', 'lstm']

    def is_classifier_param(param_name: str) -> bool:
        return any(hl in param_name for hl in higher_lr)

    def ignore_in_weight_decay(param_name: str) -> bool:
        return any(nd in param_name for nd in no_decay)

    optimizer_grouped_parameters = [
        {'params': [p for name, p in param_optimizer
                    if not ignore_in_weight_decay(name)
                    and not is_classifier_param(name)],
         'weight_decay': 0.01},
        {'params': [p for name, p in param_optimizer
                    if not ignore_in_weight_decay(name)
                    and is_classifier_param(name)],
         'weight_decay': 0.01,
         'lr': args.classifier_lr},
        {'params': [p for name, p in param_optimizer
                    if ignore_in_weight_decay(name)
                    and not is_classifier_param(name)],
         'weight_decay': 0.0},
    ]

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      correct_bias=False)
    num_warmup_steps = t_total * args.warmup_proportion
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_warmup_steps,
                                     t_total=t_total)

    return optimizer, scheduler


def train(args: Namespace,
          model: torch.nn.Module,
          train_dataset: Dataset,
          train_metrics: SequenceMetrics,
          train_output_composer: OutputComposer,
          valid_dataset: Optional[Dataset] = None,
          valid_metrics: Optional[SequenceMetrics] = None,
          valid_output_composer: Optional[OutputComposer] = None) -> None:
    """Train routine."""

    logger.info("***** Running training *****")

    train_dl, train_eval_dl, valid_dl = prepare_dataloaders(
        args, train_dataset, valid_dataset)

    optimizer, scheduler = prepare_optimizer_and_scheduler(
        args, model, num_batches=len(train_dl))

    # Multi-gpu, distributed and fp16 setup
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            msg = ("Please install apex from "
                   "https://www.github.com/nvidia/apex to use fp16 training.")
            raise ImportError(msg)
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    global_step = 0
    train_losses = []
    if valid_dataset:
        min_val_loss = float('inf')

    # Training loop
    try:
        epoch_tqdm = trange(int(args.num_train_epochs), desc="Epoch")
        loss_accum = RunningAccumulator()
        for epoch in epoch_tqdm:
            model.train()
            stats = {}

            train_tqdm = tqdm(train_dl, desc="Iter")
            for step, batch in enumerate(train_tqdm):
                if args.n_gpu == 1:
                    # multi-gpu does scattering it-self
                    batch = tuple(t.to(args.device) for t in batch)
                # Unpack batch
                input_ids = batch[0]
                input_mask = batch[1]
                segment_ids = batch[2]
                label_ids = batch[3]
                prediction_mask = batch[4]
                # example_ixs = batch[5]
                # doc_span_ixs = batch[6]

                outs = model(input_ids, segment_ids,
                             input_mask, label_ids, prediction_mask)
                loss = outs['loss']
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss_accum.accumulate(loss.item())
                running_mean_loss = loss_accum.mean()
                train_tqdm.set_postfix({'loss': running_mean_loss})

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    clip_grad_norm_(amp.master_params(
                        optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:

                    # Perform gradient clipping
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            clip_grad_norm_(p, 1)

                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

            global_step += 1

            train_losses.append(loss_accum.mean())

            stats['loss'] = format_tqdm_metric(train_losses[-1],
                                               float(min(train_losses)),
                                               fmt='{:.3e}')

            # Evaluate train set
            if epoch % 5 == 0 or epoch == args.num_train_epochs - 1:
                trn_epoch_metrics = evaluate(
                    args,
                    model,
                    tqdm(train_eval_dl, desc="Train metrics"),
                    train_output_composer,
                    train_metrics,
                )

                stats['trn_f1'] = format_tqdm_metric(
                    trn_epoch_metrics['f1_score'],
                    train_metrics.get_best('f1_score'),
                    fmt='{:.2%}')

                epoch_tqdm.set_postfix(stats)
                epoch_tqdm.refresh()

            if valid_dataset:
                # Evaluate validation set
                val_epoch_metrics = evaluate(
                    args,
                    model,
                    tqdm(valid_dl, desc="Validation"),
                    valid_output_composer,
                    valid_metrics,
                )

                # Show metrics on tqdm
                if 'loss' in val_epoch_metrics:
                    epoch_val_loss = val_epoch_metrics['loss']
                    min_val_loss = min(min_val_loss, epoch_val_loss)
                    stats['val_loss'] = format_tqdm_metric(
                        epoch_val_loss, min_val_loss, fmt='{:.3e}')

                stats['val_f1'] = format_tqdm_metric(
                    val_epoch_metrics['f1_score'],
                    valid_metrics.get_best('f1_score'),
                    fmt='{:.2%}')

                best_epoch = valid_metrics.get_best_epoch('f1_score')
                stats['best_epoch'] = best_epoch

                # Save model if best epoch
                if best_epoch == epoch + 1:
                    tqdm.write('Best epoch. Saving model.')
                    save_model(model, args)

            epoch_tqdm.set_postfix(stats)
            epoch_tqdm.refresh()

        # End of training
        if args.valid_file:
            logger.info("  Validation F1 scores: %s",
                        valid_metrics.history['f1_score'])
            best_epoch = valid_metrics.get_best_epoch('f1_score')
            logger.info("  Validation confusion matrix:")
            logger.info("  Epoch %d", best_epoch)
            conf_mat = valid_metrics.get_value("confusion_matrix", best_epoch)
            logger.info("\n" + str(conf_mat))
            logger.info("  Validation classification report:")
            classif_report = valid_metrics.get_value(
                "classification_report", best_epoch)
            logger.info("\n" + str(classif_report))

    except KeyboardInterrupt:
        action = ''
        while action.lower() not in ('y', 'n'):
            action = input(
                '\nInterrupted. Continue execution to save model '
                'weights? [Y/n]')
            if action == 'n':
                sys.exit()

    if not valid_dataset:
        # If not using valid dataset, save model of last epoch
        logger.info('Saving model from last epoch.')
        save_model(model, args)

    if args.results_file:
        # Append this run results
        write_jsonl_results(
            compile_results(args, train_metrics,
                            valid_metrics, train_losses=train_losses),
            args.results_file,
        )


def evaluate(args: Namespace,
             model: nn.Module,
             dataloader: DataLoader,
             output_composer: OutputComposer,
             sequence_metrics: SequenceMetrics,
             reset: bool = True,
             ) -> Dict[str, Any]:
    """Runs a model forward pass on the entire dataloader to compute predictions
    for all examples. Final predictions are gathered in `output_composer`,
    combining the max-context tokens of each forward pass. Returns the
    metrics dict computed by `sequence_metrics.calculate_metrics()`."""
    # Evaluate
    model.eval()

    losses = []
    for step, batch in enumerate(dataloader):
        if args.n_gpu == 1:
            batch = tuple(t.to(args.device) for t in batch)
        # Unpack batch
        input_ids = batch[0]
        input_mask = batch[1]
        segment_ids = batch[2]
        label_ids = batch[3]
        prediction_mask = batch[4]
        example_ixs = batch[5]
        doc_span_ixs = batch[6]

        with torch.no_grad():
            if args.no_crf:
                # BERT or BERT-LSTM
                outs = model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    labels=label_ids,
                    prediction_mask=prediction_mask)
            else:
                # BERT-CRF or BERT-LSTM-CRF.
                # We do not pass `labels` otherwise y_pred is not calculated.
                outs = model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    prediction_mask=prediction_mask)

            y_pred = outs['y_pred']

        output_composer.insert_batch(example_ixs, doc_span_ixs, y_pred)

        loss = outs.get('loss')
        if loss is not None:
            loss = loss.item()
            losses.append(loss)

    y_true = [example.labels for example in output_composer.examples]
    y_pred = output_composer.get_outputs()
    metrics = sequence_metrics.calculate_metrics(y_true, y_pred)

    if losses:
        metrics['loss'] = float(np.mean(losses))

    return metrics


def format_tqdm_metric(value: float, best_value: float, fmt: str) -> str:
    """Formats a value to display in tqdm."""
    if value == best_value:
        return (fmt + '*').format(value)

    return (fmt + ' (' + fmt + '*)').format(value, best_value)


def main(
    load_and_cache_examples_fn: Callable[
        [Namespace, BertTokenizer, NERTagsEncoder, str],
        Tuple[Dataset, List[Example], List[InputSpan]]],
    get_train_metrics_fn: Callable[[NERTagsEncoder], SequenceMetrics],
    get_valid_metrics_fn: Callable[[NERTagsEncoder], SequenceMetrics],
    get_eval_metrics_fn: Callable[[NERTagsEncoder], SequenceMetrics]
):
    """Script entry-point. Performs training and/or evaluation routines.

    This function handles model training and evaluation. All arguments are
    functions that handle 1) training and evaluation data loading and
    preprocessing or 2) defining evaluation metrics. By modifying these
    functions, one can adapt this script to other NER datasets in distinct
    formats.

    Args:
        load_and_cache_examples_fn: a function that handles dataset loading and
            preprocessing. The data should be loaded and converted into
            `preprocessing.Example` instances, that can then be used to
            generate InputSpans and a BERT-ready Dataset.

            This function receives the following inputs:

            args: a Namespace object of parsed CLI arguments with model
                hyperparameters and dataset input files.
            bert_tokenizer: a loaded instance of BertTokenizer.
            tag_encoder: a NERTagsEncoder instance created from the tasks NER
                classes.
            mode: a mode string (train|valid|eval) to select which input file
                to read (args.train_file, args.valid_file or args.eval_file).

        get_train_metrics_fn: a function that receives a NERTagsEncoder and
            returns a SequenceMetrics object to evaluate the model on train
            data during training (`--do_train`).
        get_valid_metrics_fn: a function that receives a NERTagsEncoder and
            returns a SequenceMetrics object to evaluate the model on
            validation data during training (`--do_train`).
        get_eval_metrics_fn: a function that receives a NERTagsEncoder and
            returns a SequenceMetrics object to evaluate the model on test data
            during evaluation (`--do_eval`).
    """
    parser = argparse.ArgumentParser()

    # Model and hyperparameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model name or path to a "
                        "checkpoint directory.")
    parser.add_argument("--tokenizer_model", default=None, type=str,
                        required=False,
                        help="Path to tokenizer files. If empty, defaults to "
                        "--bert_model.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for "
                        "uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after "
                        "WordPiece tokenization. Sequences longer than this "
                        "will be split into multiple spans, and sequences "
                        "shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                        "how much stride to take between chunks.")
    parser.add_argument('--labels_file',
                        required=True,
                        help="File with all NER classes to be considered, one "
                        "per line.")
    parser.add_argument('--scheme',
                        default='bio', help='NER tagging scheme (BIO|BILUO).')
    parser.add_argument('--no_crf',
                        action='store_true',
                        help='Remove the CRF layer (use plain BERT or '
                        'BERT-LSTM).')
    parser.add_argument('--pooler',
                        default='last',
                        help='Pooling strategy for extracting BERT encoded '
                        'features from last BERT layers. '
                        'One of "last", "sum" or "concat".')
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Freeze BERT layers' parameters. If True, uses "
                        "either a BERT-LSTM or BERT-LSTM-CRF model.")
    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=100,
                        help=('Hidden dimension of the LSTM (only used when '
                              'the BERT model is frozen.'))
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=1,
                        help=('Number of LSTM layers (only used when the BERT '
                              'model is frozen.'))
    # General
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints"
                        " and predictions will be written.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data "
                        "processing will be printed.")
    parser.add_argument('--override_cache',
                        action='store_true',
                        help='Override feature caches of input files.')

    # Training related
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_file", default=None,
                        type=str, help="JSON for training.")
    parser.add_argument("--valid_file", default=None, type=str,
                        help="JSON for validating during training.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--classifier_lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate of the classifier and CRF layers.')
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before "
                             "performing a backward/update pass.")
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help="Maximum value of gradient norm on update.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of"
                        " 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling "
                             "value.\n")
    parser.add_argument('--few_samples',
                        type=int, default=-1,
                        help="Turn on few samples for training.")
    parser.add_argument('--results_file',
                        default=None,
                        required=False,
                        help='Optional JSONlines file to log train runs.')

    # Evaluation related
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--eval_file", default=None, type=str,
                        help="JSON for evaluating the model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits "
                "training: {}".format(
                    device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    logger.info("seed: {}, output_dir: {}".format(args.seed, args.output_dir))

    if args.gradient_accumulation_steps < 1:
        message = ("Invalid gradient_accumulation_steps parameter: {}, should "
                   "be >= 1".format(args.gradient_accumulation_steps))
        raise ValueError(message)

    set_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be"
            "True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_eval:
        if not args.eval_file:
            raise ValueError(
                "If `do_eval` is True, then `eval_file` must be "
                "specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer_path = args.tokenizer_model or args.bert_model
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=args.do_lower_case)

    # Instantiate NER Tag encoder
    tag_encoder = NERTagsEncoder.from_labels_file(
        args.labels_file, scheme=args.scheme.upper())

    args.num_labels = tag_encoder.num_labels

    # Load a pretrained model
    model = load_model(args, args.bert_model, training=args.do_train)
    model.to(device)

    train_examples = None
    valid_dataset, valid_examples, valid_features = None, None, None

    # Train
    if args.do_train:
        # Read examples and get features and dataset
        train_dataset, train_examples, train_features = load_and_cache_examples_fn(
            args,
            tokenizer,
            tag_encoder,
            mode='train',
        )

        # Instantiate OutputComposer to post-process train examples
        train_output_comp = OutputComposer(
            train_examples,
            train_features,
            output_transform_fn=tag_encoder.convert_ids_to_tags)

        if args.valid_file:
            logger.info("Reading validation examples.")

            valid_dataset, valid_examples, valid_features = load_and_cache_examples_fn(
                args,
                tokenizer,
                tag_encoder,
                mode='valid',
            )
            # Instantiate OutputComposer to post-process valid examples
            valid_output_comp = OutputComposer(
                valid_examples,
                valid_features,
                output_transform_fn=tag_encoder.convert_ids_to_tags)

        if args.freeze_bert:
            # Freeze BERT layers
            logger.info("Freezing BERT layers.")
            model.freeze_bert()
            assert model.frozen_bert

            logger.info("Creating BERT encoded datasets...")

            train_dataset = get_bert_encoded_dataset(
                model, train_dataset, args.per_gpu_train_batch_size,
                args.device)
            if valid_dataset:
                valid_dataset = get_bert_encoded_dataset(
                    model, valid_dataset, args.per_gpu_train_batch_size,
                    args.device)

        # Initialize Metrics tracker
        train_metrics = get_train_metrics_fn(tag_encoder)

        if args.valid_file:
            valid_metrics = get_valid_metrics_fn(tag_encoder)

        # Training loop
        train(
            args,
            model,
            train_dataset,
            train_metrics=train_metrics,
            train_output_composer=train_output_comp,
            valid_dataset=valid_dataset,
            valid_metrics=valid_metrics,
            valid_output_composer=valid_output_comp,
        )

        # Save tokenizer
        tokenizer.save_pretrained(args.output_dir)

        # Load a trained model and config that you have fine-tuned
        logger.info('Loading best model')
        model = load_model(args, model_path=args.output_dir, training=False)
        model.to(device)

    if args.do_eval and (
            args.local_rank == -1 or torch.distributed.get_rank() == 0):

        logger.info("Reading evaluation examples.")
        eval_dataset, eval_examples, eval_features = load_and_cache_examples_fn(
            args,
            tokenizer,
            tag_encoder,
            mode='eval',
        )
        # Instantiate OutputComposer to post-process eval examples
        eval_output_comp = OutputComposer(
            eval_examples,
            eval_features,
            output_transform_fn=tag_encoder.convert_ids_to_tags)

        logger.info("***** Running evaluation predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.per_gpu_eval_batch_size,
                                     num_workers=os.cpu_count())

        # Define SequenceMetrics that handle the postprocessing
        eval_metrics = get_eval_metrics_fn(tag_encoder)

        model.frozen_bert = False

        metrics = evaluate(
            args,
            model,
            tqdm(eval_dataloader, desc="Evaluation"),
            eval_output_comp,
            eval_metrics,
            reset=False,
        )

        # Display and save test metrics
        metrics_values = []
        for metric_name in ('f1_score', 'precision', 'recall'):
            metric_value = metrics[metric_name]
            metrics_values.append(metric_value)
            logger.info("%s: %s", metric_name, metric_value)

        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as fd:
            fd.write(','.join(map(str, metrics_values)))

        logger.info('Classification report:')
        logger.info('\n%s', metrics['classification_report'])

        conll_file = os.path.join(args.output_dir, 'predictions_conll.txt')
        logger.info('Writing CoNLL style prediction file to %s.', conll_file)

        # Get predictions for all examples
        y_pred = eval_output_comp.get_outputs()
        # Filter invalid predictions
        y_pred_filt = [tag_encoder.decode_valid(preds) for preds in y_pred]

        # Write CoNLL file
        write_conll_prediction_file(conll_file, eval_examples, y_pred_filt)
