"""This script is an example on how to perform NER inference on plain texts.

Input file must be either a JSON file (that can have multiple documents) or a
txt file with a single document.
"""
import json
import logging
import os
import tempfile
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import torch
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import get_dataset
from eval_tools import (SequenceMetrics, write_conll_prediction_file,
                        write_outputs_to_json)
from postprocessing import OutputComposer
from preprocessing import (Example, InputSpan, get_features_from_examples,
                           read_examples)
from tag_encoder import NERTagsEncoder
from trainer import evaluate
from utils import load_model

logger = logging.getLogger(__name__)


def convert_txt_to_tmp_json_file(txt_file: str) -> str:
    """Converts a txt file with inference content to a JSON file with schema
    expected by read_examples. Returns a filename to the temp JSON file."""
    with open(txt_file) as fd:
        text = fd.read()

    tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json_data = [{"doc_id": 0, "doc_text": text}]

    tmp_file.write(json.dumps(json_data))
    tmp_file.close()

    return tmp_file.name


def load_and_cache_examples(
    input_file: str,
    args: Namespace,
    tokenizer: BertTokenizer,
    tag_encoder: NERTagsEncoder,
    mode: str,
) -> Tuple[Dataset, List[Example], List[InputSpan]]:
    """Preprocesses an input JSON file to generate inference examples and
    convert to BERT format according to the provided args (tokenizer,
    tag_encoder/scheme, max sequence length, doc stride, etc)."""

    examples = read_examples(
        input_file=input_file,
        is_training=False,
        classes=tag_encoder.classes,
        scheme=args.scheme)
    features = get_features_from_examples(
        examples,
        tag_encoder,
        tokenizer,
        args,
        mode=mode,
        unique_id_start=0,
        verbose=args.verbose_logging)

    dataset = get_dataset(features)

    return dataset, examples, features


if __name__ == "__main__":

    parser = ArgumentParser("NER inference CLI")

    # Model and hyperparameters
    parser.add_argument("--input_file",
                        required=True,
                        help="File to load examples for inference (JSON or "
                             "txt).")
    parser.add_argument("--output_file",
                        default='-',
                        help="File to save prediction results. Defaults to "
                             "stdout.")
    parser.add_argument("--output_format",
                        choices=("json", "conll"),
                        default="json",
                        help="Format to save the predictions (json or conll). "
                             "Defaults to json.")

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
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disables CUDA devices for inference.')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='Batch size.')
    parser.add_argument('--verbose_logging', action='store_true')

    args = parser.parse_args()
    args.local_rank = -1

    logging.basicConfig()

    if torch.cuda.is_available and not args.no_cuda:
        args.device = torch.device("cuda")
        args.n_gpu = 1
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    tokenizer_path = args.tokenizer_model or args.bert_model
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=args.do_lower_case)

    # Instantiate NER Tag encoder
    tag_encoder = NERTagsEncoder.from_labels_file(
        args.labels_file, scheme=args.scheme.upper())

    args.num_labels = tag_encoder.num_labels
    args.override_cache = True

    # Load a pretrained model
    model = load_model(args, args.bert_model, training=False)
    model.to(args.device)

    if args.input_file.endswith('.txt'):
        args.inference_file = convert_txt_to_tmp_json_file(args.input_file)
    else:
        args.inference_file = args.input_file

    args.override_cache = True

    dataset, examples, features = load_and_cache_examples(
        args.inference_file,
        args=args,
        tokenizer=tokenizer,
        tag_encoder=tag_encoder,
        mode='inference',
    )

    output_composer = OutputComposer(
        examples,
        features,
        output_transform_fn=tag_encoder.convert_ids_to_tags)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(examples))
    logger.info("  Num split examples = %d", len(features))
    logger.info("  Batch size = %d", args.batch_size)

    # Run prediction for full data
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=os.cpu_count())

    model.frozen_bert = False

    metrics = evaluate(
        args,
        model,
        tqdm(dataloader, desc="Prediction"),
        output_composer=output_composer,
        sequence_metrics=SequenceMetrics([]),  # Empty metrics
        reset=True,
    )

    # Get predictions for all examples
    all_y_pred_raw = output_composer.get_outputs()
    # Filter invalid predictions
    all_y_pred = [tag_encoder.decode_valid(y_pred)
                  for y_pred in all_y_pred_raw]

    # Write predictions to output file
    if args.output_format == 'conll':
        write_conll_prediction_file(args.output_file, examples, all_y_pred)

    elif args.output_format == 'json':
        write_outputs_to_json(args.output_file, examples, all_y_pred)
