import collections
import json
import logging
import os

from argparse import Namespace
from typing import (Dict, List, Optional)
import torch

from tag_encoder import NERTagsEncoder, SCHEMES
from tokenization import (
    Token,
    TokenizerWithAlignment,
    reconstruct_text_from_tokens,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

LOGGER = logging.getLogger(__name__)


NETag = collections.namedtuple("NETag", ['doc_id',
                                         'entity_id',
                                         'text',
                                         'type',
                                         'start_position',
                                         'end_position'])


class Example(object):
    """
    A single training/test example for NER training.
    """

    def __init__(self,
                 doc_id: int,
                 orig_text: str,
                 doc_tokens: List[Token],
                 tags: List[NETag],
                 labels: List[str],
                 ):
        self.doc_id = doc_id
        self.orig_text = orig_text
        self.doc_tokens = doc_tokens
        self.tags = tags
        self.labels = labels

        for token in doc_tokens:
            token._example = self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = ('doc_id: {}\n'
             'orig_text:{}\n'
             'doc_tokens: {}\n'
             'labels: {}\n'
             'tags: {}\n').format(self.doc_id, self.orig_text, self.doc_tokens,
                                  self.labels, self.tags)
        return s


def read_examples(input_file: str,
                  is_training: bool,
                  classes: List[str] = None,
                  scheme: str = 'BIO',
                  ) -> List[Example]:
    """Read a JSON file into a list of Examples.

    The JSON file should contain a list of dictionaries, one dict per input
    document. Each dict should have the following entries:

    doc_id: an example unique identifier (for debugging).
    doc_text: the document text.
    entities: a list of dicts of named entities contained in `doc_text`.
        Each entity dict should have the following entries:

            entity_id: an identifier for the entity (debugging purposes).
            label: the named entity gold label.
            start_offset: start char offset of the entity in `doc_text`.
            end_offset: **exclusive** end char offset of the entity in
                `doc_text`.
            text: the named entity text. It should be equal to the slice of the
                document text using `start_offset` and `end_offset`, e.g.,
                `doc_text[start_offset:end_offset]`.
    """
    scheme = scheme.upper()
    if scheme not in SCHEMES:
        raise ValueError("Invalid tagging scheme `{}`.".format(scheme))

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    tokenizer_with_alignment = TokenizerWithAlignment()

    for document in input_data:
        doc_text = document["doc_text"]
        doc_id = document["doc_id"]

        # Perform whitespace and punctuation tokenization keeping track of char
        # alignment (char_to_word_offset)
        doc_tokens, char_to_word_offset = tokenizer_with_alignment(doc_text)
        labels = ["O"] * len(doc_tokens)
        tags = []

        def set_label(index, tag):
            if labels[index] != 'O':
                LOGGER.warning('Overwriting tag %s at position %s to %s',
                               labels[index], index, tag)
            labels[index] = tag

        if is_training:
            for entity in document["entities"]:
                entity_id = entity["entity_id"]
                entity_text = entity["text"]
                entity_type = entity["label"]
                start_token = None
                end_token = None

                entity_start_offset = entity["start_offset"]
                entity_end_offset = entity["end_offset"]
                start_token = char_to_word_offset[entity_start_offset]
                # end_offset is NOT inclusive to the text, e.g.,
                # entity_text == doc_text[start_offset:end_offset]
                end_token = char_to_word_offset[entity_end_offset - 1]

                assert start_token <= end_token, \
                    "End token cannot come before start token."
                reconstructed_text = reconstruct_text_from_tokens(
                    doc_tokens[start_token:(end_token + 1)])
                assert entity_text.strip() == reconstructed_text, \
                    "Entity text and reconstructed text are not equal: %s != %s" % (
                        entity_text, reconstructed_text)

                if scheme == 'BILUO':
                    # BILUO scheme
                    if start_token == end_token:
                        tag = 'U-' + entity_type
                        set_label(start_token, tag)
                    else:
                        for token_index in range(start_token, end_token + 1):
                            if token_index == start_token:
                                tag = 'B-' + entity_type
                            elif token_index == end_token:
                                tag = 'L-' + entity_type
                            else:
                                tag = 'I-' + entity_type

                            set_label(token_index, tag)

                elif scheme == 'BIO':
                    # BIO scheme
                    for token_index in range(start_token, end_token + 1):
                        if token_index == start_token:
                            tag = 'B-' + entity_type
                        else:
                            tag = 'I-' + entity_type
                        set_label(token_index, tag)

                entity = NETag(
                    doc_id,
                    entity_id,
                    entity_text,
                    entity_type,
                    start_token,
                    end_token,
                )
                tags.append(entity)

        example = Example(
            doc_id=doc_id,
            orig_text=doc_text,
            doc_tokens=doc_tokens,
            tags=tags,
            labels=labels)
        examples.append(example)

    return examples


class InputSpan(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id: int,
                 example_index: int,
                 doc_span_index: int,
                 tokens: List[Token],
                 token_to_orig_map: Dict[int, int],
                 token_is_max_context: List[bool],
                 input_ids: List[int],
                 input_mask: List[int],
                 segment_ids: List[int],
                 prediction_mask: List[bool],
                 labels: Optional[List[str]] = (),
                 label_ids: Optional[List[int]] = (),
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels or []
        self.label_ids = label_ids or []
        self.prediction_mask = prediction_mask

    def __repr__(self):
        return "<Input Features: example {}, span {}>".format(
            self.example_index, self.doc_span_index,
        )

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.tokens)


def _check_is_max_context(doc_spans: List[InputSpan],
                          cur_span_index: int,
                          position: int,
                          ) -> bool:
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a
    # single token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_spans(examples: List[Example],
                              ner_tags_converter: NERTagsEncoder,
                              tokenizer: BertTokenizer,
                              max_seq_length: int,
                              doc_stride: int,
                              is_training: bool,
                              unique_id_start: Optional[int] = None,
                              verbose: bool = True,
                              ) -> List[InputSpan]:
    """Converts examples to BERT input-ready data tensor-like structures,
    splitting large documents into spans of `max_seq_length` using a stride of
    `doc_stride` tokens."""

    unique_id = unique_id_start or 1000000000

    features = []
    for (example_index, example) in enumerate(examples):

        doc_tokens = example.doc_tokens
        doc_labels = example.labels

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_doc_labels = []
        all_prediction_mask = []

        for i, token in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token.text)
            for j, sub_token in enumerate(sub_tokens):
                # Create mapping from subtokens to original token
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                # Mask all subtokens (j > 0)
                all_prediction_mask.append(j == 0)

                if j == 0:
                    label = doc_labels[i]
                    all_doc_labels.append(label)
                else:
                    all_doc_labels.append('X')

        assert len(all_doc_tokens) == len(all_prediction_mask)
        if is_training:
            assert len(all_doc_tokens) == len(all_doc_labels)

        # The -1 accounts for [CLS]. For NER we have only one sentence, so no
        # [SEP] tokens.
        max_tokens_for_doc = max_seq_length - 1

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = []
            segment_ids = []
            labels = None
            label_ids = None
            prediction_mask = []
            # Include [CLS] token
            tokens.append("[CLS]")
            segment_ids.append(0)
            prediction_mask.append(False)

            # Ignore [CLS] label
            if is_training:
                labels = ['X']

            for i in range(doc_span.length):
                # Each doc span will have a dict that indicates if it is the
                # *max_context span* for the tokens inside it
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans,
                                                       doc_span_index,
                                                       split_token_index)
                token_is_max_context.append(is_max_context)
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
                if is_training:
                    labels.append(all_doc_labels[split_token_index])
                prediction_mask.append(
                    all_prediction_mask[split_token_index])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if is_training:
                label_ids = ner_tags_converter.convert_tags_to_ids(labels)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                if is_training:
                    label_ids.append(ner_tags_converter.ignore_index)
                prediction_mask.append(False)

            # If not training, use placeholder labels
            if not is_training:
                labels = ['O'] * len(input_ids)
                label_ids = [ner_tags_converter.ignore_index] * len(input_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(prediction_mask) == max_seq_length
            if is_training:
                assert len(label_ids) == max_seq_length

            if verbose and example_index < 20:
                LOGGER.info("*** Example ***")
                LOGGER.info("unique_id: %s" % (unique_id))
                LOGGER.info("example_index: %s" % (example_index))
                LOGGER.info("doc_span_index: %s" % (doc_span_index))
                LOGGER.info("tokens: %s" % " ".join(tokens))
                LOGGER.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                LOGGER.info("token_is_max_context: %s", token_is_max_context)
                LOGGER.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                LOGGER.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                LOGGER.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                LOGGER.info("prediction_mask: %s" % " ".join([
                    str(x) for x in prediction_mask
                ]))
                if is_training:
                    LOGGER.info(
                        "label_ids: %s" % " ".join([str(x) for x in label_ids]))

                LOGGER.info("tags:")
                inside_label = False
                for tok, lab, lab_id in zip(tokens, labels, label_ids):
                    if lab[0] == "O":
                        if inside_label and tok.startswith("##"):
                            LOGGER.info(f'{tok}\tX')
                        else:
                            inside_label = False
                    else:
                        if lab[0] in ("B", "I", "L", "U") or inside_label:
                            if lab[0] in ("B", "U"):
                                # new entity
                                LOGGER.info('')
                            inside_label = True
                            LOGGER.info(f'{tok}\t{lab}\t{lab_id}')

            features.append(
                InputSpan(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    labels=labels,
                    label_ids=label_ids,
                    prediction_mask=prediction_mask,
                ))
            unique_id += 1

    return features


def get_features_from_examples(examples: List[Example],
                               ner_tags_converter: NERTagsEncoder,
                               tokenizer: BertTokenizer,
                               args: Namespace,  # args from ArgumentParser
                               mode: str,
                               unique_id_start: int = None,
                               verbose: bool = True,
                               ) -> List[InputSpan]:
    """Convert examples to input spans. Read from cache if possible."""

    assert mode in ('train', 'valid', 'eval', 'inference'), "Invalid mode."
    examples_file = getattr(args, mode + '_file') or mode

    cached_features_file = examples_file + '_{0}_{1}_{2}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.doc_stride))

    spans = None
    loaded_from_cache = False

    if os.path.isfile(cached_features_file) and not args.override_cache:
        # Read from cache
        LOGGER.info('Reading cached features from {}'
                    .format(cached_features_file))
        spans = torch.load(cached_features_file)
        loaded_from_cache = True
    else:  # noqa: E772
        LOGGER.info('Converting examples to features.')
        is_training = True if mode in ('train', 'valid', 'eval') else False
        spans = convert_examples_to_spans(
            examples=examples,
            ner_tags_converter=ner_tags_converter,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=is_training,
            unique_id_start=unique_id_start,
            verbose=verbose)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if not loaded_from_cache or args.override_cache:
                LOGGER.info(
                    "  Saving %s features into cached file %s",
                    mode, cached_features_file)
                torch.save(spans, cached_features_file)

    return spans
