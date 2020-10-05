import logging
import os
import pickle
from typing import List, Tuple

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
)
from tqdm import tqdm

from model import BertForNERClassification
from preprocessing import InputSpan


logger = logging.getLogger(__name__)


def get_dataset(features: List[InputSpan]) -> TensorDataset:
    """Generate a TensorDataset from lists of tensors."""
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_ids for f in features], dtype=torch.long)
    all_prediction_mask = torch.tensor(
        [f.prediction_mask for f in features], dtype=torch.uint8)
    all_example_index = torch.tensor(
        [f.example_index for f in features], dtype=torch.long)
    all_doc_span_index = torch.tensor(
        [f.doc_span_index for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                         all_label_ids, all_prediction_mask,
                         all_example_index, all_doc_span_index)


def get_bert_encoded_features(model: BertForNERClassification,
                              dataset: Dataset,
                              batch_size: int,
                              device: torch.device,
                              ) -> Tuple[torch.Tensor, ...]:
    """Returns a BERT encoded tensors of the dataset, to be used to speed up
    the training of the classifier model with frozen BERT."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_encoded_inputs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting frozen BERT features"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, *_ = batch

            encoded_batch = model.bert_encode(
                input_ids, segment_ids, input_mask)
            encoded_batch = encoded_batch.cpu()
            all_encoded_inputs.append(encoded_batch)

    all_encoded_inputs = torch.cat(all_encoded_inputs, dim=0)

    return (all_encoded_inputs,
            *dataset.tensors[1:])


def get_bert_encoded_dataset(model: BertForNERClassification,
                             dataset: Dataset,
                             batch_size: int,
                             device: torch.device,
                             ) -> TensorDataset:
    """Returns a BERT encoded version of the dataset, to be used to speed up
    the training of the classifier model with frozen BERT."""
    encoded_data = get_bert_encoded_features(
        model, dataset, batch_size, device)

    return TensorDataset(*encoded_data)
