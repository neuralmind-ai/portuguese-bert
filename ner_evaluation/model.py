"""Implementations of BERT, BERT-CRF, BERT-LSTM and BERT-LSTM-CRF models."""

import logging
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Type

import torch
from pytorch_transformers.modeling_bert import (BertConfig,
                                                BertForTokenClassification)
from torchcrf import CRF

LOGGER = logging.getLogger(__name__)


def sum_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Sums the last 4 hidden representations of a sequence output of BERT.
    Args:
    -----
    sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
        For BERT base, the Tuple has length 13.

    Returns:
    --------
    summed_layers: Tensor of shape (batch, seq_length, hidden_size)
    """
    last_layers = sequence_outputs[-4:]
    return torch.stack(last_layers, dim=0).sum(dim=0)


def get_last_layer(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Returns the last tensor of a list of tensors."""
    return sequence_outputs[-1]


def concat_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Concatenate the last 4 tensors of a tuple of tensors."""
    last_layers = sequence_outputs[-4:]
    return torch.cat(last_layers, dim=-1)


POOLERS = {
    'sum': sum_last_4_layers,
    'last': get_last_layer,
    'concat': concat_last_4_layers,
}


def get_model_and_kwargs_for_args(
        args: Namespace,
        training: bool = True,
) -> Tuple[Type[torch.nn.Module], Dict[str, Any]]:
    """Given the parsed arguments, returns the correct model class and model
    args.

    Args:
        args: a Namespace object (from parsed argv command).
        training: if True, sets a high initialization value for classifier bias
            parameter after model initialization.
    """
    bias_O = 6 if training else None
    model_args = {
        'pooler': args.pooler,
        'bias_O': bias_O,
    }

    if args.freeze_bert:
        # Possible models: BERT-LSTM or BERT-LSTM-CRF
        model_args['lstm_layers'] = args.lstm_layers
        model_args['lstm_hidden_size'] = args.lstm_hidden_size
        if args.no_crf:
            model_class = BertLSTM
        else:
            model_class = BertLSTMCRF

    else:
        # Possible models: BertForNERClassification or BertCRF
        if args.no_crf:
            model_class = BertForNERClassification
        else:
            model_class = BertCRF

    return model_class, model_args


class BertForNERClassification(BertForTokenClassification):
    """BERT model for NER task.

    The number of NER tags should be defined in the `BertConfig.num_labels`
    attribute.

    Args:
        config: BertConfig instance to build BERT model.
        weight_O: loss weight value for "O" tags in CrossEntropyLoss.
        bias_O: optional value to initiate the classifier's bias value for "O"
            tag.
        pooler: which pooler configuration to use to pass BERT features to the
            classifier.
    """

    def __init__(self,
                 config: BertConfig,
                 weight_O: float = 0.01,
                 bias_O: Optional[float] = None,
                 pooler='last'):
        super().__init__(config)
        del self.classifier  # Deletes classifier of BertForTokenClassification

        num_labels = config.num_labels

        if pooler not in POOLERS:
            message = ("Invalid pooler: %s. Pooler must be one of %s."
                       % (pooler, list(POOLERS.keys())))
            raise ValueError(message)

        self._build_classifier(config, pooler)
        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights = [1.] * num_labels
        weights[0] = weight_O
        weights = torch.tensor(weights)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

        self.frozen_bert = False
        self.pooler = POOLERS.get(pooler)

    def _build_classifier(self, config, pooler):
        """Build tag classifier."""
        if pooler in ('last', 'sum'):
            self.classifier = torch.nn.Linear(config.hidden_size,
                                              config.num_labels)
        else:
            assert pooler == 'concat'
            self.classifier = torch.nn.Linear(4 * config.hidden_size,
                                              config.num_labels)

    def set_bias_tag_O(self, bias_O: Optional[float] = None):
        """Increase tag "O" bias to produce high probabilities early on and
        reduce instability in early training."""
        if bias_O is not None:
            LOGGER.info('Setting bias of OUT token to %s.', bias_O)
            self.classifier.bias.data[0] = bias_O

    def freeze_bert(self):
        """Freeze all BERT parameters. Only the classifier weights will be
        updated."""
        for p in self.bert.parameters():
            p.requires_grad = False
        self.frozen_bert = True

    def bert_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        """Gets encoded sequence from BERT model and pools the layers accordingly.
        BertModel outputs a tuple whose elements are:
        1- Last encoder layer output. Tensor of shape (B, S, H)
        2- Pooled output of the [CLS] token. Tensor of shape (B, H)
        3- Encoder inputs (embeddings) + all Encoder layers' outputs. This
            requires the flag `output_hidden_states=True` on BertConfig. Returns
            List of tensors of shapes (B, S, H).
        4- Attention results, if `output_attentions=True` in BertConfig.

        This method uses just the 3rd output and pools the layers.
        """
        _, _, all_layers_sequence_outputs, *_ = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        # Use the defined pooler to pool the hidden representation layers
        sequence_output = self.pooler(all_layers_sequence_outputs)

        return sequence_output

    def predict_logits(self, input_ids, token_type_ids=None,
                       attention_mask=None):
        """Returns the logits prediction from BERT + classifier."""
        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq, tags)

        return logits

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the loss.
        Otherwise, it will return the logits and predicted tags tensors.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "y_pred"
          - "loss" (if `labels` is not `None`)
        """
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss

        return outputs


class BertCRF(BertForNERClassification):
    """BERT-CRF model.

    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        del self.loss_fct  # Delete unused CrossEntropyLoss
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` is not `None`, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "loss" (if `labels` is not `None`)
          - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        outputs['logits'] = logits

        # mask: mask padded sequence and also subtokens, because they must
        # not be used in CRF.
        mask = prediction_mask
        batch_size = logits.shape[0]

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        else:
            # Same reasons for iterating
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])

            outputs['y_pred'] = output_tags

        return outputs


class BertLSTM(BertForNERClassification):
    """BERT model with an LSTM model as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        lstm_hidden_size: hidden size of LSTM layers. Defaults to 100.
        lstm_layers: number of LSTM layers. Defaults to 1.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self,
                 config: BertConfig,
                 lstm_hidden_size: int = 100,
                 lstm_layers: int = 1,
                 **kwargs: Any):

        lstm_dropout = 0.2 if lstm_layers > 1 else 0
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        pooler = kwargs.get('pooler', 'last')

        super().__init__(config, **kwargs)

        if pooler in ('last', 'sum'):
            lstm_input_size = config.hidden_size
        else:
            assert pooler == 'concat'
            lstm_input_size = 4 * config.hidden_size

        self.lstm = torch.nn.LSTM(input_size=lstm_input_size,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=lstm_layers,
                                  dropout=lstm_dropout,
                                  batch_first=True,
                                  bidirectional=True)

    def _build_classifier(self, config, pooler):
        """Build label classifier."""
        self.classifier = torch.nn.Linear(2 * self.lstm_hidden_size,
                                          config.num_labels)

    def _pack_bert_encoded_sequence(self, encoded_sequence, attention_mask):
        """Returns a PackedSequence to be used by LSTM.

        The encoded_sequence is the output of BERT, of shape (B, S, H).
        This method sorts the tensor by sequence length using the
        attention_mask along the batch dimension. Then it packs the sorted
        tensor.

        Args:
        -----
        encoded_sequence (tensor): output of BERT. Shape: (B, S, H)
        attention_mask (tensor): Shape: (B, S)

        Returns:
        --------
        sorted_encoded_sequence (tensor): sorted `encoded_sequence`.
        sorted_ixs (tensor): tensor of indices returned by `torch.sort` when
            performing the sort operation. These indices can be used to unsort
            the output of the LSTM.
        """
        seq_lengths = attention_mask.sum(dim=1)   # Shape: (B,)
        sorted_lengths, sort_ixs = torch.sort(seq_lengths, descending=True)

        sorted_encoded_sequence = encoded_sequence[sort_ixs, :, :]

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_encoded_sequence,
            sorted_lengths,
            batch_first=True)

        return packed_sequence, sort_ixs

    def _unpack_lstm_output(self, packed_sequence, sort_ixs):
        """Unpacks and unsorts a sorted PackedSequence that is output by LSTM.

        Args:
            packed_sequence (PackedSequence): output of LSTM. Shape: (B, S, Hl)
            sort_ixs (tensor): the indexes of be used for unsorting. Shape: (B,)

        Returns:
            The unsorted sequence.
        """
        B = len(sort_ixs)

        # Unpack
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence,
                                                             batch_first=True)

        assert unpacked.shape <= (B, 512, 2 * self.lstm.hidden_size)

        # Prepare indices for unsort
        sort_ixs = sort_ixs.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        # (B, S, Hl)
        sort_ixs = sort_ixs.expand(-1, unpacked.shape[1], unpacked.shape[2])
        # Unsort
        unsorted_sequence = (torch.zeros_like(unpacked)
                             .scatter_(0, sort_ixs, unpacked))

        return unsorted_sequence

    def forward_lstm(self, bert_encoded_sequence, attention_mask):
        packed_sequence, sorted_ixs = self._pack_bert_encoded_sequence(
            bert_encoded_sequence, attention_mask)

        packed_lstm_out, _ = self.lstm(packed_sequence)
        lstm_out = self._unpack_lstm_output(packed_lstm_out, sorted_ixs)

        return lstm_out

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, prediction_mask=None):
        """Performs the forward pass of the network.

        Computes the logits, predicted tags and if `labels` is not None, it will
        it will calculate and return the the loss, that is, the negative
        log-likelihood of the batch.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:
            - "logits"
            - "y_pred"
            - "loss" (if `labels` is not `None`)
        """
        outputs = {}

        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask)

        sequence_output = self.dropout(sequence_output)  # (batch, seq, H)

        lstm_out = self.forward_lstm(
            sequence_output, attention_mask)  # (batch, seq, Hl)
        sequence_output = self.dropout(lstm_out)

        logits = self.classifier(sequence_output)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                # Adjust mask and labels to have the same length as logits
                mask = mask[:, :logits.size(1)].contiguous()
                labels = labels[:, :logits.size(1)].contiguous()

                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

            outputs['loss'] = loss

        return outputs


class BertLSTMCRF(BertLSTM):
    """BERT model with an LSTM-CRF as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass (see BertLSTM).
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:

            - "logits"
            - "loss" (if `labels` is not `None`)
            - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask)

        sequence_output = self.dropout(sequence_output)  # (batch, seq, H)

        lstm_out = self.forward_lstm(
            sequence_output, attention_mask)  # (batch, seq, Hl)
        sequence_output = self.dropout(lstm_out)
        logits = self.classifier(sequence_output)
        outputs['logits'] = logits

        mask = prediction_mask  # (B, S)
        # Logits sequence length depends on the inputs:  logits.shape <= (B, S)
        # We have to make the mask and labels the same size.
        mask = mask[:, :logits.size(1)].contiguous()

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            labels = labels[:, :logits.size(1)].contiguous()
            batch_size = input_ids.size(0)
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        else:
            # Same reasons for iterating
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])

            outputs['y_pred'] = output_tags

        return outputs
