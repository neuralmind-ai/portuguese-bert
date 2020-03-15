** **This is a work in progress** **

# Portuguese BERT 

This repository contains pre-trained [BERT](https://github.com/google-research/bert) models trained on the Portuguese language. BERT-Base and BERT-Large Cased variants were trained on the [BrWaC (Brazilian Web as Corpus)](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese), a large Portuguese corpus, for 1,000,000 steps, using whole-word mask. Model artifacts for TensorFlow and PyTorch can be found below.

 The models are a result of an ongoing Master's Program. The [text submission for Qualifying Exam](qualifying_exam-portuguese_named_entity_recognition_using_bert_crf.pdf) is also included in the repository in PDF format, which contains more details about the pre-training procedure, vocabulary generation and downstream usage in the task of Named Entity Recognition.

## Download

| Model | TensorFlow checkpoint | PyTorch checkpoint | Vocabulary |
|-|:-------------------------: |:-----------------:|:----------:|
|`bert-base-portuguese-cased`  | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_pytorch_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt) |
|`bert-large-portuguese-cased` | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_tensorflow_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_pytorch_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/vocab.txt) |


## NER Benchmarks

The models were benchmarked on the Named Entity Recognition task and compared to previous published results and [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md). Reported results are for BERT or BERT-CRF architectures, while other results comprise distinct methods.

| Test Dataset | BERT-Large Portuguese | BERT-Base Portuguese | BERT-Base Multilingual | Previous SOTA
|-|:-------------------------: |:-----------------:|:----------:|:-----:|
|MiniHAREM (5 classes)  | **83.30** | 83.03 | 79.44 | 82.26 [[1]](#References), 76.27[[2]](#References)
|MiniHAREM (10 classes) | **78.67** | 77.98 | 74.15 | 74.64 [[1]](#References), 70.33[[2]](#References)

## PyTorch usage example

Our PyTorch artifacts are compatible with the [🤗Huggingface Transformers](https://github.com/huggingface/transformers) library and are also available on the [Community models](https://huggingface.co/models):

    from transformers import AutoModel, AutoTokenizer

    # Using the community model
    # BERT Base
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    # BERT Large
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')
    model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')

    # or, using BertModel and BertTokenizer directly
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('path/to/vocab.txt', do_lower_case=False)
    model = BertModel.from_pretrained('path/to/bert_dir')  # Or other BERT model class


## Acknowledgement

We would like to thank Google for Cloud credits under a research grant that allowed us to train these models.

## References

[1] [Assessing the Impact of Contextual Embeddings for Portuguese Named Entity Recognition](https://github.com/jneto04/ner-pt)

[2] [Portuguese Named Entity Recognition using LSTM-CRF](https://www.researchgate.net/publication/326301193_Portuguese_Named_Entity_Recognition_using_LSTM-CRF)

## How to cite this work

    @article{souza2019portuguese,
        title={Portuguese Named Entity Recognition using BERT-CRF},
        author={Souza, Fabio and Nogueira, Rodrigo and Lotufo, Roberto},
        journal={arXiv preprint arXiv:1909.10649},
        url={http://arxiv.org/abs/1909.10649},
        year={2019}
    }
