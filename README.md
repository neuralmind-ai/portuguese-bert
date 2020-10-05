
# BERTimbau - Portuguese BERT

This repository contains pre-trained [BERT](https://github.com/google-research/bert) models trained on the Portuguese language. BERT-Base and BERT-Large Cased variants were trained on the [BrWaC (Brazilian Web as Corpus)](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese), a large Portuguese corpus, for 1,000,000 steps, using whole-word mask. Model artifacts for TensorFlow and PyTorch can be found below.

The models are a result of an ongoing Master's Program. The [text submission for Qualifying Exam](qualifying_exam-portuguese_named_entity_recognition_using_bert_crf.pdf) is also included in the repository in PDF format, which contains more details about the pre-training procedure, vocabulary generation and downstream usage in the task of Named Entity Recognition.

## Download

| Model | TensorFlow checkpoint | PyTorch checkpoint | Vocabulary |
|-|:-------------------------: |:-----------------:|:----------:|
| BERTimbau Base (aka `bert-base-portuguese-cased`)  | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_pytorch_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt) |
| BERTimbau Large (aka `bert-large-portuguese-cased`) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_tensorflow_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_pytorch_checkpoint.zip) | [Download](https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/vocab.txt) |

## Evaluation benchmarks

The models were benchmarked on three tasks (Sentence Textual Similarity, Recognizing Textual Entailment and Named Entity Recognition) and compared to previous published results and [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md). Metrics are: Pearson's correlation for STS and F1-score for RTE and NER.

| Task | Test Dataset           | BERTimbau-Large | BERTimbau-Base | mBERT  |      Previous SOTA    |
|:----:|:----------------------:|:---------------:|:-------------: | :-----:| :--------------------:| 
| STS  | ASSIN2                 |    **0.852**    |     0.836      |  0.809 | 0.83 [[1]](#References) |
| RTE  | ASSIN2                 |    **90.0**     |     89.2       |  86.8  | 88.3 [[1]](#References) |
| NER  | MiniHAREM (5 classes)  |    **83.7**     |     83.1       |  79.2  | 82.3 [[2]](#References) |
| NER  | MiniHAREM (10 classes) |    **78.5**     |     77.6       |  73.1  | 74.6 [[2]](#References) |

### NER experiments code

Code and instructions to reproduce the Named Entity Recognition experiments are in [`ner_evaluation/`](ner_evaluation/) directory.


## PyTorch usage example

Our PyTorch artifacts are compatible with the [🤗Huggingface Transformers](https://github.com/huggingface/transformers) library and are also available on the [Community models](https://huggingface.co/models):

- [BERTimbau Base model card](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [BERTimbau Large model card](https://huggingface.co/neuralmind/bert-large-portuguese-cased)

```python
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
```

## Acknowledgement

We would like to thank Google for Cloud credits under a research grant that allowed us to train these models.

## References

[1] [Multilingual Transformer Ensembles for Portuguese Natural Language Task](https://www.researchgate.net/publication/340236502_Multilingual_Transformer_Ensembles_for_Portuguese_Natural_Language_Tasks)

[2] [Assessing the Impact of Contextual Embeddings for Portuguese Named Entity Recognition](https://github.com/jneto04/ner-pt)


## How to cite this work

    @inproceedings{souza2020bertimbau,
        author    = {Souza, F{\'a}bio and Nogueira, Rodrigo and Lotufo, Roberto},
        title     = {{BERT}imbau: pretrained {BERT} models for {B}razilian {P}ortuguese},
        booktitle = {9th Brazilian Conference on Intelligent Systems, {BRACIS}, Rio Grande do Sul, Brazil, October 20-23 (to appear)},
        year      = {2020}
    }

    @article{souza2019portuguese,
        title={Portuguese Named Entity Recognition using BERT-CRF},
        author={Souza, F{\'a}bio and Nogueira, Rodrigo and Lotufo, Roberto},
        journal={arXiv preprint arXiv:1909.10649},
        url={http://arxiv.org/abs/1909.10649},
        year={2019}
    }
