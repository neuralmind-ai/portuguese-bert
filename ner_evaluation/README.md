# Code for Named Entity Recognition task

This directory has code to train and evaluate BERT based models on NER task using the HAREM datasets. This package implements 4 architectures divided in two approaches:

**Fine-tuning**:

- BERT-CRF
- BERT

**Feature-based (BERT embeddings)**:

- BERT-LSTM-CRF
- BERT-LSTM

The training and evaluation entry point script is `run_bert_harem.py`. `run_prediction.py` can be used to run inference on new data. All other files are modules. Commands to train and evaluate our BERT models on HAREM datasets are below for each distinct setup: Total and Selective scenarios, feature-based and Fine-tuning approaches, with and without CRF.

## Environment Setup

The code uses a Python 3.6 environment and a GPU is desirable. The following steps use Conda to create a Python virtual environment. Please install Conda before
continuing or create an virtual environment using other tools and skip to step 3.

1 - Create a Python 3.6 virtual environment. With conda:

    $ conda create -n bert_crf python=3.6

2- Activate the environment:

    $ conda activate bert_crf
    # or, for older versions of Conda,
    $ source activate bert_crf

3- Install **PyTorch 1.1.0** using pip or conda (instructions at [PyTorch Get Started guide](https://pytorch.org/get-started/previous-versions/#v110)). Other PyTorch versions were not tested.

4- Install other requirements

    $ pip install -r requirements.txt

## Trained models

Here are two checkpoints of trained NER models on First HAREM dataset.

[BERTimbau Base - BERT-CRF (selective scenario, 5 classes)](https://drive.google.com/file/d/125AMFLsAf33isxTumujUAYBVkoxE7zeT/view?usp=sharing)

[BERTimbau Base - BERT-CRF (total scenario, 10 classes)](https://drive.google.com/file/d/12PE1ypJ949rpatseSV37NXnHwB2Y8jZ5/view?usp=sharing)

## Running inference

The script `run_inference.py` can be used to get predictions for new text using a trained NER model.
Instructions:

1.  Download and extract a trained model checkpoint from above or train your own (instructions below).

2.  Save the inference data in a txt file (for a single input document) or in a JSON file (for multiple documents):

        # inference_text.txt
        Pink Floyd foi uma banda britânica de rock formada em Londres em 1965.
        O grupo foi fundado pelos estudantes Syd Barrett (guitarra, vocal), Nick Mason (bateria), Roger Waters (baixo, voz) e Richard Wright (teclados, voz). Sob a liderança de Barrett, eles lançaram dois singles e um álbum de estreia de sucesso, The Piper at the Gates of Dawn (1967).

        # inference_data.json
        [{"doc_id": 0, "text": "Text of the 1st document"}, {"doc_id": 1, "text": "Text of the 2nd document"}]

3.  Run inference command. This command assumes a downloaded checkpoint that was extracted in `bertimbau-base_bert-crf_total/`. Use the `--help` flag to display extra information about `--output_format` and `--output_file`.

        python run_inference.py \
          --bert_model bertimbau-base_bert-crf_total/ \
          --labels_file bertimbau-base_bert-crf_total/classes.txt \
          --input_file inference_text.txt \
          --output_format json \
          --output_file -

    By default, predictions will be printed in JSON format to stdout:

        [{"doc_id": 0, "text": "Pink Floyd foi uma banda brit\u00e2nica de rock formada em Londres em 1965.\nO grupo foi fundado pelos estudantes Syd Barrett (guitarra, vocal), Nick Mason (bateria), Roger Waters (baixo, voz) e Richard Wright (teclados, voz). Sob a lideran\u00e7a de Barrett, eles lan\u00e7aram dois singles e um \u00e1lbum de estreia de sucesso, The Piper at the Gates of Dawn (1967).", "entities": [{"class": "PESSOA", "start_char": 0, "end_char": 11, "text": "Pink Floyd "}, {"class": "LOCAL", "start_char": 54, "end_char": 62, "text": "Londres "}, {"class": "TEMPO", "start_char": 65, "end_char": 69, "text": "1965"}, {"class": "PESSOA", "start_char": 108, "end_char": 120, "text": "Syd Barrett "}, {"class": "PESSOA", "start_char": 139, "end_char": 150, "text": "Nick Mason "}, {"class": "PESSOA", "start_char": 161, "end_char": 174, "text": "Roger Waters "}, {"class": "PESSOA", "start_char": 189, "end_char": 204, "text": "Richard Wright "}, {"class": "PESSOA", "start_char": 240, "end_char": 247, "text": "Barrett"}, {"class": "OBRA", "start_char": 310, "end_char": 341, "text": "The Piper at the Gates of Dawn "}, {"class": "TEMPO", "start_char": 342, "end_char": 346, "text": "1967"}]}]

## Running NER trainings and evaluations

In all commands below, `{pretrained_bert_model_path}` has to be changed by either a path to BERTimbau Base or Large checkpoint (downloaded from this repository), or the string `bert-base-multilingual-cased` to use mBERT.

In each training (`--do_train`), the model is trained for `--num_train_epochs` epochs using data from `--train_file` and validation is performed using data from `--valid_file`. The final checkpoint saves the model of best epoch in the output directory `--output_dir`. When `--do_eval` is set, a txt file with the
predictions for the test set (`--eval_file` arguments) in CoNLL format will also be saved. See the next section
for the commands to calculate the CoNLL metrics.

When the training ends, some metrics are displayed on the terminal for validation and test sets:

- Micro F1-score
- Precision
- Recall
- Classification Report: metrics per class. The micro avg line displays CoNLL equivalent metrics.

#### Datasets

The `data` directory contains the preprocessed HAREM datasets for both Selective and Total scenarios converted to JSON format. First HAREM is split in train/dev sets and Mini HAREM is used as test set. These JSON files are produced from original HAREM XML files using [this script](https://github.com/fabiocapsouza/harem_preprocessing). Train/dev split is done separately.

#### Important: Multi-GPU and FP16

Running this script in multi-GPU setup **is not recommended**. If the machine has multiple GPUs, limit the GPU visibility setting the `CUDA_VISIBLE_DEVICES` environment variable. Example:

    # Only GPU 0 will be visible
    CUDA_VISIBLE_DEVICES=0 python run_bert_harem.py [...]

FP16 training was not tested and so is also not recommended.

#### Batch size

The commands below set the batch size to 16 considering a BERT Base model and an 8GB GPU. The parameters `per_gpu_train_batch_size` and `gradient_accumulation_steps` can be changed to use less or more available memory and produce the same results, as long as `per_gpu_train_batch_size * gradient_accumulation_steps == 16`.

### Fine-tuning experiments

#### BERT-CRF model

    # Total scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-total.txt \
        --do_train \
        --train_file data/FirstHAREM-total-train.json \
        --valid_file data/FirstHAREM-total-dev.json \
        --num_train_epochs 15 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-total.json \
        --output_dir output_bert-crf_total

    # Selective scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-selective.txt \
        --do_train \
        --train_file data/FirstHAREM-selective-train.json \
        --valid_file data/FirstHAREM-selective-dev.json \
        --num_train_epochs 15 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-selective.json \
        --output_dir output_bert-crf_selective

---

#### BERT model

    # Total scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-total.txt \
        --do_train \
        --train_file data/FirstHAREM-total-train.json \
        --valid_file data/FirstHAREM-total-dev.json \
        --no_crf \
        --num_train_epochs 50 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-total.json \
        --output_dir output_bert_total

    # Selective scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-selective.txt \
        --do_train \
        --train_file data/FirstHAREM-selective-train.json \
        --valid_file data/FirstHAREM-selective-dev.json \
        --no_crf \
        --num_train_epochs 50 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-selective.json \
        --output_dir output_bert_selective

---

### Feature-based experiments

These experiments use the `--freeze_bert` flag to freeze all BERT's parameters and train a LSTM-CRF or LSTM model using BERT embeddings. `--pooler sum` indicates that BERT embeddings will be produced by summing the last 4 layers of BERT instead of using only the last layer.

#### BERT-LSTM-CRF model

    # Total scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-total.txt \
        --do_train \
        --train_file data/FirstHAREM-total-train.json \
        --valid_file data/FirstHAREM-total-dev.json \
        --freeze_bert \
        --pooler sum \
        --num_train_epochs 50 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-total.json \
        --output_dir output_bert-lstm-crf_total

    # Selective scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-selective.txt \
        --do_train \
        --train_file data/FirstHAREM-selective-train.json \
        --valid_file data/FirstHAREM-selective-dev.json \
        --freeze_bert \
        --pooler sum \
        --num_train_epochs 50 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-selective.json \
        --output_dir output_bert-lstm-crf_selective

---

#### BERT-LSTM model

    # Total scenario
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-total.txt \
        --do_train \
        --train_file data/FirstHAREM-total-train.json \
        --valid_file data/FirstHAREM-total-dev.json \
        --freeze_bert \
        --pooler sum \
        --no_crf \
        --num_train_epochs 100 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-total.json \
        --output_dir output_bert-lstm_total

    # Selective
    python run_bert_harem.py \
        --bert_model {pretrained_bert_model_path} \
        --labels_file data/classes-selective.txt \
        --do_train \
        --train_file data/FirstHAREM-selective-train.json \
        --valid_file data/FirstHAREM-selective-dev.json \
        --freeze_bert \
        --pooler sum \
        --no_crf \
        --num_train_epochs 100 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-selective.json \
        --output_dir output_bert-lstm_selective

---

### Computing CoNLL metrics

The [conlleval](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt) script should be used to compute the evaluation metrics using the `predictions_conll.txt` file that is output in the evaluation procedure, as explained below. However,
the package uses the [seqeval library](https://github.com/chakki-works/seqeval) to compute CoNLL equivalent metrics which are printed in the console.

#### Using conlleval

Download the script and make it executable.

    $ chmod +x conlleval.txt

Then, run the command below inputing the corresponding `output_dir` of the trained model

    $ conlleval.txt < {output_dir}/predictions_conll.txt

For example, for BERTimbau-Large-CRF on Total scenario:

    $ ./conlleval.txt < output_bertimbau-large_BERT-CRF_total/predictions_conll.txt
    processed 64853 tokens with 3642 phrases; found: 3523 phrases; correct: 2828.
    accuracy:  96.80%; precision:  80.27%; recall:  77.65%; FB1:  78.94
           ABSTRACCAO: precision:  59.33%; recall:  59.05%; FB1:  59.19  209
        ACONTECIMENTO: precision:  36.51%; recall:  40.35%; FB1:  38.33  63
                COISA: precision:  61.26%; recall:  40.00%; FB1:  48.40  111
                LOCAL: precision:  89.71%; recall:  84.30%; FB1:  86.92  826
                 OBRA: precision:  64.62%; recall:  65.97%; FB1:  65.28  195
          ORGANIZACAO: precision:  71.11%; recall:  75.50%; FB1:  73.24  637
                OUTRO: precision:  50.00%; recall:  14.29%; FB1:  22.22  4
               PESSOA: precision:  86.92%; recall:  83.79%; FB1:  85.33  803
                TEMPO: precision:  94.52%; recall:  90.61%; FB1:  92.52  347
                VALOR: precision:  80.79%; recall:  81.29%; FB1:  81.04  328

### Available hyperparameters

Run `python run_bert_harem.py --help` to display the available hyperparameters. The default values are set to the ones used in our experiments.
