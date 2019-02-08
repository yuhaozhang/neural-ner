Neural NER models
=====

This repo contains PyTorch implementation of an LSTM-CRF and a CNN-LSTM-CRF model with combined character-level and word-level representations for named entity recognition (NER). 

## Requirements

- Python 3
- PyTorch (>= v1.0.0)
- tqdm

## Training

To train an English NER tagger on the CoNLL03 dataset, please follow the procedures below:

### Preparing data

The repo already contains preprocessed CoNLL03 data in both IOB and IOBES tagging scheme (see `dataset/conll03` folder). However, you need to convert the corresponding data into json-line format to run the training code. To use the IOBES scheme, run:
```bash
python -m prepro.process_data dataset/conll03/iobes dataset/conll_iobes --scheme iobes
```

### Preparing vocabulary and word vectors

Download the 100d GloVe vector file from [here](http://nlp.stanford.edu/data/glove.6B.zip). Copy the file to a folder `dataset/glove/`, and run the following script:
```bash
python prepare_vocab.py dataset/conll_iobes dataset/vocab --all --lower
```
This will create vocab and embedding files in the directory `dataset/vocab`.

### Training

To train a CNN-LSTM-CRF model, run the script:
```bash
bash train_cnn_lstm_crf.sh 0
```
where `0` is the training ID. The model files will then be saved into a folder `saved_models/00/`.

### Evaluation

Run evaluation using a trained model on the test split of the dataset with:
```bash
python eval.py saved_models/00 --dataset testb
```
where `testb` is the name of the test split.

