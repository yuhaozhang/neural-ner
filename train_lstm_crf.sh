#/bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 MODEL_ID OTHER_ARGS"
    exit 1
fi

ID=$1; shift
ARGS=$@
python train.py --id $ID --data_dir dataset/conll_iobes --scheme iobes\
    --vocab_dir dataset/vocab --emb_dim 100 --hidden_dim 100 --num_layers 1 --lower --crf \
    --char_type rnn --char_emb_dim 25 --char_hidden_dim 25 --dropout 0.5 --word_dropout 0.0\
    --optim sgd --lr 0.1 --lr_decay 0.9 --momentum 0.9 --batch_size 10 --num_epoch 100 --decay_epoch 50\
    --info "An LSTM-CRF NER model" $ARGS

