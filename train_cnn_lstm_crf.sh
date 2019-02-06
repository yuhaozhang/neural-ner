#/bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 MODEL_ID OTHER_ARGS"
    exit 1
fi

ID=$1; shift
ARGS=$@
python train.py --id $ID --vocab_dir dataset/vocab_all_100d --emb_dim 100 --hidden_dim 200 --num_layers 1 --lower --crf \
    --char_type cnn --char_emb_dim 30 --char_fmin 3 --char_fmax 3 --char_fsize 30 --char_cnn_dim -1 \
    --optim sgd --lr 0.015 --lr_decay 0.05 --momentum 0.9 --batch_size 10 --num_epoch 150 --decay_epoch 100 \
    --info "Train a Ma et al. NER model" $ARGS
    #--always_decay --optim sgd --lr 0.015 --lr_decay 0.05 --momentum 0.9 --batch_size 10 \

