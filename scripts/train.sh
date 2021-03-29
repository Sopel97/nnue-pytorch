#!/bin/bash

cd ..

if [ ! -d "../nnue-pytorch-training/experiment_$1" ]
then
    mkdir ../nnue-pytorch-training/experiment_$1
    mkdir ../nnue-pytorch-training/experiment_$1/nnue-pytorch

    cp -R . ../nnue-pytorch-training/experiment_$1/nnue-pytorch/
fi

mkdir ../nnue-pytorch-training/experiment_$1/run_$2

python3 train.py \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    --gpus "$2," \
    --threads 1 \
    --num-workers 1 \
    --batch-size 16384 \
    --progress_bar_refresh_rate 20 \
    --smart-fen-skipping \
    --random-fen-skipping 3 \
    --features=HalfKA^ \
    --lambda=1.0 \
    --max_epochs=75 \
    --default_root_dir ../nnue-pytorch-training/experiment_$1/run_$2

python3 train.py \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    --gpus "$2," \
    --threads 1 \
    --num-workers 1 \
    --batch-size 16384 \
    --progress_bar_refresh_rate 20 \
    --smart-fen-skipping \
    --random-fen-skipping 3 \
    --model-features=HalfKA^ \
    --features=HalfKA \
    --lr "0.3e-3" \
    --lambda=1.0 \
    --max_epochs=325 \
    --resume_from_model ../nnue-pytorch-training/experiment_$1/run_$2/default/version_0/checkpoints/last.ckpt \
    --default_root_dir ../nnue-pytorch-training/experiment_$1/run_$2
