#!/bin/bash

docker-compose run --rm python python main.py \
    simple \
    --image-size 256 \
    --batch-size 16 \
    --depth 6 \
    --src-channels 1 \
    --dst-channels 1 \
    --channels 32 \
    --use-sn-g \
    --use-sn-d \
    --norm-g bn \
    --norm-d bn \
    --act-g relu \
    --act-d lrelu \
    --up-mode bilinear \
    --down-mode avg \
    --rgb-output \
    --num-d-layers 3 \
    --init normal \
    --learning-rate-g 0.0002 \
    --learning-rate-d 0.0002 \
    --beta1 0.5 \
    --beta2 0.999 \
    --max-iter -1 \
    --l1-lambda 100 \
    --use-amp \
    --result-folder result \
    --save-interval 500