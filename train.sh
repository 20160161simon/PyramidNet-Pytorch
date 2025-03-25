MASTER_ADDR=localhost MASTER_PORT=29500 python train.py \
    --depth 110 \
    --alpha 120 \
    --data-root-dir /media/aa/HDD1/ \
    --lr 0.05 \
    --batch-size 128 \
    --model-name Pyramid-110-120 \
    --epochs 1200 > 1.out