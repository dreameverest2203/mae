CUDA_VISIBLE_DEVICES=5,6,7,8,9 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 4 \
    --finetune "./pretrain_weights_noise/checkpoint-19.pth" \
    --batch_size 64 \
    --model vit_base_patch16 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path "/data5/chengxuz/Dataset/imagenet_raw/"
