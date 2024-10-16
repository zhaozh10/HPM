python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 --node_rank 0 \
    main_pretrain.py \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_base_patch16_dec512d8b \
    --input_size 224 \
    --token_size 14 \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-3 --weight_decay 0.05 \
    --output_dir  ./output_dir/pretrain \
    --log_dir   ./log_dir/pretrain \
    --experiment hpm_in1k_ep800 \
    --learning_loss --relative