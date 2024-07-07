# optional modelUT: "sdt-v2_8_512, sdt-v1, vanilla_ViT_b_16"
python main.py \
    --batch_size 1 \
    --test_img_dir "/data/dataset/ImageNet/val/n07892512/ILSVRC2012_val_00033598.JPEG" \
    --modelUT "sdt-v2_8_512" \
    --parameterUT "attn_mp" \
    --model metaspikformer_8_512 \
    --data_path /data/dataset/ImageNet \
    --eval \
    --resume ./checkpoint/55M_kd_T4.pth \
    --output_dir ./vis_output
