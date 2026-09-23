python CKAA_master/train_eval.py -d imagenet_a \
    -sf 10s_imagenet_a \
    -m vit_base_patch16_224.augreg_in21k\
    -b 110 --temperature 28.0\
    -tf 0.05 -kg 20 -tg 0.2 -tc 3.0 -kc 20 \
    --null_eta1 0.95 --null_eta2 0.95 \
    --seed 2024 $@
