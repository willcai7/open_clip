python -m open_clip_train.main \
    --imagenet-val="./data/imagenet-1k/val/" \
    --model  ViT-H-14-378-quickgelu \
    --pretrained dfn5b \
    --save-logits