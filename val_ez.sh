python -m open_clip_train.main \
    --imagenet-val "./data/imagenet-1k/val/" \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --vary-clip