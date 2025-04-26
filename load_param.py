from open_clip.factory import load_state_dict
import torch


path_param = "./src/logs/20250412-203622-model_RN50-lr_0.0006-b_128-loss_siglip-lexp_False/checkpoints/epoch_32.pt"

dict = load_state_dict(path_param) 
print(torch.exp(dict["logit_scale"]))
print(dict["logit_bias"])


path_param = "./src/logs/20250412-212403-model_RN50-lr_0.0003-b_128-loss_siglip-lexp_False/checkpoints/epoch_32.pt"

dict = load_state_dict(path_param) 
print(torch.exp(dict["logit_scale"]))
print(dict["logit_bias"])