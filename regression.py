import torch 

# Load the saved dataloader and logits
dataloader = torch.load('dataloader_ViT-B-32-quickgelu.pt', weights_only=False)
total_logits = torch.load('total_logits_ViT-B-32-quickgelu.pt')

selected_classes = torch.load('selected_classes_ViT-B-32-quickgelu.pt', weights_only=False)
print(selected_classes.shape)
print(selected_classes)
print(total_logits.shape)