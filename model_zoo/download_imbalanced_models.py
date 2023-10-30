import os, subprocess, torch
import torch.nn as nn
from torchvision import models

urls = {
        "resnext50_imagenet_lt": "https://dl.fbaipublicfiles.com/classifier-balancing/ImageNet_LT/models/resnext50_uniform_e90.pth"
    }

def download_and_get_imbalanced(model_name):
    cache_dir = os.environ.get("CACHE_DIR", "~/.cache")
    
    model_path = os.path.join(cache_dir, f"{model_name}.pth")
    
    if "resnext50" in model_name:
         model = models.resnext50_32x4d(pretrained=False)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        subprocess.call(["wget", "-O", model_path, urls[model_name]])
    ckpt = torch.load(model_path)
    model = nn.DataParallel(model)
    model_ckpt = ckpt["state_dict_best"]["feat_model"]
    for k in ckpt["state_dict_best"]["classifier"]:
        model_ckpt[k] = ckpt["state_dict_best"]["classifier"][k]
    model.load_state_dict(model_ckpt)
    model = model.eval()
    return model