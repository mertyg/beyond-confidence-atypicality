import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, models

from .embedding_utils import BasicEmbeddingWrapper
from .download_imbalanced_models import download_and_get_imbalanced


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTopNoSoftmax(nn.Module):
    def __init__(self, original_model):
        super(ResNetTopNoSoftmax, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])
    def forward(self, x):
        x = self.features(x)
        return x

class ResNextBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNextBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.module.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNextTopNoSoftmax(nn.Module):
    def __init__(self, original_model):
        super(ResNextTopNoSoftmax, self).__init__()
        self.features = nn.Sequential(*[list(original_model.module.children())[-1]])
    def forward(self, x):
        x = self.features(x)
        return x


def get_image_classifier(model_name, device="cuda"):
    if model_name == "resnet50":
        imagenet_mean_pxs = np.array([0.485, 0.456, 0.406])
        imagenet_std_pxs = np.array([0.229, 0.224, 0.225])

        imagenet_resnet_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)
        ])
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()
        model = model.to(device)
        model_bottom, model_top = ResNetBottom(model), ResNetTopNoSoftmax(model)
        model_bottom.device = device
        preprocess = imagenet_resnet_transforms
        extractor = BasicEmbeddingWrapper(model_bottom, model_top, model_name)
        extractor.model_name = "ResNet50"
    
    elif model_name == "resnext50_imagenet_lt":
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model = download_and_get_imbalanced(model_name)
        model = model.eval()
        model_bottom, model_top = ResNextBottom(model), ResNextTopNoSoftmax(model)
        model_bottom.device = device
        extractor = BasicEmbeddingWrapper(model_bottom, model_top, model_name)

    elif model_name == "resnet152_places":
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model = download_and_get_imbalanced(model_name)
        model = model.eval()
        model_bottom, model_top = ResNextBottom(model), ResNextTopNoSoftmax(model)
        model_bottom.device = device
        model_top.device = device
        extractor = BasicEmbeddingWrapper(model_bottom, model_top, model_name)
    else:
        raise NotImplementedError("Unknown model name: {}".format(model_name))
    
    return extractor, preprocess
