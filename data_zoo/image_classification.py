import os
import subprocess
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                impath = line.split()[0]
                self.img_path.append(os.path.join(root, impath))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

def get_image_dataset(dataset_name, preprocess=None):
    """Careful: This function returns 
    a) train_dataset, test_dataset for balanced classification
    b) train_dataset, val_dataset, test_dataset for imbalanced classification (since earlier papers already proposed used splits for val/test.)
    """
    if dataset_name == "imagenet":
        imagenet_dir = os.environ.get("IMAGENET_DIR", None)
        if imagenet_dir is None:
            raise ValueError("IMAGENET_DIR not set. Please set your environment variable to the path of the ImageNet dataset, or modify here.")
        train_dataset = datasets.ImageNet(root=imagenet_dir, split = 'train', transform=preprocess)
        train_dataset.dataset_name = "imagenet_train"
        test_dataset = datasets.ImageNet(root=imagenet_dir, split = 'val', transform=preprocess)
        test_dataset.dataset_name = "imagenet_test"
        return train_dataset, test_dataset
    
    elif dataset_name == "imagenet_lt":
        imagenet_dir = os.environ.get("IMAGENET_DIR", None)
        if imagenet_dir is None:
            raise ValueError("IMAGENET_DIR not set. Please set your environment variable to the path of the ImageNet dataset, or modify here.")

        cache_dir = os.environ.get("CACHE_DIR", "~/.cache")
        urls = {
            "train": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_train.txt",
            "val": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_val.txt",
            "test": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_test.txt" 
        }
        
        train_txt = os.path.join(cache_dir, "imagenet_lt", "train.txt")
        val_txt = os.path.join(cache_dir, "imagenet_lt", "val.txt")
        test_txt = os.path.join(cache_dir, "imagenet_lt", "test.txt")
        if not os.path.exists(val_txt):
            os.makedirs(os.path.dirname(test_txt), exist_ok=True)
            subprocess.call(["wget", "-O", test_txt, urls["test"]])
            subprocess.call(["wget", "-O", val_txt, urls["val"]])
            subprocess.call(["wget", "-O", train_txt, urls["train"]])
            
        # Needs pointing to the right place, i.e. /imagenet/ILSVRC/Data/CLS-LOC/
        train_dataset = LT_Dataset(root=imagenet_dir, txt=train_txt, transform=preprocess)
        val_dataset = LT_Dataset(root=imagenet_dir, txt=val_txt, transform=preprocess)
        test_dataset = LT_Dataset(root=imagenet_dir, txt=test_txt, transform=preprocess)

        train_dataset.dataset_name = "imagenet_lt_train"
        val_dataset.dataset_name = "imagenet_lt_val"
        test_dataset.dataset_name = "imagenet_lt_test"
        return train_dataset, val_dataset, test_dataset
    
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")