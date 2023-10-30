import os, torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def split_test(lbls, acts, logits, split=0.2, seed=1):
    indices = np.arange(len(lbls))
    lbls_1, lbls_2, acts_1, acts_2, logits_1, logits_2, train_idx, test_idx = train_test_split(lbls, acts, logits, indices, test_size=1-split, random_state=seed, stratify=lbls)    
    return lbls_1, acts_1, logits_1, train_idx, lbls_2, acts_2, logits_2, test_idx


class BasicEmbeddingWrapper:
    def __init__(self, backbone, model_top, model_name):
        self.backbone = backbone
        self.model_top = model_top
        self.model_name = model_name

    @torch.no_grad()
    def get_outputs(self, loader):
        """Runs and returns model embeddings, labels, and logits for the given dataset."""
        device = self.backbone.device
        activations = []
        all_labels = []
        all_logits = []
        for batch in tqdm(loader):
            if isinstance(batch, dict):
                # Some datasets had the following format
                image =  batch["image"]
                labels = batch["class_idx"]
            else:
                image, labels = batch
            image = image.to(device)
            batch_act = self.backbone(image).view(image.shape[0], -1)
            activations.append(batch_act.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            all_logits.append(self.model_top(batch_act).detach().cpu().numpy())
        
        activations = np.concatenate(activations, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        return activations, all_labels, all_logits

    @torch.no_grad()
    def run_and_cache_outputs(self, dataset, batch_size, output_dir):
        """
        If the experiment files (embeddings, labels, logits) already exist, load them. Otherwise, run the model and cache the outputs.
        """
        
        acts_file = os.path.join(output_dir, f"{self.model_name}_{dataset.dataset_name}_acts.npy")
        lbls_file = os.path.join(output_dir, f"{self.model_name}_{dataset.dataset_name}_lbls.npy")
        logits_file = os.path.join(output_dir, f"{self.model_name}_{dataset.dataset_name}_logits.npy")
        if os.path.exists(logits_file):
            print(f"Found: {logits_file}, loading.")
            acts = np.load(acts_file)
            lbls = np.load(lbls_file)
            logits = np.load(logits_file)
        else:
            print(f"Not found: {logits_file}, extracting.")
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            acts, lbls, logits = self.get_outputs(loader)
            np.save(acts_file, acts)
            np.save(lbls_file, lbls)
            np.save(logits_file, logits)
        return acts, logits, lbls
