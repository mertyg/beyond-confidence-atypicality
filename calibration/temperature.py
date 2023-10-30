import torch
import numpy as np
from scipy.special import softmax
from torch import nn, optim
from sklearn.preprocessing import StandardScaler


class TemperatureScaler:
    def __init__(self):
        self.T = None

    def fit(self, calib_logits: np.array, calib_labels: np.array, device="cpu", max_iters=1000, lr=0.1, T_default=1.):

        logits_tensor = torch.tensor(calib_logits).to(device)
        labels_tensor = torch.tensor(calib_labels).long().to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        T = nn.Parameter(torch.Tensor([T_default]).to(device))

        optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iters, line_search_fn='strong_wolfe')
        
        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(logits_tensor/T, labels_tensor)
            loss.backward()
            return loss
        optimizer.step(eval_fn)
        
        self.T = T.detach().item()
        print(f"Platt Fitted! T = {self.T}")
    
    def predict_proba(self, logits: np.array):
        return softmax(logits/self.T, axis=1)    


class GroupPlattScaler:
    def __init__(self, n_groups=4):
        self.platts = {}
        self.quantiles = None
        self.n_groups = n_groups
        
    def get_groups(self, typicality):
        if self.quantiles is None:
            bins = np.linspace(0, 1, self.n_groups+1)[1:]
            self.quantiles = np.quantile(typicality, bins)
            self.quantiles[-1] = np.inf
            
        data_binned_indices = np.digitize(typicality, self.quantiles, right=True)
        return data_binned_indices
    
    def fit(self, logits: np.array, labels: np.array, typicality: np.array):
        groups = self.get_groups(typicality)
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_logits = logits[group_mask]
            group_lbls = labels[group_mask]
            self.platts[group] = TemperatureScaler()
            
            self.platts[group].fit(group_logits, group_lbls)
    
    def predict_proba(self, logits: np.array, typicality: np.array):
        groups = self.get_groups(typicality)
        calibrated_probs = np.zeros_like(logits)
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_logits = logits[group_mask]
            if group in self.platts:
                calibrated_probs[group_mask] = self.platts[group].predict_proba(group_logits)
            else:
                calibrated_probs[group_mask] = softmax(group_logits, axis=1)
        return calibrated_probs


class AtypicalityAwareCalibrator:
    def __init__(self , reg_lambda=0):
        self.T = None
        self.reg_lambda = reg_lambda
        self.ss = StandardScaler()

    def get_params(self, logits, atypicality):
        return (self.a*(atypicality**2) + self.b*(atypicality) + self.d) * logits + self.c
    
    def get_params_np(self, logits, atypicality):
        return (self.a*(atypicality**2) + self.b*(atypicality) + self.d) * logits + self.c
    
    def fit(self, calib_logits: np.array, calib_atypicality: np.array, calib_labels: np.array, device="cpu", max_iters=3000, lr=0.1, T_default=1.):
        calib_atypicality = self.ss.fit_transform(np.min(calib_atypicality, axis=1, keepdims=True))
        
        logits_tensor = torch.tensor(calib_logits).to(device)
        labels_tensor = torch.tensor(calib_labels).long().to(device)
        atypicality_tensor = torch.tensor(calib_atypicality).to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        self.a = nn.Parameter(torch.Tensor([0]).to(device))
        self.b = nn.Parameter(torch.Tensor([0]).to(device))
        self.d = nn.Parameter(torch.Tensor([0]).to(device))
        self.c = nn.Parameter(torch.Tensor([np.ones((calib_logits.shape[1]))*T_default]).to(device))
        optimizer = optim.LBFGS([self.a, self.b, self.c, self.d], lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
        
        def eval_fn():
            optimizer.zero_grad()
            norms = self.a**2 + self.b**2 + torch.sum(self.c**2)
            loss = nll_criterion(self.get_params(logits_tensor, atypicality_tensor), labels_tensor) + self.reg_lambda*norms
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)

        self.a = self.a.detach().cpu().numpy()
        self.b = self.b.detach().cpu().numpy()
        self.c = self.c.detach().cpu().numpy()
        self.d = self.d.detach().cpu().numpy()
        print(f"Parametrs fitted!")

    def predict_proba(self, logits: np.array, atypicality: np.array):
        atypicality_rescaled = self.ss.transform(np.min(atypicality, axis=1, keepdims=True))
        return softmax(self.get_params_np(logits, atypicality_rescaled), axis=1) 
    