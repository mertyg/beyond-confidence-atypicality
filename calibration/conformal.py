import numpy as np

class RAPS:
    """
    Code is taken from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    and modified. The code has the MIT License.
    """
    def __init__(self, alpha=0.1, lam_reg=0.01, k_reg=None, disallow_zero_sets=True, rand=False, aa_sub=False):
        self.alpha = alpha # 1-alpha is the desired coverage
        # Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
        self.lam_reg = lam_reg
        self.k_reg = k_reg
        self.disallow_zero_sets = disallow_zero_sets # Set this to False in order to see the coverage upper bound hold
        self.rand = rand # Set this to True in order to see the coverage upper bound hold
        self.qhat = None
        self.name = f"APS@{alpha}" if lam_reg == 0 else f"RAPS@{alpha}_lam={lam_reg}_k={k_reg}_{disallow_zero_sets}_{rand}"
        self.aa_sub = aa_sub

    def fit(self, probs, labels):
        if self.k_reg is None:
            paramtune_idx = np.random.choice(probs.shape[0], 10000)
            self.k_reg = self.fit_kreg(probs[paramtune_idx], labels[paramtune_idx])
            print("Fitted k: ", self.k_reg)
            self.lam_reg = self.fit_lambda(probs[paramtune_idx], labels[paramtune_idx])
            print("Fitted lam: ", self.lam_reg)

        n = probs.shape[0]
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        cal_pi = probs.argsort(1)[:,::-1]
        
        cal_srt = np.take_along_axis(probs,cal_pi,axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == labels[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
        # Get the score quantile
        q = 1-self.alpha if self.aa_sub else np.ceil((n+1)*(1-self.alpha))/n
        qhat = np.quantile(cal_scores, q, interpolation='higher')
        self.qhat = qhat
    
    def get_sets(self, probs):
        n_val = probs.shape[0]
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        val_pi = probs.argsort(1)[:,::-1]
        val_srt = np.take_along_axis(probs, val_pi,axis=1)
        val_srt_reg = val_srt + reg_vec
        val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
        indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= self.qhat if self.rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= self.qhat
        if self.disallow_zero_sets: 
            indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
        return prediction_sets
    
    def fit_kreg(self, probs, labels):
        gt_locs_kstar = np.array([np.where(np.argsort(-probs[i]) == labels[i])[0][0] for i in range(probs.shape[0])])
        kstar = np.quantile(gt_locs_kstar, 1-self.alpha, interpolation='higher') + 1
        return kstar 
    
    def fit_lambda(self, probs, labels):
        lamda_star = 0
        best_size = probs.shape[1]
        for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 10.0]: # predefined grid, change if more precision desired.
            temp_raps = RAPS(alpha=self.alpha, lam_reg=temp_lam, k_reg=self.k_reg, 
                             disallow_zero_sets=self.disallow_zero_sets, rand=self.rand)
            temp_raps.fit(probs, labels)
            sets = temp_raps.get_sets(probs)
            sz_avg = sets.sum(axis=1).mean(axis=0)
            print(temp_lam, sz_avg)
            if sz_avg < best_size:
                best_size = sz_avg
                lamda_star = temp_lam
        return lamda_star


class AAPS():
    def __init__(self, subroutine_fn, n_groups=8, alpha=0.1, rand=False, allow_zero_sets=False, k_reg=5, lam_reg=0): 
        self.n_groups = n_groups
        self.alpha = alpha
        self.aqs = None
        self.cqs = None
        self.rand = rand
        self.allow_zero_sets = allow_zero_sets
        self.k_reg = k_reg
        self.lam_reg = lam_reg
        self.subroutines = {}
        self.subroutine_fn = subroutine_fn
        sub = self.subroutine_fn()
        self.name = f"AA-{sub.name}"
        
    def fit(self, probs, labels, atypicality):
        quantiles_x = np.linspace(0.0, 1.0, self.n_groups+1)
        quantiles_y = np.linspace(0.0, 1.0, self.n_groups+1)
        confidence = np.max(probs, axis=1)
        atypicality_min = np.min(atypicality, axis=1)
        
        
        self.aqs = np.quantile(atypicality_min, q=quantiles_x)
        self.aqs[0] = -np.inf
        self.aqs[-1] = np.inf
        self.cqs = np.quantile(confidence, q=quantiles_y)
        self.cqs[0] = -np.inf
        self.cqs[-1] = np.inf
        
        for i in range(self.n_groups):
            self.subroutines[i] = {}
            atypicality_mask = (atypicality_min <= self.aqs[i+1]) & (atypicality_min > self.aqs[i])
            for j in range(self.n_groups):   
                confidence_mask = (confidence <= self.cqs[j+1]) & (confidence > self.cqs[j])
                mask = confidence_mask & atypicality_mask
                probs_masked = probs[mask]
                labels_masked = labels[mask]
                self.subroutines[i][j] = self.subroutine_fn()
                self.subroutines[i][j].fit(probs_masked, labels_masked)

        
    def get_sets(self, probs, atypicality):
        confidence = np.max(probs, axis=1)
        atypicality_min = np.min(atypicality, axis=1)
        prediction_sets = np.zeros((probs.shape[0], probs.shape[1]))
        for i in range(self.n_groups):
            atypicality_mask = (atypicality_min <= self.aqs[i+1]) & (atypicality_min > self.aqs[i])
            for j in range(self.n_groups):   
                confidence_mask = (confidence <= self.cqs[j+1]) & (confidence > self.cqs[j])
                mask = confidence_mask & atypicality_mask
                prediction_sets[mask] = self.subroutines[i][j].get_sets(probs[mask])
        return prediction_sets