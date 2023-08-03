"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


from distutils.log import error
from functools import partial
from mimetypes import init
from types import new_class
from matplotlib.pyplot import axis
import numpy as np
from sklearn.utils import deprecated
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch import Tensor, ctc_loss
from typing import Callable, Optional

# Note: This sentence would impact the device selection
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HCEL: # HWCEL with equavilent hierarchy weights.
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    def __init__(self, H) -> None:
        self.H = H
        self.HH = LPSL.compute_levels(H)
        self.ce = [0]*len(self.HH)
        
    def loss(self, logits, labels):
        labels = labels.float()
        batch_size = labels.shape[0]
        ce = 0
        for i, hh in enumerate(self.HH):
            ce += F.cross_entropy(logits[...,hh],labels[...,hh], reduction="sum")
            self.ce[i] += (np.sum(ce.detach().cpu().numpy()) - np.sum(self.ce[:i])) / batch_size
        return ce / batch_size

class LPSL:
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    # weights: the weights for each level
    # hvp_f: hierarchical violation penalty function, default is "relu", also can be "silu"
    def __init__(self, H, lambda1=1, lambda2=1, hvp_f="relu") -> None:
        self.lambda1 = lambda1
        self.lambda2 = lambda2 # eta
        self.H = H
        # 统计每一层的类别下标 -> 2D array
        # 比如H=[-1,-1,0,0,1,1] -> HH = [[0,1],[2,3,4,5]]
        self.HH = LPSL.compute_levels(H)
        self.hvp_f = hvp_f
        self.ce, self.lps, self.hvp = [0]*len(self.HH),0,0

    def loss(self, logits, labels):
        labels = labels.float()
        batch_size = labels.shape[0]
        #exp_logits = torch.exp(logits)
        probs = torch.zeros_like(logits)
        # Compute CE
        ce = 0
        for i, hh in enumerate(self.HH):
            #probs[...,hh] = exp_logits[...,hh] / torch.sum(exp_logits[...,hh],dim=1)[...,None]
            #probs[...,hh] = torch.nan_to_num(probs[...,hh], nan=1.0)
            #ce += torch.sum(F.binary_cross_entropy(input=probs[..., hh],target=labels[..., hh], reduction="none") * labels[...,hh])
            probs[...,hh] = F.softmax(logits[...,hh],dim=1)
            # Ensure the numeric stability
            hce = F.cross_entropy(logits[...,hh], labels[...,hh], reduction="sum")
            ce += hce
            self.ce[i] += np.sum(hce.detach().cpu().numpy()) / batch_size
        # Compute LPS and HVP
        lps = 0
        hvp = 0
        for i in range(batch_size):
            PSRi, PARi = LPSL.computePSR_PAR(self.H, labels[i])
            lps += torch.sum(probs[i, PSRi])
            for j in PARi:
                if self.H[j] != -1:
                    hvp_value = 0
                    if self.hvp_f == "relu":
                        hvp_value = F.relu(probs[i,j] - probs[i, self.H[j]])
                    elif self.hvp_f == "silu":
                        hvp_value = F.silu(probs[i,j] - probs[i, self.H[j]])
                    else: raise Exception("This type of HVP function is not implemented.")
                    hvp += hvp_value 
        # We normalize the loss to avoid the situation where the learning rate is too large.
        _loss = (ce + self.lambda1*lps + self.lambda2*hvp) / batch_size / (1 + self.lambda1 + self.lambda2)
        self.lps += np.sum(lps.detach().cpu().numpy()) / batch_size
        self.hvp += np.sum(hvp.detach().cpu().numpy()) / batch_size
        return _loss
        
    def computePSR_PAR(H, one_hot_label): # one_hot_label
        PSR = [] # class indices
        PAR = []
        for i in range(len(H)):
            if H[i] != -1 and one_hot_label[H[i]] == 0:
                PSR.append(i)
            else: PAR.append(i)
        return PSR, PAR
    
    def compute_levels(H):
        max_h = 0
        for i in range(len(H)):
            h = 1
            j = i
            while (H[j] != -1): 
                j = H[j]
                h += 1
            if h > max_h:
                max_h = h
        HH = [[] for _ in range(max_h)]
        for i in range(len(H)):
            h = 1
            j = i
            while (H[j] != -1): 
                j = H[j]
                h += 1
            HH[h-1].append(i)
        return HH
                
    def test_compute_levels():
        print(LPSL.compute_levels([-1,-1,0,0,1,1]))
        print(LPSL.compute_levels([-1,-1,0,0,1,1,2,2]))
        print(LPSL.compute_levels([-1,-1,0,0,1,1,2,1]))
        print(LPSL.compute_levels([-1,-1,0,0,1,1,2,6]))

class LPSL_Lag: # Lagrangian function type of LPSL
    # 暂时不考虑性能
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    # weights: the weights for each level
    # hvp_f: hierarchical violation penalty function, default is "relu", also can be "silu"
    # init_lambda, init_gamma, max_gamma are relative to single sample
    def __init__(self, H, 
                 num_samples, 
                 init_lambda=1, 
                 init_eta=1, 
                 init_gamma=1) -> None:
        self.H = H
        # 统计每一层的类别下标 -> 2D array
        # 比如H=[-1,-1,0,0,1,1] -> HH = [[0,1],[2,3,4,5]]
        self.HH = LPSL.compute_levels(H)
        self.num_samples = num_samples
        self._lambda = init_lambda * num_samples
        self.gamma = init_gamma
        self.eta = init_eta * num_samples
        self.mu = 1 # Does not update gamma
        self.c_epoch = 1
        self.c_epoch_incre = 0
        self.h_theta = 0 
        self.g_theta = 0
        self.ce = 0
    
    def loss(self, c_epoch, logits, labels):
        if c_epoch - self.c_epoch >= 1: # New epoch start
            self.c_epoch_incre += c_epoch - self.c_epoch
            self.c_epoch = c_epoch
            if self.c_epoch_incre >= 10: # Updating paramaters every 40 epochs
                self._lambda += self.gamma * self.h_theta
                self.eta = np.max([0, self.eta + self.gamma * self.g_theta])
                self.gamma = self.gamma * self.mu
                self.c_epoch_incre = 0
            self.h_theta = 0
            self.g_theta = 0
            self.ce = 0

        labels = labels.float()
        batch_size = labels.shape[0]
        exp_logits = torch.exp(logits)
        probs = torch.zeros_like(logits)
        # Compute CE
        ce = 0
        for hh in self.HH:
            probs[...,hh] = exp_logits[...,hh] / torch.sum(exp_logits[...,hh],dim=1)[...,None]
            probs[...,hh] = torch.nan_to_num(probs[...,hh], nan=1.0)
            ce += torch.sum(F.binary_cross_entropy(input=probs[..., hh],target=labels[..., hh], reduction="none") * labels[...,hh])
            #ce += F.cross_entropy(logits[...,hh], labels[...,hh])
        self.ce += np.sum(ce.detach().cpu().numpy())
        # Compute LPS and HVP
        lps = 0
        hvp = 0
        for i in range(batch_size):
            lps_i = 0
            hvp_i = 0
            PSRi, PARi = LPSL.computePSR_PAR(self.H, labels[i])
            temp_sum = torch.sum(probs[i, PSRi]) 
            self.h_theta += temp_sum.detach().cpu().numpy()
            lps_i += self._lambda / self.num_samples * temp_sum
            lps_i += self.gamma / 2 * torch.pow(temp_sum, 2)        
            for j in PARi:
                if self.H[j] != -1: # Ignore those class whose parent is the root.
                    hvp_i += probs[i, j] - probs[i,self.H[j]]
            self.g_theta += hvp_i.detach().cpu().numpy()
            hvp_i = torch.pow(F.relu(self.eta/self.num_samples + self.gamma*hvp_i), 2) - (self.eta / self.num_samples)**2
            hvp_i = 1 / (2 * self.gamma) * hvp_i
            lps += lps_i
            hvp += hvp_i        
                    
        _loss = (ce + lps + hvp) / batch_size  
        return _loss   

class LPSL_Lag_val: 
    # Lagrangian function type of LPSL for validation
    # 暂时不考虑性能
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    # hvp_f: hierarchical violation penalty function, default is "relu", also can be "silu"
    # init_lambda, init_gamma, max_gamma are relative to single sample
    # num_samples should be the the number of validation samples.
    def __init__(self, H, num_samples) -> None:
        self.H = H
        # 统计每一层的类别下标 -> 2D array
        # 比如H=[-1,-1,0,0,1,1] -> HH = [[0,1],[2,3,4,5]]
        self.HH = LPSL.compute_levels(H)
        self.num_samples = num_samples
        self.h_theta = 0 
        self.g_theta = 0
        self.ce = 0
        self.c_epoch = 1 
    
    def loss(self, c_epoch, logits, labels):
        if c_epoch - self.c_epoch >= 1: 
            # New validation epoch start
            self.c_epoch = c_epoch
            self.h_theta = 0
            self.g_theta = 0
            self.ce = 0

        labels = labels.float()
        labels = torch.clamp(labels, 0, 1)
        batch_size = labels.shape[0]
        exp_logits = torch.exp(logits)
        probs = torch.zeros_like(logits)
        # Compute CE
        ce = 0
        for hh in self.HH:
            probs[...,hh] = exp_logits[...,hh] / torch.sum(exp_logits[...,hh],dim=1)[...,None]
            probs[...,hh] = torch.nan_to_num(probs[...,hh], nan=1.0)
            ce += torch.sum(F.binary_cross_entropy(input=probs[..., hh],target=labels[..., hh], reduction="none") * labels[...,hh])
        self.ce += np.sum(ce.detach().cpu().numpy())
        # Compute LPS and HVP
        for i in range(batch_size):
            hvp_i = 0
            PSRi, PARi = LPSL.computePSR_PAR(self.H, labels[i])
            temp_sum = torch.sum(probs[i, PSRi]) 
            self.h_theta += temp_sum.detach().cpu().numpy()       
            for j in PARi:
                if self.H[j] != -1: # Ignore those class whose parent is the root.
                    hvp_i += probs[i, j] - probs[i,self.H[j]]
            self.g_theta += hvp_i.detach().cpu().numpy()  
                  
        # We do not need to return the total loss for validation

class LPSMGL_Lag: # Lagrangian function type of LPSL
    # 暂时不考虑性能
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    # weights: the weights for each level
    # hvp_f: hierarchical violation penalty function, default is "relu", also can be "silu"
    # init_lambda, init_gamma, max_gamma are relative to single sample
    def __init__(self, H, 
                 num_samples, 
                 init_lambda=1, 
                 init_eta=1, 
                 init_gamma=1, DEVICE=None) -> None:
        self.H = H
        self.device=DEVICE
        # 统计每一层的类别下标 -> 2D array
        # 比如H=[-1,-1,0,0,1,1] -> HH = [[0,1],[2,3,4,5]]
        self.HH = LPSL.compute_levels(H)
        self.num_samples = num_samples
        self._lambda = init_lambda * num_samples
        self.gamma = init_gamma
        self.eta = init_eta * num_samples
        self.mu = 1 # Does not update gamma
        self.c_epoch = 1
        self.c_epoch_incre = 0
        self.h_theta = 0 
        self.g_theta = 0
        self.ce = 0
        self.f = lambda ec: np.exp(4*ec) + 1
        # Initialize fec
        self.fec = torch.tensor(np.ones(len(H))).float().to(self.device)
        
    def set_fec_by_confusion_matrix(self, confusion_matrix):
        nc = np.sum(confusion_matrix, axis=1)
        ec = 1 - np.diag(confusion_matrix) / nc
        self.fec = self.f(ec)
        for hh in self.HH:
            # Normalize
            self.fec[hh] = self.fec[hh] * len(hh) / np.sum(self.fec[hh])
        self.fec = torch.tensor(self.fec).float().to(self.device)
        
    def loss(self, c_epoch, logits, labels):
        if c_epoch - self.c_epoch >= 1: # New epoch start
            self.c_epoch_incre += c_epoch - self.c_epoch
            self.c_epoch = c_epoch
            if self.c_epoch_incre >= 40: # Updating paramaters every 40 epochs
                self._lambda += self.gamma * self.h_theta
                self.eta = np.max([0, self.eta + self.gamma * self.g_theta])
                self.gamma = self.gamma * self.mu
                self.c_epoch_incre = 0
            self.h_theta = 0
            self.g_theta = 0
            self.ce = 0

        labels = labels.float()
        batch_size = labels.shape[0]
        exp_logits = torch.exp(logits)
        probs = torch.zeros_like(logits)
        fec = torch.tile(self.fec, dims=(batch_size, 1))
        # Compute CE
        ce = 0
        for hh in self.HH:
            probs[...,hh] = exp_logits[...,hh] / torch.sum(exp_logits[...,hh],dim=1)[...,None]
            probs[...,hh] = torch.nan_to_num(probs[...,hh], nan=1.0)
            _ce = fec[..., hh] * labels[...,hh] * F.binary_cross_entropy(input=probs[..., hh],target=labels[..., hh], reduction="none")
            ce += torch.sum(_ce)
        self.ce += np.sum(ce.detach().cpu().numpy())
        # Compute LPS and HVP
        lps = 0
        hvp = 0
        for i in range(batch_size):
            lps_i = 0
            hvp_i = 0
            PSRi, PARi = LPSL.computePSR_PAR(self.H, labels[i])
            temp_sum = torch.sum(probs[i, PSRi]) 
            self.h_theta += temp_sum.detach().cpu().numpy()
            lps_i += self._lambda / self.num_samples * temp_sum
            lps_i += self.gamma / 2 * torch.pow(temp_sum, 2)        
            for j in PARi:
                if self.H[j] != -1: # Ignore those class whose parent is the root.
                    hvp_i += probs[i, j] - probs[i,self.H[j]]
            self.g_theta += hvp_i.detach().cpu().numpy()
            hvp_i = torch.pow(F.relu(self.eta/self.num_samples + self.gamma*hvp_i), 2) - (self.eta / self.num_samples)**2
            hvp_i = 1 / (2 * self.gamma) * hvp_i
            lps += lps_i
            hvp += hvp_i        
                    
        _loss = (ce + lps + hvp) / batch_size  
        return _loss 

class LPSMGL_Lag_val: 
    # Lagrangian function type of LPSL for validation
    # 暂时不考虑性能
    # H: a 1-D array that represents the tree structure using parent representation
    # for example: [-1,-1,0,0,1,1] represents that a two class-level tree structure, ignoring the root!
    # hvp_f: hierarchical violation penalty function, default is "relu", also can be "silu"
    # init_lambda, init_gamma, max_gamma are relative to single sample
    # num_samples should be the the number of validation samples.
    def __init__(self, H, num_samples,DEVICE=None) -> None:
        self.H = H
        self.device=DEVICE
        # 统计每一层的类别下标 -> 2D array
        # 比如H=[-1,-1,0,0,1,1] -> HH = [[0,1],[2,3,4,5]]
        self.HH = LPSL.compute_levels(H)
        self.num_samples = num_samples
        self.h_theta = 0 
        self.g_theta = 0
        self.ce = 0
        self.c_epoch = 1
        self.f = lambda ec: np.exp(4*ec) + 1
        self.fec = torch.tensor(np.ones(len(H))).float().to(self.device)
        
    def set_fec_by_confusion_matrix(self, confusion_matrix):
        nc = np.sum(confusion_matrix, axis=1)
        ec = 1 - np.diag(confusion_matrix) / nc
        self.fec = self.f(ec)
        for hh in self.HH:
            # Normalize
            self.fec[hh] = self.fec[hh] * len(hh) / np.sum(self.fec[hh])
        self.fec = torch.tensor(self.fec).float().to(self.device)
        
    def loss(self, c_epoch, logits, labels):
        if c_epoch - self.c_epoch >= 1: 
            # New validation epoch start
            self.c_epoch = c_epoch
            self.h_theta = 0
            self.g_theta = 0
            self.ce = 0

        labels = labels.float()
        labels = torch.clamp(labels, 0, 1)
        batch_size = labels.shape[0]
        exp_logits = torch.exp(logits)
        probs = torch.zeros_like(logits)
        fec = torch.tile(self.fec, dims=(batch_size, 1))
        # Compute CE
        ce = 0
        for hh in self.HH:
            probs[...,hh] = exp_logits[...,hh] / torch.sum(exp_logits[...,hh],dim=1)[...,None]
            probs[...,hh] = torch.nan_to_num(probs[...,hh], nan=1.0)
            _ce = fec[...,hh] * labels[...,hh] * F.binary_cross_entropy(input=probs[..., hh],target=labels[..., hh], reduction="none")
            ce += torch.sum(_ce)
        self.ce += np.sum(ce.detach().cpu().numpy())
        # Compute LPS and HVP
        for i in range(batch_size):
            hvp_i = 0
            PSRi, PARi = LPSL.computePSR_PAR(self.H, labels[i])
            temp_sum = torch.sum(probs[i, PSRi]) 
            self.h_theta += temp_sum.detach().cpu().numpy()       
            for j in PARi:
                if self.H[j] != -1: # Ignore those class whose parent is the root.
                    hvp_i += probs[i, j] - probs[i,self.H[j]]
            self.g_theta += hvp_i.detach().cpu().numpy()  
                  
        # We do not need to return the total loss for validation

if __name__ == '__main__':
    CLASS_HIERARCHY = [
                       # Level-1
                       -1, -1, -1, -1, 
                       # Level-2
                       0, 0, 1, 1, 0, 2, 3, 0, 1, 0, 3, 1, 3, 2, 1, 2, 0, 0, 1, 0, 0,
                       # Level-3
                       4, 5, 12, 12, 6, 11, 19, 4, 10, 7, 13, 19, 9, 19, 4, 19, 12, 22, 6, 22, 22, 6, 14, 16, 13, 15, 13, 10, 17, 18, 22, 6, 20, 19, 8, 22, 22, 21, 22, 6, 6, 22, 22, 23, 24
                    ]
    HH = LPSL.compute_levels(CLASS_HIERARCHY)
    one_hot = [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    print(LPSL.computePSR_PAR(CLASS_HIERARCHY,one_hot))
    print(HH)
    