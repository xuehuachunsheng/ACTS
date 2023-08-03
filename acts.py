# This file implements the acts strategy and information measures
# Author: Y.X. Wu, et al.

from abc import ABC, abstractmethod

import os,sys
import torch
import numpy as np
from PIL import Image
from losses import LPSL

def softmax(logit:np.ndarray):
    exp_logit = np.exp(logit)
    return exp_logit / (np.sum(exp_logit) + 1e-6)

# Information Measures
class IM:
    def he(prob:np.ndarray, H, HH):
        prob = prob + 1e-6
        he = 0
        for h in HH:
            ent_h = np.sum(prob[h] * np.log(prob[h]))
            he += ent_h
        return -he

    def pdce(prob:np.ndarray, H, HH):
        prob = prob + 1e-6
        value = 0
        for c_id, p_id in enumerate(H):
            if p_id == -1: 
                continue
            value += prob[c_id] * np.log(prob[p_id])
        return -value
                 
    def cdce(prob:np.ndarray, H, HH):
        prob = prob + 1e-6
        value = 0
        H = np.asarray(H, dtype=np.int32)
        for level in HH[:-1]:
            for p_id in level:
                c_idx = np.where(H == p_id)[0]
                p_prob = prob[p_id]
                c_prob_sum = np.sum(prob[c_idx])
                value += p_prob * np.log(c_prob_sum)
        return -value

    def hcce(prob:np.ndarray, H, HH):
        prob = prob + 1e-6
        value = 0
        for c_id, p_id in enumerate(H):
            if p_id == -1 or prob[c_id] <= prob[p_id]:
                continue
            value += (prob[c_id] - prob[p_id]) * np.log(prob[c_id] - prob[p_id])
        return -value

    def hdre(prob, H, HH):
        return IM.he(prob, H, HH) + 0.25 * IM.pdce(prob, H, HH) + 0.25 * IM.cdce(prob, H, HH) + IM.hcce(prob, H, HH)

    def hdre(he, pdce, cdce, hcce):
        return he + 0.25 * pdce + 0.25 * cdce + hcce
    
# Batch Sample Selection Sttrategy for DeepAL
class BatchSampleSelectionStrategy(ABC):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        self.U = U # unlabeled pool
        self.classifier = classifier # classifier without softmax
        self.q_budget = q_budget
        self.device = DEVICE
        #self.compute_logits()
        
    @abstractmethod
    def select(self): # Return the selected sample idx
        pass
    
    def compute_logits(self):
        idx = self.U.dataidx
        logits = {}
        self.classifier.eval()
        with torch.no_grad():
            for count,i in enumerate(idx):
                label,img = self.U.data[i]
                img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
                img = img.convert("RGB")
                img = self.U.transform(img).float()
                img = img.to(self.device)
                logit = self.classifier(img[None, ...])[0].cpu().numpy()
                logits[i] = logit
                if (count + 1) % 50 == 0:
                    print("\r U count:{}/total:{}".format(count,len(idx)), end="")
        self.logits = logits
    
    # Shift sample idd from U --> L
    def update(L, U, sample_idx:list):
        for idd in sample_idx:
            assert idd in U.data
            assert idd not in L.data
            L.data[idd] = U.data[idd]
            del U.data[idd]
        L.dataidx = list(L.data)
        U.dataidx = list(U.data)

class RandomStrategy(BatchSampleSelectionStrategy):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
    
    def select(self):
        idx = self.U.dataidx
        idx_copy = np.array(idx, dtype=np.int32)
        np.random.shuffle(idx_copy) 
        return np.asarray(idx_copy[:self.q_budget])

class EntropyStrategy(BatchSampleSelectionStrategy):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
    
    def select(self):
        self.compute_logits()
        logits = self.logits
        idx = self.U.dataidx
        assert len(logits) == len(idx)
        ents = {}
        for p_key in logits:
            prob = softmax(logits[p_key])
            prob = prob + 1e-6
            ents[p_key] = -np.sum(prob * np.log(prob))
        ents = list(ents.items())
        ents = sorted(ents, key=lambda x:-x[1]) # Sort by descedant order
        select_idx = []
        for e in ents[:self.q_budget]:
            select_idx.append(e[0])
        return np.asarray(select_idx)

class MarginSamplingStrategy(BatchSampleSelectionStrategy):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
    
    def select(self):
        self.compute_logits()
        logits = self.logits
        idx = self.U.dataidx
        assert len(logits) == len(idx)
        ms = {}
        for p_key in logits:
            prob = softmax(logits[p_key])
            sorted_prob = np.sort(prob)
            ms[p_key] = sorted_prob[-1] - sorted_prob[-2] # max - max2
        
        ms = list(ms.items())
        ms = sorted(ms, key=lambda x:x[1]) # Sort by ascending order
        select_idx = []
        for e in ms[:self.q_budget]:
            select_idx.append(e[0])
        return np.asarray(select_idx)

class LeastConfidenceStrategy(BatchSampleSelectionStrategy):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
    
    def select(self):
        self.compute_logits()
        logits = self.logits
        idx = self.U.dataidx
        assert len(logits) == len(idx)
        ls = {}
        for p_key in logits:
            prob = softmax(logits[p_key])
            ls[p_key] = np.max(prob)
        ls = list(ls.items())
        ls = sorted(ls, key=lambda x:x[1]) # Sort by ascending order
        select_idx = []
        for e in ls[:self.q_budget]:
            select_idx.append(e[0])
        return np.asarray(select_idx)

class BatchSampleSelectionStrategyHC(BatchSampleSelectionStrategy):
    # H: the class hierarchy defined by parent representation
    def __init__(self, U, classifier, q_budget, DEVICE, H) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
        self.H = H
        self.HH = LPSL.compute_levels(H) # Compute idx for each class level

    @abstractmethod
    def select(self):
        pass

class HEntropyStrategy(BatchSampleSelectionStrategyHC):
    def __init__(self, U, classifier, q_budget, DEVICE, H) -> None:
        super().__init__(U, classifier, q_budget, DEVICE, H)
    
    def select(self):
        self.compute_logits()
        logits = self.logits
        idx = self.U.dataidx
        assert len(logits) == len(idx)
        ents = {}
        for p_key in logits:
            ents[p_key] = 0
            for level in self.HH:
                prob = softmax(logits[p_key][level])
                ents[p_key] += -np.sum(prob * np.log(prob))
        ents = list(ents.items())
        ents = sorted(ents, key=lambda x:-x[1]) # Sort by descedant order
        print(ents[:self.q_budget])
        select_idx = []
        for e in ents[:self.q_budget]:
            select_idx.append(e[0])
        return np.asarray(select_idx)

class HLeastConfidenceStrategy(BatchSampleSelectionStrategyHC):
    pass

class HMarginSamplingStrategy(BatchSampleSelectionStrategyHC):
    pass

# Approximate class balance typical sampling
class Acts(BatchSampleSelectionStrategyHC):
    # Note that this class need labeled dataset to compute the number of samples of each class
    def __init__(self, 
                 L,  # Labeled dataset
                 U,  # Unlabeled Pool
                 classifier, # Classifier
                 q_budget, # Budget at each query, but not total budget, the total budget is q_budget * m
                 DEVICE, # gpu or cpu
                 H, # 
                 delta, # Delta function
                 m, # The number for querying
                 T1=0.1, 
                 Tm=0.001,
                 DEBUG=True, # Experimental mode
                 exp_out_file=None, # Output file path
                 ) -> None:
        super().__init__(U, classifier, q_budget, DEVICE, H)
        assert delta in ["F", "min", "max"]
        self._delta = delta
        self.L = L
        self.T1 = T1
        self.Ti = T1
        self.Tm = Tm
        self.alpha = np.power(Tm/(T1 + 1e-9), 1/(m-1)) if self.T1 != 0 else 0
        self.DEBUG = DEBUG
        self.exp_out_file = exp_out_file
        
        # DEBUG mode
        if DEBUG:
            assert exp_out_file is not None and isinstance(exp_out_file, str)
            assert os.path.exists(os.path.dirname(exp_out_file))
            f = open(exp_out_file, "w")
            nc_str = ["c_{}".format(i) for i in range(len(self.HH[-1]))]
            nc_str = ",".join(nc_str)
            bc_str = ["bc_{}".format(i) for i in range(len(self.HH[-1]))]
            bc_str = ",".join(bc_str)
            f.write("QueryID,AVG_HDRE,AVG_HE,AVG_PDCE,AVG_CDCE,AVG_HCCE,AVG_DELTA,SELECT_AVG_HDRE,SELECT_AVG_HE,SELECT_AVG_PDCE,SELECT_AVG_CDCE,SELECT_AVG_HCCE,SELECT_ACG_DELTA,{},{}\n".format(nc_str,bc_str))
            f.close()
        
    def select(self):
        # data structure of act_scores:
        # [
        #    {sample1_id: (scores1, delta1), sample2_id: (scores2, delta2), ...}, # class_0
        #    {sample1_id: (scores1, delta1), sample2_id: (scores2, delta2), ...}, # class_1        
        # ]
        print("\nCompute nc...")
        self.nc = self.compute_nc() # 计算最后一层每个类别的样本数量
        print("\n", self.nc)
        print("\nCompute logits...")
        self.compute_logits()
        print("\nCompute bc...")
        bc = self.compute_bc()
        print(bc)
        assert len(self.nc) == len(self.HH[-1])
        assert len(self.U.data) == len(self.U.dataidx)
        assert len(self.L.data) == len(self.L.dataidx)
        act_scores = [{} for _ in range(len(self.nc))] # act scores for the last level
        
        print("Compute act scores...")
        # Step 1. Compute hdre and delta value
        for s_id in self.U.dataidx:
            logit = self.logits[s_id]
            prob = np.zeros(len(logit), dtype=np.float32)
            for hh in self.HH: # softmax from class level
                prob[hh] = softmax(logit[hh])
            #hdre_x = IM.hdre(prob,self.H,self.HH)
            he_x = IM.he(prob, self.H, self.HH)
            he_x = 0 if np.isnan(he_x) else he_x
            pdce_x = IM.pdce(prob, self.H, self.HH)
            pdce_x = 0 if np.isnan(pdce_x) else pdce_x
            cdce_x = IM.cdce(prob, self.H, self.HH)
            cdce_x = 0 if np.isnan(cdce_x) else cdce_x
            hcce_x = IM.hcce(prob, self.H, self.HH)
            hcce_x = 0 if np.isnan(hcce_x) else hcce_x
            hdre_x = IM.hdre(he_x, pdce_x, cdce_x, hcce_x)
            
            delta = self.compute_delta(prob)
            # Note: ccc is the class id with max probability at the last level
            # Meanwhile, ccc also relates to class id of the origin data.
            # equavilent to
            # ccc = np.argmax(prob[-len(nc):])
            ccc = np.argmax(prob[self.HH[-1]])
            act_scores[ccc][s_id] = (hdre_x, he_x, pdce_x, cdce_x, hcce_x, delta)

        print("Compute sorted act scores by HDRE...")
        # Step 2. Ranking by HDRE descendant
        sorted_act_scores = []
        for c_scores in act_scores:
            if len(c_scores) == 0:
                sorted_act_scores.append([])
                continue
            s_scores = sorted(list(c_scores.items()), key=lambda x: -x[1][0])
            sorted_act_scores.append(s_scores)
        print("sorted_act_score length: ", [len(s_score) for s_score in sorted_act_scores])
        print("Filter the act score and select samples...")
        # Step 3. Filter according to delta.
        # Note: Ensure two issues: 
        # (1) The number of queried samples of each class is enough
        # (2) The budget is satisfied predefined q_budget
        assert self.q_budget < len(self.U.dataidx)
        selected_CC = [[] for _ in range(len(self.nc))]
        rem_idx = [[] for _ in range(len(self.nc))]
        n_selected = 0
        for i, c_scores in enumerate(sorted_act_scores):
            if len(c_scores) <= bc[i]:
                for ele in c_scores:
                   selected_CC[i].append(ele[0]) # 全部选择
                   n_selected += 1
                   if n_selected % 10 == 0:
                       print("\r{}".format(n_selected), end="")
                continue
            
            for j,ele in enumerate(c_scores):
                if ele[1][-1] >= self.Ti:
                    if len(selected_CC[i]) >= bc[i]:
                        for t_ele in c_scores[j+1:]:
                            rem_idx[i].append(t_ele[0])    
                        break
                    selected_CC[i].append(ele[0])
                    n_selected += 1
                    if n_selected % 10 == 0:
                        print("\r{}".format(n_selected), end="")
                else:
                    rem_idx[i].append(ele[0])
        num_rem_idx = sum([len(j) for j in rem_idx])
        print("\nRemaining idx length: ", num_rem_idx)
        print("\nSupply the remaining indices if n_selected <= self.q_budget...")
        
        assert n_selected <= self.q_budget
        c_id = 0 # 第几个类
        cc_idx = [0] * len(self.nc) # 第几个类当前挑选的下标
        n_rem = 0
        while n_selected < self.q_budget and n_rem < num_rem_idx:
            if cc_idx[c_id] < len(rem_idx[c_id]):
                selected_CC[c_id].append(rem_idx[c_id][cc_idx[c_id]])
                cc_idx[c_id] += 1
                n_selected += 1
                n_rem += 1
            c_id = (c_id + 1) % len(self.nc)
        select_idx = []
        #print(selected_CC)
        for cc in selected_CC:
            select_idx.extend(cc)
        assert len(set(select_idx)) == len(select_idx)
        self.Ti = self.Ti * self.alpha
        print("\n Select Done!")
        
        # Experimental mode
        if self.DEBUG:
            print("[DEBUG] Writing file....")
            avg_hdre, avg_he, avg_pdce, avg_cdce, avg_hcce, avg_delta = 0,0,0,0,0,0
            s_avg_hdre, s_avg_he, s_avg_pdce, s_avg_cdce, s_avg_hcce, s_avg_delta = 0,0,0,0,0,0
            select_idx_set = set(select_idx)
            for c_act_scores in act_scores:
                for k in c_act_scores:
                    avg_hdre += c_act_scores[k][0]
                    avg_he += c_act_scores[k][1]
                    avg_pdce += c_act_scores[k][2]
                    avg_cdce += c_act_scores[k][3]
                    avg_hcce += c_act_scores[k][4]
                    avg_delta += c_act_scores[k][5]
                    if k in select_idx_set:
                        s_avg_hdre += c_act_scores[k][0]
                        s_avg_he += c_act_scores[k][1]
                        s_avg_pdce += c_act_scores[k][2]
                        s_avg_cdce += c_act_scores[k][3]
                        s_avg_hcce += c_act_scores[k][4]
                        s_avg_delta += c_act_scores[k][5]
            
            avg_hdre /= len(self.U.dataidx)
            avg_he /= len(self.U.dataidx)
            avg_pdce /= len(self.U.dataidx)
            avg_cdce /= len(self.U.dataidx)
            avg_hcce /= len(self.U.dataidx)
            avg_delta /= len(self.U.dataidx)
            
            s_avg_hdre /= len(select_idx)
            s_avg_he /= len(select_idx)
            s_avg_pdce /= len(select_idx)
            s_avg_cdce /= len(select_idx)
            s_avg_hcce /= len(select_idx)
            s_avg_delta /= len(select_idx)
            
            nc_str = ",".join([str(x) for x in self.nc])
            bc_str = ",".join([str(x) for x in bc])
            
            if not hasattr(self,"q_id"):
                self.q_id = 1
            f = open(self.exp_out_file, "a")
            #f.write("QueryID,AVG_HDRE,AVG_HE,AVG_PDCE,AVG_CDCE,AVG_HCCE,AVG_DELTA, \
            #        SELECT_AVG_HDRE,SELECT_AVG_HE,SELECT_AVG_PDCE,SELECT_AVG_CDCE,SELECT_AVG_HCCE,SELECT_ACG_DELTA,{},{}\n".format(nc_str,bc_str))
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{}\n".format(self.q_id,
                                                avg_hdre,avg_he,avg_pdce,avg_cdce,avg_hcce,avg_delta,
                                                s_avg_hdre,s_avg_he,s_avg_pdce,s_avg_cdce,s_avg_hcce,s_avg_delta, 
                                                nc_str, bc_str))
            f.close()
            self.q_id += 1
            print("Writing Done...")
            
        #return np.asarray(select_idx)
        self.select_idx = select_idx
    

    def compute_bc(self):
        nc = np.asarray(self.nc)
        C_h = len(nc)
        mean_nc = np.mean(nc)
        C_hat = np.where(nc - mean_nc < self.q_budget / C_h)[0] # C_hat
        assert len(C_hat) != 0
        bc = np.zeros(C_h, dtype=np.int32)
        for i in range(C_h): 
            if nc[i] - mean_nc < self.q_budget / C_h:
                bc[i] = np.round(mean_nc - nc[i] + self.q_budget / len(C_hat))
        
        # 严格控制查询预算
        t_Bi = int(np.sum(bc))
        while t_Bi < self.q_budget: # 如果实际小于预期budget，则随机选择C_hat中的类别提高其查询数量，直到满足当前budget
            tc = np.random.randint(low=0, high=len(C_hat))
            bc[C_hat[tc]] += 1
            t_Bi += 1
        while t_Bi > self.q_budget: # # 如果实际大于预期budget，则随机选择C_hat中的类别降低其查询数量，直到满足当前budget
            tc = np.random.randint(low=0, high=len(C_hat))
            if bc[C_hat[tc]] > 0: # 不允许bc小于0
                bc[C_hat[tc]] -= 1
                t_Bi -= 1
        assert t_Bi == self.q_budget
        return bc

    def compute_nc(self):
        nc = np.zeros(len(self.HH[-1]), dtype=np.int32)
        assert len(self.L.dataidx) == len(self.L.data)
        for i, id in enumerate(self.L.dataidx):
            print("\r Ldata:{}/{}".format(i, len(self.L.dataidx)), end="")
            label,_ = self.L.data[id]
            nc[label] += 1
        return nc
    
    # Shift sample idd from U --> L
    # Update n_c also
    def update_data(self):
        L,U = self.L,self.U
        for idd in self.select_idx:
            assert idd in U.data
            assert idd not in L.data
            L.data[idd] = U.data[idd]
            del U.data[idd]
        L.dataidx = list(L.data)
        U.dataidx = list(U.data)
        
    def compute_delta(self, prob):
        value = 0
        levels_probs = []
        for h in self.HH:
            sorted_prob_h = np.sort(prob[h])
            levels_probs.append(sorted_prob_h)
        if self._delta == "F":
            value = levels_probs[-1][-1] - levels_probs[-1][-2]
        else:
            values = [levels_probs[i][-1] - levels_probs[i][-2] for i in range(len(levels_probs))]
            value = np.max(values) if self._delta == "max" else np.min(values)
        return value

def testQuery():
    import os,sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import torch
    import torch.nn as nn
    from torchvision import models
    import numpy as np
    from FashionMNIST.dataset import FashionMNISTDatasetAL
    
    n_classes = 18
    HLevel=[1,2,3]
    L,U = FashionMNISTDatasetAL.createLURandomly(HLevel=HLevel,nL=55000)
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])
    model.classifier.add_module("logits", nn.Linear(4096, n_classes))
    # 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
    model.classifier[-1].bias.data.fill_(-np.log(n_classes-1))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.load_state_dict(torch.load("/home/wyx/vscode_projects/hier/models/FashionMNIST/best_model_hce.pth"))
    #es = EntropyStrategy(U, model, 500, DEVICE)
    #sample_idx = es.select()
    
    # es = MarginSamplingStrategy(U, model, 500, DEVICE)
    # sample_idx = es.select()
    # print(sample_idx)
    
    # es = LeastConfidenceStrategy(U, model, 500, DEVICE)
    # sample_idx = es.select()
    # print(sample_idx)
    
    # H = [-1,-1,
    #         0,0,0,0,1,1,
    #         2,2,2,3,4,5,6,7,7,7]
    # es = HEntropyStrategy(U, model, 500, DEVICE, H)
    # sample_idx = es.select()
    # print(sample_idx)
    
    H = [-1,-1,
            0,0,0,0,1,1,
            2,2,2,3,4,5,6,7,7,7]
    es = Acts(L, U, model, 500, DEVICE, H, delta="F", m=10, T1=0.1, Tm=0.001)
    sample_idx = es.select()
    print(sample_idx)
    
    Acts.update(L,U,sample_idx)
    sample_idx = es.select()
    print(sample_idx)
    
    Acts.update(L,U,sample_idx)
    sample_idx = es.select()
    print(sample_idx)

def testHE():
    H = [-1,-1,
        0,0,1,1]
    HH = LPSL.compute_levels(H)
    prob = np.asarray([0.7, 0.3, 0.4, 0.2, 0.2, 0.2])
    he = IM.he(prob, H, HH)
    pdce = IM.pdce(prob, H, HH)
    cdce = IM.cdce(prob, H, HH)
    hcce = IM.hcce(prob, H, HH)
    hdre = IM.hdre(prob, H, HH)
    assert abs(he - 1.943) < 1e-3
    assert abs(pdce - 0.696) < 1e-3
    assert abs(cdce - 0.632) < 1e-3
    assert hcce == 0
    assert abs(hdre - 2.275) < 1e-3
    
    prob = np.asarray([0.2,0.8,0.5,0.1,0.1,0.3])
    he = IM.he(prob, H, HH)
    pdce = IM.pdce(prob, H, HH)
    cdce = IM.cdce(prob, H, HH)
    hcce = IM.hcce(prob, H, HH)
    hdre = IM.hdre(prob, H, HH)
    assert abs(he - 1.669) < 1e-3
    assert abs(pdce - 1.055) < 1e-3
    assert abs(cdce - 0.835) < 1e-3
    assert abs(hcce - 0.361) < 1e-3
    assert abs(hdre - 2.502) < 1e-3

if __name__ == "__main__":
    testQuery()
    #testHE()
    