import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_algo import BasePointAlgo

# Refer to: https://github.com/ULTR-Community/ULTRA_pytorch/blob/main/ultra/learning_algorithm/lambda_rank.py

class LambdaRank(BasePointAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(optimizer, scheduler, args)
        self.sigma = config['algorithm'].get('sigma', 1.0)
        self.EM_step_size = config['algorithm'].get('EM_step_size', 0.05)
        self.regulation_p = config['algorithm'].get('regulation_p', 1)
        
        self.t_plus = torch.ones([1, args.k], device=device)
        self.t_minus = torch.ones([1, args.k], device=device)
        self.t_plus.requires_grad = False
        self.t_minus.requires_grad = False
    
    def fit(self, model, input_data, device):
        model.train()
        label = input_data['label'].to(device)
        output = model(input_data['feature'].to(device)).squeeze(-1) #[bs, k]
        preds_sorted, ranking = torch.sort(output, dim=-1, descending=True)
        rank_label = torch.gather(label, dim=1, index=ranking)
        
        std_diffs = torch.unsqueeze(rank_label, dim=2) - torch.unsqueeze(rank_label, dim=1)
        std_Sij = torch.clamp(std_diffs, min=-1.0, max=1.0)
        std_p_ij = 0.5 * (1.0 + std_Sij)
        
        s_ij = torch.unsqueeze(preds_sorted, dim=2) - torch.unsqueeze(preds_sorted, dim=1)
        p_ij = 1.0 / (torch.exp(-self.sigma * s_ij) + 1.0)
        
        ideally_sorted_labels, _ = torch.sort(label, dim=1, descending=True)
        delta_ndcg = self.cal_delta_ndcg(ideally_sorted_labels, rank_label, device)
        loss = nn.BCEWithLogitsLoss(delta_ndcg, reduction='none')(p_ij, std_p_ij)
        pair_loss = torch.sum(loss,0)
        t_plus_loss_list = torch.sum(pair_loss / self.t_minus, 1)
        pair_loss_ji = torch.transpose(pair_loss, 0, 1)
        t_minus_loss_list = torch.sum(pair_loss_ji / self.t_plus,1)
        t_plus_t_minus = torch.unsqueeze(self.t_plus,2) * self.t_minus
        pair_loss_debias =  pair_loss / (t_plus_t_minus + 1e-10)
        loss = torch.sum(pair_loss_debias)
        
        with torch.no_grad():
            self.t_plus = (1 - self.EM_step_size) * self.t_plus + self.EM_step_size * torch.pow(
                t_plus_loss_list / (t_plus_loss_list[0] + 1e-10), 1 / (self.regulation_p + 1))
            self.t_minus = (1 - self.EM_step_size) * self.t_minus + self.EM_step_size * torch.pow(
                t_minus_loss_list / (t_minus_loss_list[0] + 1e-10), 1 / (self.regulation_p + 1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        return loss.item(), None

    def cal_delta_ndcg(self, ideal_labels, pred_labels, device):
        log_pos = torch.log2(torch.arange(ideal_labels.shape[1], dtype=torch.float32, device=device).unsqueeze(0) + 2.0)
        gain = torch.pow(2.0, ideal_labels) - 1.0
        idcg = torch.sum(gain / log_pos, dim=1, keepdim=True) + 1e-10  # 防止除零

        gain_pred = torch.pow(2.0, pred_labels)
        gain_diff = torch.unsqueeze(gain_pred, dim=2) - torch.unsqueeze(gain_pred, dim=1)
        decay = 1.0 / log_pos
        decay_diff = torch.unsqueeze(decay, dim=2) - torch.unsqueeze(decay, dim=1)

        delta_ndcg = torch.abs(gain_diff * decay_diff) / idcg.unsqueeze(-1)
        return delta_ndcg