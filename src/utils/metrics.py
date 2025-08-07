import torch
import numpy as np

def _safe_div(numerator, denominator):
    return torch.where(
        torch.eq(denominator, 0),
        torch.zeros_like(numerator),
        torch.div(numerator, denominator))

def _prepare(rankings, labels, device, topn):
    assert rankings.shape == labels.shape
    assert rankings.dim() == 2
    list_size = rankings.shape[1]
    topn = list_size if topn is None else min(topn, list_size)
    is_label_valid = (labels >= 0).to(device)
    labels = torch.where(is_label_valid, labels, 0)
    return labels, topn

def cal_dcg(labels, topn, device):
    list_size = labels.shape[1]
    discounts = (torch.tensor(1) / torch.log2(torch.arange(list_size, dtype=torch.float) + 2.0)).to(device)
    gains = torch.pow(torch.tensor(2.0, device=device), labels.to(torch.float32)) - 1.0
    discounted_gains = (gains * discounts)[:, :topn]
    return torch.sum(discounted_gains, dim=1)
    
def dcg(rankings, labels, max_label, device, topn=None):
    labels, topn = _prepare(rankings, labels, device, topn)
    dcg = cal_dcg(torch.gather(labels, dim=1, index=rankings), topn, device)
    return dcg, torch.mean(dcg)

def ndcg(rankings, labels, max_label, device, topn=None):
    labels, topn = _prepare(rankings, labels, device, topn)
    ranking_labels = torch.gather(labels, dim=1, index=rankings)
    ideal_ranking_labels, _ = torch.sort(ranking_labels, dim=1, descending=True)
    ndcg = _safe_div(cal_dcg(ranking_labels, topn, device), cal_dcg(ideal_ranking_labels, topn, device))
    return ndcg, torch.mean(ndcg)

def err(rankings, labels, max_label, device, topn=None):
    labels, topn = _prepare(rankings, labels, device, topn)
    list_size = labels.shape[1]
    ranking_labels = torch.gather(labels, dim=1, index=rankings)
    relevance = (torch.pow(torch.tensor(2.0, device=device), ranking_labels) - 1) / \
      torch.pow(torch.tensor(2.0, device=device), torch.as_tensor(max_label,device=device))
    non_rel = torch.cumprod(1.0 - relevance, dim=1) / (1.0 - relevance)
    reciprocal_rank = 1.0 / \
      torch.arange(start=1, end=list_size + 1,device=device,dtype=torch.float32)
    mask = torch.ge(reciprocal_rank, 1.0 / topn).type(torch.float32)
    err = torch.sum(relevance * non_rel * reciprocal_rank * mask, dim=1)
    return err, torch.mean(err)