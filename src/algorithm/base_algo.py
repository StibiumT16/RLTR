import torch
from abc import ABC 
import utils.dicts as dicts
import utils.fair as fair

class BasePointAlgo(ABC):
    def __init__(self, optimizer, scheduler, args):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = args.clip_grad_norm
    
    def eval(self, model, input_data, device):
        model.eval()
        with torch.no_grad():
            output = model(input_data['feature'].to(device)).squeeze(-1)
        return output
    
    def log_pi_batch(self, scores, ranking): # PL
        '''
        def logsumexp(inputs, dim=None, keepdim=False):
            if dim is None:
                inputs = inputs.view(-1)
                dim = 0
            s, _ = torch.max(inputs, dim=dim, keepdim=True)
            outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
            if not keepdim:
                outputs = outputs.squeeze(dim)
            return outputs
        
        size, cutoff = ranking.shape
        subtracts, log_probs = torch.zeros_like(scores), torch.zeros_like(scores)
        
        for j in range(cutoff):
            posj = ranking[:, j]
            log_probs[:, [j]] = scores.gather(dim = 1, index = posj.unsqueeze(1)) - logsumexp(scores - subtracts, dim=1).view(-1, 1)
            subtracts[range(size), posj] = scores[range(size), posj] + 1e6
        '''
        ranked_scores = torch.gather(scores, dim=1, index=ranking)
        logS = torch.flip(torch.logcumsumexp(torch.flip(ranked_scores, dims=[1]), dim=1),dims=[1])
        log_probs = ranked_scores - logS
        return torch.sum(log_probs, dim = 1), log_probs


class BaseRLAlgo(BasePointAlgo):
    def __init__(self, config, optimizer, scheduler, args):
        super().__init__(optimizer, scheduler, args)
        self.max_label = config['data']['max_label']
        self.n_samples = config['algorithm'].get('group_size', 8)
        
        rewards = config['algorithm'].get('reward', 'ndcg@10')
        rewards = rewards.split(',')
        
        self.reward_funcs = []
        self.reward_topks = []
        
        for reward_func in rewards:
            try: 
                topk = int(self.reward_func.split('@')[1])
            except:
                topk = None
            reward = reward_func.split('@')[0]
        
            self.reward_funcs.append(reward)
            self.reward_topks.append(topk)

        self.reward_weights = config['algorithm'].get('reward_weights', [1.0 for _ in range(len(self.reward_funcs))])
        assert len(self.reward_weights) == len(self.reward_funcs)
    
    
    def get_reward(self, ranking, output, label, device):
        rank_label = torch.cat([label for _ in range(self.n_samples)], \
                dim = 1).view(label.shape[0] * self.n_samples, -1) 
        
        with torch.no_grad():
            reward, avg_reward = 0., 0.
            
            for (reward_func, reward_topk, reward_weight) in zip(self.reward_funcs, self.reward_topks, self.reward_weights):
                if reward_func == 'fair':
                    cur_reward, cur_avg_reward = fair.fairness_reward(ranking, output, label, self.max_label, self.n_samples, device, reward_topk)
                else:
                    cur_reward, cur_avg_reward = dicts.metric_dict[reward_func](ranking, rank_label, self.max_label, device, reward_topk)

                reward += cur_reward * reward_weight
                avg_reward += cur_avg_reward * reward_weight
            
        return reward, avg_reward
        