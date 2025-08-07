import torch
import numpy as np
from .direct_label_input import eval_direct_label_input

class deterministic_online_label_input(eval_direct_label_input):
    def __init__(self, data_path, setting, args):
        print("Build deterministic_online_label_input")
        super().__init__(data_path, setting, args)
        self.k=args.k
    
    def process_input(self, input_data, model, algorithm, device):
        scores = algorithm.eval(model, input_data, device)
        scores = torch.where(input_data['mask'].to(device), scores, torch.tensor(-1e8, dtype=torch.float32, device=device))
        _, rank = torch.sort(scores, dim = 1, descending = True)
        rank = rank[:, :self.k].cpu()

        new_input_data=input_data.copy()
        new_input_data['candidate'] = torch.gather(input_data['candidate'], dim=1, index=rank)
        new_input_data['label'] = torch.gather(input_data['label'], dim=1, index=rank)
        new_input_data['feature'] = torch.tensor(self.get_feature(new_input_data['candidate']))
        del new_input_data['mask']
        return new_input_data




class stochastic_online_label_input(eval_direct_label_input):
    def __init__(self, data_path, setting, args):
        print("Build stochastic_online_label_input")
        super().__init__(data_path, setting, args)
        self.k=args.k
    
    def process_input(self, input_data, model, algorithm, device):
        scores = algorithm.eval(model, input_data, device)
        scores = torch.where(input_data['mask'].to(device), scores, torch.tensor(-1e8, dtype=torch.float32, device=device))
        rank = torch.multinomial(torch.softmax(scores, dim = 1), scores.shape[1], replacement=False)
        rank = rank[:, :self.k].cpu()

        new_input_data=input_data.copy()
        new_input_data['candidate'] = torch.gather(input_data['candidate'], dim=1, index=rank)
        new_input_data['label'] = torch.gather(input_data['label'], dim=1, index=rank)
        new_input_data['feature'] = torch.tensor(self.get_feature(new_input_data['candidate']))
        del new_input_data['mask']
        return new_input_data