import numpy as np
from .DataReader import DataReader
from torch.utils.data import Dataset

class eval_direct_label_input(DataReader, Dataset):
    def __init__(self, data_path, setting, args):
        super().__init__(data_path, setting['data']['feature_size'])
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        qid = self.qids[item]
        candidate = self.candidates[item][:self.max_candidiates]
        mask = [True] * len(candidate) + [False] * (self.max_candidiates - len(candidate))
        candidate += [self.pad_id] * (self.max_candidiates - len(candidate))
        return {
            'qid' : qid,
            'candidate' : np.array(candidate),
            'mask' : np.array(mask),
            'feature' : self.get_feature(candidate),
            'label' : self.get_label(candidate)
        }



class direct_label_input(DataReader, Dataset):
    def __init__(self, data_path, setting, args):
        print("Build direct_label_input")
        super().__init__(data_path, setting['data']['feature_size'])
        self.k = args.k
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        qid = self.qids[item]
        candidate = self.candidates[item][:self.k]
        candidate += [self.pad_id] * (self.k - len(candidate))
        return {
            'qid' : qid,
            'candidate' : np.array(candidate),
            'feature' : self.get_feature(candidate),
            'label' : self.get_label(candidate)
        }
    
    def process_input(self, input_data, model, algorithm, device):
        return input_data


