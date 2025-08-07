import numpy as np
from tqdm import tqdm

class DataReader:
    def __init__(self, data_path, feature_size):
        self.qids = []
        self.candidates = []
        self._features = []
        self._labels = []
        #self.dids = []
        self.max_candidiates = 0
        self.total_len = 0
        self.pad_id = 0
        
        with open(data_path) as fr:
            qid_to_qidx = {}
            for i, line in tqdm(enumerate(fr)):
                arr = line.strip().split(' ')
                qid = arr[1].split(':')[1]
                
                if qid not in qid_to_qidx:
                    qid_to_qidx[qid] = len(qid_to_qidx)
                    self.qids.append(qid)
                    self.candidates.append([])
                
                qidx = qid_to_qidx[qid]
                self.candidates[qidx].append(i)
                self._labels.append(int(arr[0]))
                #self.dids.append(f'{qid}_{i}')

                feature = [0.0 for _ in range(feature_size)]
                for x in arr[2:]:
                    arr2 = x.split(':')
                    feature_id = int(arr2[0]) - 1
                    if feature_id < feature_size:
                        feature[feature_id] = float(arr2[1])
                self._features.append(feature)
        
        self._features.append([0.0 for _ in range(feature_size)])
        self._labels.append(0)
        self.pad_id = len(self._features) - 1
        self._features = np.array(self._features)
        self._labels = np.array(self._labels)
            
        # Remove invalid qids
        invalid_qidx = [qidx for qidx in range(len(self.qids))[::-1] if \
            len(self.candidates[qidx]) <= 1 or sum(self._labels[self.candidates[qidx]]) <= 0]
        print(f'Remove {len(invalid_qidx)} invalid queries.')
        
        for qidx in invalid_qidx:
            del self.qids[qidx]
            del self.candidates[qidx]
            
        for rank_list in self.candidates:
            if self.max_candidiates < len(rank_list):
                self.max_candidiates = len(rank_list)
        
        self.total_len = len(self.qids)
        print(f"Load Dataset Successfully! Query Count: {self.total_len}\tMax Candidate Count: {self.max_candidiates}")
    
    def get_feature(self, index):
        return self._features[index]

    def get_label(self, index):
        return self._labels[index]