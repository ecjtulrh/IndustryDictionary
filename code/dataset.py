import os
from itertools import zip_longest
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from build_corpus import build_corpus

class MyDataset(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        datas = []
        tags = []
        batch_lens = []

        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas, dtype=torch.int64, device=device), torch.tensor(tags, dtype=torch.long,
                                                                                   device=device), batch_lens