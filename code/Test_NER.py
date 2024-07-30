import os
from itertools import zip_longest
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from build_corpus import build_corpus
from dataset import MyDataset
from model_crf import Mymodel
from predict import predict



if __name__ == "__main__":
    # train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
    # global model
    # model=torch.load('model5.pt')
    predict()