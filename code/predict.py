# import os
# from itertools import zip_longest
# from torch.utils.data import Dataset, DataLoader
# import torch
# # import torch.nn as nn
# # from sklearn.metrics import f1_score
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics import recall_score
# # from build_corpus import build_corpus
# # from dataset import MyDataset
# # from model_crf import Mymodel


import torch
from build_corpus import build_corpus
# global word_2_index, model, index_2_tag, device
####预测函数
def predict():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
    index_2_tag = [i for i in tag_2_index]
    model=torch.load('model5.pt')
    
    while True:
        text = input("请输入：")
        text_index = [[word_2_index.get(i, word_2_index["<UNK>"]) for i in text] + [word_2_index["<END>"]]]

        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
        pre = model.test(text_index, [len(text) + 1])
        pre = [index_2_tag[i] for i in pre]
        print([f'{w}-{s}' for w, s in zip(text, pre)])
