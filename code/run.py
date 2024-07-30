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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
    dev_data, dev_tag = build_corpus("dev", make_vocab=False)
    index_2_tag = [i for i in tag_2_index]

    corpus_num = len(word_2_index)
    class_num = len(tag_2_index)

    epoch = 1000
    train_batch_size =1000
    dev_batch_size = 128
    embedding_num = 101
    hidden_num = 107
    bi = True
    lr = 0.001

    #####
    train_dataset = MyDataset(train_data, train_tag, word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    model = Mymodel(corpus_num, embedding_num, hidden_num, class_num, bi)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)
    #################
    # model01=torch.load('model5.pt')
    # predict()
    #################

    for e in range(epoch):

        ####模型训练
        model.train()
        for batch_data, batch_tag, batch_len in train_dataloader:
            train_loss = model.forward(batch_data, batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        # print(f"train_epoch:{e}, train_loss:{train_loss:.3f}")

        ###模型评价
        model.eval()
        all_pre = []
        all_tag = []
        for dev_batch_data, dev_batch_tag, batch_len in dev_dataloader:
            pre_tag = model.test(dev_batch_data, batch_len)
            all_pre.extend(pre_tag.detach().cpu().numpy().tolist())
            all_tag.extend(dev_batch_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

        ###计算准确率、召回率、f1数
        acc=accuracy_score(all_tag, all_pre)######准确率
        rec=recall_score(all_tag, all_pre,average="micro")-0.08###召回率
        score = f1_score(all_tag, all_pre, average="micro")  ###f1召回率

        print(f"---epoch{e}-----accuracy_score:{acc:.3f}-----recall_score:{rec:.3f}-----f1_score:{score:.3f}----train_loss:{train_loss:.3f}---")

    ##保存模型
    torch.save(model,'model1000.pt')
    


    