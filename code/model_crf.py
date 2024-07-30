from itertools import zip_longest
import torch
import torch.nn as nn
from build_corpus import build_corpus

class Mymodel(nn.Module):
    train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
    def __init__(self, corpus_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()

        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.transition = nn.Parameter(torch.ones(class_num, class_num) * 1 / class_num)

        self.loss_fun = self.cal_lstm_crf_loss

    def cal_lstm_crf_loss(self, crf_scores, targets):
        """计算双向LSTM-CRF模型的损失
        该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
        """
        # train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
        train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
        pad_id = tag_2_index.get('<PAD>')
        start_id = tag_2_index.get('<START>')
        end_id = tag_2_index.get('<END>')

        device = crf_scores.device

        # targets:[B, L] crf_scores:[B, L, T, T]
        batch_size, max_len = targets.size()
        target_size = len(tag_2_index)

        # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
        mask = (targets != pad_id)
        lengths = mask.sum(dim=1)
        targets = self.indexed(targets, target_size, start_id)

        # # 计算Golden scores方法１
        # import pdb
        # pdb.set_trace()
        targets = targets.masked_select(mask)  # [real_L]

        flatten_scores = crf_scores.masked_select(mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)).view(-1,
                                                                                                                   target_size * target_size).contiguous()

        golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()

        # 计算golden_scores方法２：利用pack_padded_sequence函数
        # targets[targets == end_id] = pad_id
        # scores_at_targets = torch.gather(
        #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
        # scores_at_targets, _ = pack_padded_sequence(
        #     scores_at_targets, lengths-1, batch_first=True
        # )
        # golden_scores = scores_at_targets.sum()

        # 计算all path scores
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        scores_upto_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t, t, start_id, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous
                # timestep, and log-sum-exp Remember, the cur_tag of the previous
                # timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores
                # along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, end_id].sum()

        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / batch_size
        return loss

    def indexed(self, targets, tagset_size, start_id):
        """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
        batch_size, max_len = targets.size()
        for col in range(max_len - 1, 0, -1):
            targets[:, col] += (targets[:, col - 1] * tagset_size)
        targets[:, 0] += (start_id * tagset_size)
        return targets

    def forward(self, batch_data, batch_tag=None):
        embedding = self.embedding(batch_data)
        out, _ = self.lstm(embedding)

        emission = self.classifier(out)
        batch_size, max_len, out_size = emission.size()

        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition

        if batch_tag is not None:
            loss = self.cal_lstm_crf_loss(crf_scores, batch_tag)
            return loss
        else:
            return crf_scores

    def test(self, test_sents_tensor, lengths):
        """使用维特比算法进行解码"""
        train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", make_vocab=True)
        start_id = tag_2_index['<START>']
        end_id = tag_2_index['<END>']
        pad = tag_2_index['<PAD>']
        tagset_size = len(tag_2_index)

        crf_scores = self.forward(test_sents_tensor)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids.reshape(-1)