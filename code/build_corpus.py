import os
######读取数据变成向量
def build_corpus(split, make_vocab=True, data_dir="data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split + ".bio"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                # word, tag = line.strip('\n').split()
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list + ["<END>"])

                tag_lists.append(tag_list + ["<END>"])

                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag_2_index
    if make_vocab:
        word2id = build_map(word_lists)
        tag_2_index = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)
        word2id["<START>"] = len(word2id)
        # word2id["<END>"]   = len(word2id)

        tag_2_index['<PAD>'] = len(tag_2_index)
        tag_2_index["<START>"] = len(tag_2_index)
        # tag_2_index["<END>"] = len(tag_2_index)
        return word_lists, tag_lists, word2id, tag_2_index
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps