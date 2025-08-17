import os
import torch
import matplotlib.pyplot as plt
import matplotlib

import utility
from utility_NLP import Vocab

matplotlib.use('TkAgg')

from utility import DATA_HUB, DATA_URL, download_extract, set_figsize

DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
def read_data_nmt():
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir,'fra.txt'),'r',encoding='utf-8') as f:
        return f.read()
def process_nmt(text):
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char !=' '
        # set 元素唯一，元素无序，可变，元素必须哈希
        # in函数判断某个元素是否属于某个容器，例如x in y，判断x是否在y内
    text = text.replace('\u202f',' ').replace('\xa0',' ').lower()
    # lower()取小写字母
    # replace('\xa0',' ')用来替换'\xa0'，把不间断空格换成普通空格
    # replace('\u202f', ' ')用来替换'\u202f'，窄不间断空格替换成普通空格
    out = [' ' + char if i > 0 and no_space(char, text[i- 1]) else char
 for i, char in enumerate(text)]
    # 这行代码是为非英文字符前面加一个空格
    return ''.join(out)
    # join函数来将一个可迭代对象，每个元素都是字符串，拼接起来，sep插在元素之间
    # 'sep'.join(iterable)

def tokenize_nmt(text,num_examples=None):
    # 词元化数据集
    source, target = [],[]
    for i,line in enumerate(text.split('\n')):
        if num_examples  and i >num_examples:
            break
        parts = line.split('\t')
        # \t制表符，按了一次tab
        if len(parts) ==2:
            source.append(parts[0].split(' '))
            # str.split(sep = None,maxsplit = -1)，sep为分隔符，maxsplit代表分割次数
            # 返回一个列表list[str]包含分割后的字串，加入不给sep，则默认为任意空白字符

            target.append(parts[1].split(' '))
    return source, target

def show_list_len_pair_hist(legend, xlabel, ylabel,xlist,ylist):
    # 绘制直方图
    set_figsize()
    # plt.ion()
    _,_,patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]]
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.show()

def truncate_pad(line, num_steps, padding_tokens):
    # 截断或者填充文本序列
    if len(line)>num_steps:
        return line[:num_steps]
    return line +[padding_tokens] * (num_steps - len(line))

def build_array_nmt(lines,vocab,num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    # 为每个序列末尾添加<eos>，表示序列结束
    array = torch.tensor([truncate_pad(
        l,num_steps,vocab['<pad>']
    ) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples = 600):
    text = process_nmt(read_data_nmt())
    source, target = tokenize_nmt(text,num_examples)
    src_vocab = Vocab(source,min_freq = 2, reserved_tokens=['<pad>','<bos>','<eos>'])
    tgt_vocab = Vocab(target,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
    src_array, src_valid_len = build_array_nmt(source,src_vocab,num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target,tgt_vocab,num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = utility.load_data(data_arrays)
    return data_iter,src_vocab,tgt_vocab



raw_text = read_data_nmt()

text = process_nmt(raw_text)
print(text[:80])
source, target = tokenize_nmt(text)
print(source[:6])
print(target[:6])
show_list_len_pair_hist(['source', 'target'], '#tokens per sequence', 'count',
                        source,target)
src_vocab = Vocab(source,min_freq=2,
                  reserved_tokens = ['<pad>','<bos>','<eos>'])
print(len(src_vocab))
truncate_pad(src_vocab[source[0]],10,src_vocab['<pad>'])


