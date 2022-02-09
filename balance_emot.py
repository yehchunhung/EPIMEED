import numpy as np
from tqdm import tqdm
from os.path import join
from random import sample
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pytorch_transformers import RobertaTokenizer

max_length = 100
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dm_labels = {}
with open('balance_emot/deepmoji/dm_labels.txt', 'r') as f:
    for line in f:
        emoji, idx = line.strip().split(',')
        dm_labels[int(idx)] = emoji

ebp_labels = {}
with open('balance_emot/emobert+/ebp_labels.txt', 'r') as f:
    for line in f:
        label, index = line.strip().split(',')
        ebp_labels[int(index)] = label

def create_balance_indices(read_path, write_path):
    encoded_path = join(read_path, 'encoded.txt')
    f = open(encoded_path, 'r', encoding = 'utf-8')
    print('Reading encoded data from \"{}\"...'.format(encoded_path))
    lines = f.read().splitlines()

    uttr_emots_path = join(read_path, 'uttr_emots.npy')
    print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
    uttr_emots = np.load(uttr_emots_path)
    uttr_emots = np.argmax(uttr_emots, axis = 1)

    emot_cnt = {}

    n = 0
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
        j, k = [int(s) for s in idx_str.split(',')]

        target_emot = uttr_emots[k]
        if target_emot not in emot_cnt:
            emot_cnt[target_emot] = []
        emot_cnt[target_emot].append(i)

    cnt = [len(indices) for _, indices in emot_cnt.items()]
    min_cnt = min(cnt)
    avg_cnt = sum(cnt) // len(cnt)

    removed = []
    for _, indices in emot_cnt.items():
        if len(indices) > avg_cnt:
            removed += sample(indices, len(indices) - avg_cnt)

    with open('balance_emot/{}'.format(write_path), 'w') as f_o:
        f_o.write(','.join([str(i) for i in removed]))

    f.close()

def get_emotion_distribution(read_path, write_path, label_mapping):
    encoded_path = join(read_path, 'encoded.txt')
    f = open(encoded_path, 'r', encoding = 'utf-8')
    print('Reading encoded data from \"{}\"...'.format(encoded_path))
    lines = f.read().splitlines()

    uttr_emots_path = join(read_path, 'uttr_emots.npy')
    print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
    uttr_emots = np.load(uttr_emots_path)
    uttr_emots = np.argmax(uttr_emots, axis = 1)

    emot_cnt = {}

    n = 0
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
        j, k = [int(s) for s in idx_str.split(',')]

        target_emot = uttr_emots[k]
        if target_emot not in emot_cnt:
            emot_cnt[target_emot] = []
        emot_cnt[target_emot].append(i)

    sorted_emot_cnt = sorted([(k, len(v)) for k, v in emot_cnt.items()], key = lambda x: -x[1])

    with open('balance_emot/{}'.format(write_path), 'w') as f_o:
        for k, v in sorted_emot_cnt:
            f_o.write('{},{},{}\n'.format(k, label_mapping[k], v))

    f.close()

def get_ed_emotion_distribution(read_path, write_path, label_mapping):
    print('Reading data from \"{}\"...'.format(read_path))

    with open(join(read_path, 'uttrs.txt'), 'r', encoding = 'utf-8') as f:
        uttrs = f.read().splitlines()

    dialogs_file = 'dialogs.txt'
    with open(join(read_path, dialogs_file), 'r', encoding = 'utf-8') as f:
        dialogs = [(int(i) for i in line.split(',')) for line in f.read().splitlines()]

    uttr_emots = np.load(join(read_path, 'uttr_emots.npy'))
    assert len(uttrs) == uttr_emots.shape[0]
    uttr_emots = np.argsort(uttr_emots, axis = 1)

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    emot_cnt = {}

    n = 0
    for ind, (s, t) in tqdm(enumerate(dialogs), total = len(dialogs)):
        if t - s < 2:
            continue

        uttr_ids = tokenizer.encode(uttrs[s])

        inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
        inp_seg_ids = [0] * (len(uttr_ids) + 2)
        inp_emots = [uttr_emots[s,-1]] * (len(uttr_ids) + 2)

        for i in range(s + 1, t):
            u = ' '.join(uttrs[s:i])
            if len(u.split()) > max_length: break

            uttr_ids = tokenizer.encode(uttrs[i])
            tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
            tar_seg_ids = [(i - s) % 2] * (len(uttr_ids) + 2)

            if (len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length):
                target_emot = uttr_emots[i,-1]
                if target_emot not in emot_cnt:
                    emot_cnt[target_emot] = 0
                emot_cnt[target_emot] += 1
                n += 1

            inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
            inp_seg_ids += [(i - s) % 2] * (len(uttr_ids) + 2)
            inp_emots += [uttr_emots[i,-1]] * (len(uttr_ids) + 2)

    sorted_emot_cnt = sorted([(k, v) for k, v in emot_cnt.items()], key = lambda x: -x[1])

    with open('balance_emot/{}'.format(write_path), 'w') as f_o:
        for k, v in sorted_emot_cnt:
            f_o.write('{},{},{}\n'.format(k, label_mapping[k], v))


if __name__ == '__main__':
    # create_balance_indices('../os/os/train', 'os_removed.txt')
    # create_balance_indices('../os/osed/train', 'osed_removed.txt')

    get_emotion_distribution('../../os/os_emobert/train', 'emobert+/os_ebp_dist.csv', ebp_labels)
    get_emotion_distribution('../../os/osed_emobert/train', 'emobert+/osed_ebp_dist.csv', ebp_labels)
    get_ed_emotion_distribution('../../empathetic_dialogues/data_ebp/train', 'emobert+/ed_ebp_dist.csv', ebp_labels)
