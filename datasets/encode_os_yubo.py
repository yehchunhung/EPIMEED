import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import mkdir
from os.path import join, exists
from pytorch_transformers import RobertaTokenizer

max_length = 100
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

def get_old_random_idx(read_path):
    random_idx = []
    print('Getting the old random indices...')
    with open(read_path, 'r') as f:
        lines = f.read().splitlines()
        for line in tqdm(lines):
            s, t = line.split('\t')[-1].split(',')
            random_idx.append(int(t) - int(s))
    return random_idx

def create_dataset(csv_file_path, write_path, cascading = True, old_random_idx_path = None):
    if not cascading:
        old_random_idx = get_old_random_idx(old_random_idx_path)
        print(set(old_random_idx), len(old_random_idx))

    print('Reading from {}...'.format(csv_file_path))

    df = pd.read_csv(csv_file_path)
    print('DataFrame shape:', df.shape)

    if not exists(write_path):
        mkdir(write_path)
    f_enc = open(join(write_path, 'encoded.txt'), 'w')
    f_uttr = open(join(write_path, 'uttrs.txt'), 'w')
    uttr_emots = []

    N_rows = df.shape[0]
    current_dialog_id = -1
    current_dialog = []
    current_dialog_emots = []
    uttr_index = 0
    N_examples = 0

    final_dialog_ids = set()

    for i in tqdm(range(N_rows)):
        dialog_id = df.iloc[i]['dialogue_id']
        uttr = df.iloc[i]['text']
        emot = df.iloc[i,-42:-1].values.astype(np.float32)

        if dialog_id != current_dialog_id:
            if len(current_dialog) >= 2:
                even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
                random_idx = random.choice(even_idx)

                uttr_ids = tokenizer.encode(current_dialog[0])
                inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                inp_seg_ids = [0] * (len(uttr_ids) + 2)

                break_point = len(current_dialog)
                for k in range(1, len(current_dialog)):
                    uttr_ids = tokenizer.encode(current_dialog[k])
                    tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                    tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

                    if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:
                        inp_str = ','.join([str(x) for x in inp_ids])
                        inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                        tar_str = ','.join([str(x) for x in tar_ids])
                        tar_seg_str = ','.join([str(x) for x in tar_seg_ids])
                        if not cascading:
                            if current_dialog_id not in final_dialog_ids and k == old_random_idx[N_examples]:
                                f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                                    inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                                final_dialog_ids.add(current_dialog_id)
                                N_examples += 1
                        else:
                            f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                                inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                            N_examples += 1
                    else:
                        break_point = k
                        break

                    inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                    inp_seg_ids += tar_seg_ids

                if break_point > 1:
                    uttr_index += break_point
                    for k in range(break_point):
                        f_uttr.write('{} | {}\n'.format(current_dialog_id, current_dialog[k]))
                        uttr_emots.append(current_dialog_emots[k])

            current_dialog = [uttr]
            current_dialog_emots = [emot]
            current_dialog_id = dialog_id
        else:
            current_dialog.append(uttr)
            current_dialog_emots.append(emot)

    print(N_examples)

    if len(current_dialog) >= 2:
        even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
        random_idx = random.choice(even_idx)

        uttr_ids = tokenizer.encode(current_dialog[0])
        inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
        inp_seg_ids = [0] * (len(uttr_ids) + 2)

        break_point = len(current_dialog)
        for k in range(1, len(current_dialog)):
            uttr_ids = tokenizer.encode(current_dialog[k])
            tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
            tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

            if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:
                inp_str = ','.join([str(x) for x in inp_ids])
                inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                tar_str = ','.join([str(x) for x in tar_ids])
                tar_seg_str = ','.join([str(x) for x in tar_seg_ids])
                if not cascading:
                    if current_dialog_id not in final_dialog_ids and k == old_random_idx[N_examples]:
                        f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                            inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                        final_dialog_ids.add(current_dialog_id)
                        N_examples += 1
                else:
                    f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                        inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                    N_examples += 1
            else:
                break_point = k
                break

            inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
            inp_seg_ids += tar_seg_ids

        if break_point > 1:
            uttr_index += break_point
            for k in range(break_point):
                f_uttr.write('{} | {}\n'.format(current_dialog_id, current_dialog[k]))
                uttr_emots.append(current_dialog_emots[k])

    print('Number of examples:', N_examples)

    np.save(join(write_path, 'uttr_emots.npy'), np.array(uttr_emots))

    f_enc.close()
    f_uttr.close()

# create_dataset('os_emobert/train.csv', 'os_emobert/train')
# create_dataset('os_emobert/valid.csv', 'os_emobert/valid')
# create_dataset('os_emobert/test.csv', 'os_emobert/test')
create_dataset('os_emobert/test.csv', 'os_emobert/test_human', cascading = False, old_random_idx_path = 'os/test_human/encoded.txt')

# create_dataset('osed_emobert/train.csv', 'osed_emobert/train')
# create_dataset('osed_emobert/valid.csv', 'osed_emobert/valid')
# create_dataset('osed_emobert/test.csv', 'osed_emobert/test')
# create_dataset('osed_emobert/test.csv', 'osed_emobert/test_human', cascading = False, old_random_idx_path = 'osed/test_human/encoded.txt')
