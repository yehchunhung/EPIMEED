import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import mkdir
from os.path import join, exists
from pytorch_transformers import RobertaTokenizer
import boto3
import csv
import smart_open
import numpy as np  
import pickle

session = boto3.session.Session(aws_access_key_id='', aws_secret_access_key='')
s3 = session.resource('s3')

ED_emotions = ['afraid', 'angry','annoyed',
            'anticipating','anxious','apprehensive','ashamed','caring','confident','content','devastated','disappointed',
            'disgusted','embarrassed','excited','faithful','furious','grateful','guilty','hopeful','impressed','jealous',
            'joyful','lonely','nostalgic','prepared','proud','sad','sentimental','surprised','terrified','trusting',
            'agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']

max_length = 100
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

dataset_name = 'ed'
version_name = 'ed'

for file_name in ['test', 'valid', 'train']:

    print(file_name)

    with smart_open.open('s3://'+dataset_name+'-xenbot/6.new-version/'+file_name+'.csv', 'rt', encoding='utf-8', transport_params={'session': session}) as infile:
    
        readCSV = csv.reader(infile, delimiter=',')
        next(readCSV)
        # dialogue_id   context_emot    speaker_id  turn    uttr    eb_emot eb+_emot    speaker emotionality    listener neutrality dialogue score ==> ed
        # dialogue_id,   turn,    uttr,    eb_emot,  eb+_emot,    speaker emotionality,    listener neutrality,  dialogue score ==> os

        uttr_index = 0
        prev_id = ''
        count = 0
        current_dialog = []
        dial_count = 0
        N_examples = 0

        f_enc = open('./'+version_name+'/'+file_name+'/encoded.txt', 'w')
        f_uttr = open('./'+version_name+'/'+file_name+'/uttrs.txt', 'w')
        uttr_emots = []

        for row in readCSV:

            cur_id = row[0]

            if prev_id != '' and cur_id != prev_id:

                if len(current_dialog) >= 2:

                    dial_count += 1

                    first_turn = current_dialog[0]

                    dialog_id = first_turn[0]

                    #even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
                    #random_idx = random.choice(even_idx)

                    first_uttr = first_turn[4]
                    uttr_ids = tokenizer.encode(first_uttr)
                    inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]   # token ids of input
                    inp_seg_ids = [0] * (len(uttr_ids) + 2)    # 0 0 0 0 0 0 0 

                    break_point = len(current_dialog)

                    for k in range(1, len(current_dialog)):

                        uttr_ids = tokenizer.encode(current_dialog[k][4])
                        tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                        tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

                        if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:

                            inp_str = ','.join([str(x) for x in inp_ids])
                            inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                            tar_str = ','.join([str(x) for x in tar_ids])           # token ids of target
                            tar_seg_str = ','.join([str(x) for x in tar_seg_ids])   # 1 1 1 1 1 1 1

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
                            f_uttr.write('{} | {}\n'.format(dialog_id, current_dialog[k][4]))
                            uttr_emots.append(ED_emotions.index(current_dialog[k][6]))

                current_dialog = []

                #if dial_count % 100 == 0:
                #    print(dial_count)

            current_dialog.append(row)

            prev_id = cur_id

            count += 1

        if len(current_dialog) >= 2:

            dial_count += 1

            first_turn = current_dialog[0]

            dialog_id = first_turn[0]

            #even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
            #random_idx = random.choice(even_idx)

            first_uttr = first_turn[4]
            uttr_ids = tokenizer.encode(first_uttr)
            inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]   # token ids of input
            inp_seg_ids = [0] * (len(uttr_ids) + 2)    # 0 0 0 0 0 0 0 

            break_point = len(current_dialog)

            for k in range(1, len(current_dialog)):

                uttr_ids = tokenizer.encode(current_dialog[k][4])
                tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

                if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:

                    inp_str = ','.join([str(x) for x in inp_ids])
                    inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                    tar_str = ','.join([str(x) for x in tar_ids])           # token ids of target
                    tar_seg_str = ','.join([str(x) for x in tar_seg_ids])   # 1 1 1 1 1 1 1

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
                    f_uttr.write('{} | {}\n'.format(dialog_id, current_dialog[k][4]))
                    uttr_emots.append(ED_emotions.index(current_dialog[k][6]))

        current_dialog = []

    print('Number of dialogs:', dial_count)
    print('Number of examples:', N_examples)

    np.save('./'+version_name+'/'+file_name+'/uttr_emots.npy', np.array(uttr_emots))

    f_enc.close()
    f_uttr.close()



