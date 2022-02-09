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

import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import * 

session = boto3.session.Session(aws_access_key_id='AKIAJS7S46C2O7N2C62Q', aws_secret_access_key='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')
s3 = session.resource('s3')

ED_emotions = ['afraid', 'angry','annoyed',
            'anticipating','anxious','apprehensive','ashamed','caring','confident','content','devastated','disappointed',
            'disgusted','embarrassed','excited','faithful','furious','grateful','guilty','hopeful','impressed','jealous',
            'joyful','lonely','nostalgic','prepared','proud','sad','sentimental','surprised','terrified','trusting',
            'agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']

ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']

ED_mapping = {
            'prepared': 'a',
            'anticipating': 'b',
            'hopeful': 'c',
            'proud': 'd',

            'excited': 'e',
            'joyful': 'f',
            'content': 'g',
            'caring': 'h',

            'grateful': 'i',
            'trusting': 'j',
            'confident': 'k',
            'faithful': 'l',

            'impressed': 'm',
            'surprised': 'n',

            'terrified': 'o',
            'afraid': 'p',
            'apprehensive': 'q',
            'anxious': 'r',

            'embarrassed': 's',
            'ashamed': 't',

            'devastated': 'u',
            'sad': 'v',
            'disappointed': 'w',
            'lonely': 'x',
            'sentimental': 'y',
            'nostalgic': 'z',
            'guilty': '0',

            'disgusted': '1',

            'furious': '2',
            'angry': '3',
            'annoyed': '4',
            'jealous': '5',

            'agreeing': '6',
            'acknowledging': '7',
            'encouraging': '8',
            'consoling': '9',
            'sympathizing': '+',
            'suggesting': '*',
            'questioning': '=',
            'wishing': '&',
            'neutral': '#',
            }

max_length = 100
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

ED_emotions_ordered = ['prepared', 'anticipating', 'hopeful', 'proud', 'excited', 'joyful', 'content', 'caring', 'grateful', 'trusting', 'confident', 'faithful', 'impressed', 'surprised', 'terrified', 'afraid', 'apprehensive', 'anxious', 'embarrassed', 'ashamed', 'devastated', 'sad', 'disappointed', 'lonely', 'sentimental', 'nostalgic', 'guilty', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']


def predict_morkov(path, clf):

    initial = ""

    for emo in path:
        c = ED_mapping[emo]
        initial = initial + c

    #initial = "p=p"
    prob_arr = []

    for emo in ED_emotions_ordered:
        c = ED_mapping[emo]
        seq = initial + c
        prob = clf.log_probability(list(seq))
        #print(seq, prob)
        prob_arr.append(prob)

    equal_prob = True
    initial_prob = prob_arr[0]
    for i in range(1, len(prob_arr)):
        if prob_arr[i] != initial_prob:
            equal_prob = False

    if equal_prob:
        return 'neutral'
    else:
        winner = np.argwhere(prob_arr == np.amax(prob_arr))
        winner = winner.flatten().tolist()

        output = ED_emotions_ordered[np.random.choice(winner, 1)[0]]
        return output


for model_name in ['os-dramatic-rest-ed', 'os-dramatic-rest-osed-dramatic', 'os-dramatic-rest']:

    if model_name == 'os-dramatic-rest':
        model_dataset = 'os-dramatic-rest'
    elif model_name == 'os-dramatic-rest-ed':
        model_dataset = 'ed'
    elif model_name == 'os-dramatic-rest-osed-dramatic':
        model_dataset = 'osed-dramatic'


    # =================================================================================================

    seq_list = []

    # dialogue_id   context_emot    speaker_id  turn    uttr    eb_emot eb+_emot    speaker emotionality    listener neutrality dialogue score ==> ed
    # dialogue_id,   turn,    uttr,    eb_emot,  eb+_emot,    speaker emotionality,    listener neutrality,  dialogue score ==> os

    print(model_dataset)

    session = boto3.session.Session(aws_access_key_id='AKIAJS7S46C2O7N2C62Q', aws_secret_access_key='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')
    s3 = session.resource('s3')

    count = 0
    turn_count = 0

    path = ''
    emo_index = 100
    if model_dataset == "ed":
        path = 's3://'+model_dataset+'-xenbot/6.new-version/train.csv'
        emo_index = 6
    elif "osed-" in model_dataset:
        path = 's3://osed-xenbot/6.new-version/'+model_dataset+'/train.csv'
        emo_index = 4
    elif "os-" in model_dataset:
        path = 's3://os-xenbot/6.new-version/'+model_dataset+'/train.csv'
        emo_index = 4

    print(path)

    with smart_open.open(path, 'rt', encoding='utf-8', transport_params={'session': session}) as infile:
      
      readCSV = csv.reader(infile, delimiter=',')
      next(readCSV)

      prev_id = ''
      dialog = []

      #new_row = ['dialogue_id', 'turn', 'uttr', 'eb_emot', 'eb+_emot', 'speaker emotionality', 'listener neutrality', 'dialogue score'] ==> os, osed
      # new_row = ['dialogue_id', 'context_emot', 'speaker_id', 'turn', 'uttr', 'eb_emot', 'eb+_emot', 'speaker emotionality', 'listener neutrality', 'dialogue score' ==> ed]

      for row in readCSV:

        cur_id = row[0]

        if prev_id != '' and cur_id != prev_id:

          seq = ''

          for i in range(0, len(dialog)):
              
              emot = dialog[i][emo_index]
              character = ED_mapping[emot]

              seq += character

              turn_count += 1

          seq_list.append(seq)

          dialog = []

        dialog.append(row)

        prev_id = cur_id

        count += 1

      if len(dialog) > 0:

        seq = ''

        for i in range(0, len(dialog)):
            
            emot = dialog[i][emo_index]
            character = ED_mapping[emot]

            seq += character

            turn_count += 1

        seq_list.append(seq)

        dialog = []


    print("=== end ===")

    print("Row count = ", count)
    print("Turn count = ", turn_count)
    print("Sequence list length = ", len(seq_list))

    # =================== Building markov chain =====================

    d1_dict = {}
  
    for emotion in ED_emotions_ordered:
        character = ED_mapping[emotion]
        d1_dict[character] = 0

    #print(d1_dict)

    d2_arr = []

    for emo1 in ED_emotions_ordered:
        for emo2 in ED_emotions_ordered:
            c1 = ED_mapping[emo1]
            c2 = ED_mapping[emo2]
            arr = [c1, c2, 0]
            d2_arr.append(arr)

    #print(d2_arr)

    d3_arr = []

    for emo1 in ED_emotions_ordered:
        for emo2 in ED_emotions_ordered:
            for emo3 in ED_emotions_ordered:
                c1 = ED_mapping[emo1]
                c2 = ED_mapping[emo2]
                c3 = ED_mapping[emo3]
                arr = [c1, c2, c3, 0]
                d3_arr.append(arr)

    #print(d3_arr)

    d4_arr = []

    for emo1 in ED_emotions_ordered:
        for emo2 in ED_emotions_ordered:
            for emo3 in ED_emotions_ordered:
                for emo4 in ED_emotions_ordered:
                    c1 = ED_mapping[emo1]
                    c2 = ED_mapping[emo2]
                    c3 = ED_mapping[emo3]
                    c4 = ED_mapping[emo4]
                    arr = [c1, c2, c3, c4, 0]
                    d4_arr.append(arr)

    #print(d4_arr)

    d1 = DiscreteDistribution(d1_dict)
    d2 = ConditionalProbabilityTable(d2_arr, [d1])
    d3 = ConditionalProbabilityTable(d3_arr, [d1, d2])
    d4 = ConditionalProbabilityTable(d4_arr, [d1, d2, d3])

    clf_unigram = MarkovChain([d1, d2])
    clf_bigram = MarkovChain([d1, d2, d3])
    clf_trigram = MarkovChain([d1, d2, d3, d4])

    print("started training markov chain ...")

    clf_unigram.fit(list(seq_list))
    clf_bigram.fit(list(seq_list))
    clf_trigram.fit(list(seq_list))

    print("finished training markov chain ...")

    # =================================================================================================    

    for dataset_name in ['ed', 'osed', 'os']:

        if dataset_name == 'ed':
            version_name = 'ed'
        elif dataset_name == 'os':
            version_name = 'os-dramatic-rest'
        elif dataset_name == 'osed':
            version_name = 'osed-dramatic'

        print(model_name, version_name)
        print("=====")

        if dataset_name == 'ed':
            # dialogue_id   context_emot    speaker_id  turn    uttr    eb_emot eb+_emot    speaker emotionality    listener neutrality dialogue score
            u_index = 4
            emo_index = 6
        else:
            u_index = 2
            emo_index = 4


        if dataset_name == 'ed':
            file_path = 's3://'+dataset_name+'-xenbot/6.new-version/test.csv'
        else:
            file_path = 's3://'+dataset_name+'-xenbot/6.new-version/'+version_name+'/test.csv'


        print(file_path)

        markov_unigram_accuracy = 0
        markov_bigram_accuracy = 0
        markov_trigram_accuracy = 0

        session_2 = boto3.session.Session(aws_access_key_id='AKIAJS7S46C2O7N2C62Q', aws_secret_access_key='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')
        #s3_2 = session_2.resource('s3')

        with smart_open.open(file_path, 'rt', encoding='utf-8', transport_params={'session': session_2}) as infile:

            with open('./model-'+model_name+'/markov-listener/'+version_name+'_test.csv', 'a') as emo_file:

                writer = csv.writer(emo_file, delimiter=',')
                writer.writerow(['dialogue', 'ground', 'markov-unigram', 'markov-bigram', 'markov-trigram'])

                readCSV = csv.reader(infile, delimiter=',')
                next(readCSV)
                # dialogue_id,   turn,    uttr,    eb_emot,  eb+_emot,    speaker emotionality,    listener neutrality,  dialogue score

                uttr_index = 0
                prev_id = ''
                count = 0
                current_dialog = []
                dial_count = 0
                N_examples = 0

                #f_enc = open('./'+version_name+'-'+file_name+'-encoded.txt', 'w')
                #f_uttr = open('./'+version_name+'-'+file_name+'-uttrs.txt', 'w')
                uttr_emots = []

                for row in readCSV:

                    cur_id = row[0]

                    if prev_id != '' and cur_id != prev_id:

                        if len(current_dialog) >= 2:

                            path = []
                            context = ''

                            dial_count += 1

                            first_turn = current_dialog[0]

                            dialog_id = first_turn[0]

                            #even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
                            #random_idx = random.choice(even_idx)

                            first_uttr = first_turn[u_index]
                            path.append(first_turn[emo_index])
                            context += first_uttr+ ' ('+first_turn[emo_index]+')\n'
                            
                            uttr_ids = tokenizer.encode(first_uttr)
                            inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]   # token ids of input
                            inp_seg_ids = [0] * (len(uttr_ids) + 2)    # 0 0 0 0 0 0 0 

                            break_point = len(current_dialog)

                            for k in range(1, len(current_dialog)):

                                uttr_ids = tokenizer.encode(current_dialog[k][u_index])
                                
                                #path.append(current_dialog[k][emo_index])
                                context += current_dialog[k][u_index]+ ' ('+current_dialog[k][emo_index]+')\n'
                                
                                tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                                tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

                                real_emot = current_dialog[k][emo_index] 

                                if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:

                                    inp_str = ','.join([str(x) for x in inp_ids])
                                    inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                                    tar_str = ','.join([str(x) for x in tar_ids])           # token ids of target
                                    tar_seg_str = ','.join([str(x) for x in tar_seg_ids])   # 1 1 1 1 1 1 1

                                    #f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                                    #    inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                                    #print("simple path: ", path[-1])
                                    markov_unigram = predict_morkov(path, clf_unigram)
                                    markov_bigram = predict_morkov(path, clf_bigram)
                                    markov_trigram = predict_morkov(path, clf_trigram)

                                    writer.writerow([context, real_emot, markov_unigram, markov_bigram, markov_trigram])

                                    if real_emot == markov_unigram:
                                        markov_unigram_accuracy += 1
                                    if real_emot == markov_bigram:
                                        markov_bigram_accuracy += 1
                                    if real_emot == markov_trigram:
                                        markov_trigram_accuracy += 1

                                    #print(real_emot, simple_emot[0], complex_emot[0])

                                    N_examples += 1

                                    path.append(real_emot)

                                else:
                                    break_point = k
                                    break

                                inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                                inp_seg_ids += tar_seg_ids

                        current_dialog = []

                    current_dialog.append(row)

                    prev_id = cur_id

                    count += 1

                if len(current_dialog) >= 2:

                    path = []
                    context = ''

                    dial_count += 1

                    first_turn = current_dialog[0]

                    dialog_id = first_turn[0]

                    #even_idx = [x * 2 + 1 for x in range(len(current_dialog) // 2)][:2]
                    #random_idx = random.choice(even_idx)

                    first_uttr = first_turn[u_index]
                    path.append(first_turn[emo_index])
                    context += first_uttr+ ' ('+first_turn[emo_index]+')\n'
                    
                    uttr_ids = tokenizer.encode(first_uttr)
                    inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]   # token ids of input
                    inp_seg_ids = [0] * (len(uttr_ids) + 2)    # 0 0 0 0 0 0 0 

                    break_point = len(current_dialog)

                    for k in range(1, len(current_dialog)):

                        uttr_ids = tokenizer.encode(current_dialog[k][u_index])
                        
                        #path.append(current_dialog[k][emo_index])
                        context += current_dialog[k][u_index]+ ' ('+current_dialog[k][emo_index]+')\n'
                        
                        tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                        tar_seg_ids = [k % 2] * (len(uttr_ids) + 2)

                        real_emot = current_dialog[k][emo_index] 

                        if len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length:

                            inp_str = ','.join([str(x) for x in inp_ids])
                            inp_seg_str = ','.join([str(x) for x in inp_seg_ids])
                            tar_str = ','.join([str(x) for x in tar_ids])           # token ids of target
                            tar_seg_str = ','.join([str(x) for x in tar_seg_ids])   # 1 1 1 1 1 1 1

                            #f_enc.write('{}\t{}\t{}\t{}\t{},{}\n'.format(
                            #    inp_str, inp_seg_str, tar_str, tar_seg_str, uttr_index, uttr_index + k))
                            #print("simple path: ", path[-1])
                            markov_unigram = predict_morkov(path, clf_unigram)
                            markov_bigram = predict_morkov(path, clf_bigram)
                            markov_trigram = predict_morkov(path, clf_trigram)

                            writer.writerow([context, real_emot, markov_unigram, markov_bigram, markov_trigram])

                            if real_emot == markov_unigram:
                                markov_unigram_accuracy += 1
                            if real_emot == markov_bigram:
                                markov_bigram_accuracy += 1
                            if real_emot == markov_trigram:
                                markov_trigram_accuracy += 1

                            #print(real_emot, simple_emot[0], complex_emot[0])

                            N_examples += 1

                            path.append(real_emot)

                        else:
                            break_point = k
                            break

                        inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                        inp_seg_ids += tar_seg_ids

                    

                current_dialog = []

        



        print('Number of dialogs:', dial_count)
        print('Number of examples:', N_examples)
        print("Markob unigram accuracy: ", float(markov_unigram_accuracy/N_examples))
        print("Markob bigram accuracy: ", float(markov_bigram_accuracy/N_examples))
        print("Markob trigram accuracy: ", float(markov_trigram_accuracy/N_examples))




