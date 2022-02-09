import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model import MEED, MEEDPlus, loss_function, loss_function_
from datasets import *

import pandas as pd
from math import ceil
import csv

from model_plain import PlainTransformer, loss_function

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

# Some hyper-parameters
num_layers = 4
d_model = 300
num_heads = 6
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 102
type_vocab_size = 2  # Segments

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

num_emotions = 41
num_comms = 7
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 1
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

#model_optimal_epoch = {'meed_os': 50, 'meed_os_osed': 6, 'meed_os_ed': 10}
model_optimal_epoch = {'xenbot-os-dramatic-rest-red': 19, # vanilla meed2 on RED
                        'xenbot-os-dramatic-rest-red+': 14, # meed2+ adding comm level as the additional embedding 
                        'xenbot-os-dramatic-rest-red+_mask': 8, # meed2 with new loss function considering mask; mask value: 0, 1
                        'xenbot-os-dramatic-rest-red+_mask_': 8, # meed2 with new loss function considering mask; mask value: 0 ~ 6
                        'xenbot-os-dramatic-rest': 50,
                        'xenbot-os-dramatic-rest-ed': 9,
                        'xenbot-os-dramatic-rest-osed-dramatic': 3
                        }

plain_model_optimal_epoch = {'plain-os-dramatic-rest-red': 23,
                             'plain-os-dramatic-rest-red+_mask': 9, # plain with new loss function considering mask; mask value: 0, 1
                             'plain-os-dramatic-rest': 50,
                             'plain-os-dramatic-rest-ed': 9,
                             'plain-os-dramatic-rest-osed-dramatic': 3
                             }


def main(model_name, dataset, version):
    
    if version == 'plain':
        model_name = model_name.replace('xenbot', 'plain')
        optimal_epoch = plain_model_optimal_epoch[model_name]
        checkpoint_path = 'checkpoints/{}'.format(model_name)
        #pred_emot_path = 'emotions/model-{}/meed2/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
        save_path = 'prediction-listener/{}/plain/{}-{}.csv'.format(dataset, model_name, version)
        
    else:
        optimal_epoch = model_optimal_epoch[model_name]
        
        #checkpoint_path = 'checkpoints_ebp/{}'.format(model_name)
        checkpoint_path = 'checkpoints/{}'.format(model_name)
        #pred_emot_path = 'prediction/{}/emo_pred_{}.csv'.format(dataset, model_name[5:])
        
        #pred_emot_path = 'emotions/emo_pred_os.csv'
        if version == 'meed2':
            # using meed2's response emotion predictor
            pred_emot_path = 'emotions/model-{}/meed2/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            
            # # using meed2+'s response emotion predictor
            # pred_emot_path = 'emotions/model-{}/meed2+/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            
            save_path = 'prediction-listener/{}/meed2/{}-{}.csv'.format(dataset, model_name, version)
            
        elif version == 'meed2+':
            # using meed2's response emotion predictor
            # pred_emot_path = 'emotions/model-{}/meed2/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            
            # using meed2+'s response emotion predictor
            pred_emot_path = 'emotions/model-{}/meed2+/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            
            save_path = 'prediction-listener/{}/meed2+/{}-{}.csv'.format(dataset, model_name, version)
            
        elif version == 'alexa':
            pred_emot_path = 'emotions/model-{}/alexa/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            save_path = 'prediction-listener/{}/alexa/{}-{}.csv'.format(dataset, model_name, version)
            
        else:
            pred_emot_path = 'emotions/model-{}/prob-sampled-heldout/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            save_path = 'prediction-listener/{}/prob-sampled-heldout/{}-{}.csv'.format(dataset, model_name, version)
    
    data_path = './datasets/'+dataset+'/test'
    index_path = 'prediction-listener/{}_all.npy'.format(dataset)


    index = np.load(index_path)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        '''if dataset == 'os' or dataset == 'osed':
            data_path = '../../os/{}_emobert/test_human'.format(dataset)
            test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length, index)
        elif dataset == 'ed':
            data_path = '../../empathetic_dialogues/data_ebp'
            _, _, test_dataset, _, N = create_ed_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, index)'''

        test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, index, contain_comm = True, contain_mask = True)
        
        if version != 'plain':
            pred_emot_df = pd.read_csv(pred_emot_path).iloc[index]
            
            if version == 'meed2':
                # Define the model.
                meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                    layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

                # Build the model.
                build_meed_model(meed, max_length, vocab_size)
            elif version == 'meed2+':
                # Define the model.
                meed = MEEDPlus(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                                layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions, num_comms)

                # Build the model.
                build_meed_plus_model(meed, max_length, vocab_size)

        else:

            # Define the model.
            plain = PlainTransformer(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size)
            build_plain_model(plain, max_length, vocab_size)
        
        print('Model has been built.')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        my_loss = tf.keras.metrics.Mean(name = 'my_loss')

        # Define the checkpoint manager.
        if version != 'plain':
            ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
        else:
            ckpt = tf.train.Checkpoint(model = plain, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        # Restore from the restore_epoch.
        ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
        print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

        #@tf.function
        def valid_step(dist_inputs, loss_metric, pred_emot):
            def step_fn(inputs, pred_emot):
                # for meed2+ (considering mask)
                inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_real, tar_seg, tar_emot, tar_comm, tar_mask = inputs

                #print("inp emot = ", inp_emot.numpy())
                #print("tar emot = ", tar_emot.numpy())
                #print("tar emot = ", pred_emot)
                if pred_emot != None:
                    pred_emot_tensor = tf.convert_to_tensor([pred_emot], dtype=tf.int32)

                #print("tar emot = ", tar_emot)
                #print("tar emot = ", pred_emot_tensor)

                #inp emot =  Tensor("dist_inputs_2:0", shape=(100, 100), dtype=int32)
                #tar emot =  Tensor("dist_inputs_6:0", shape=(100,), dtype=int32)
                #tar emot =  38

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                #pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, tar_emot, False, enc_padding_mask, combined_mask, dec_padding_mask)
                if pred_emot != None:
                    if version == 'meed2':
                        pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, pred_emot_tensor, False, enc_padding_mask, combined_mask, dec_padding_mask)
                    elif version == 'meed2+':
                        pred, _ = meed(inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_seg, tar_emot, tar_comm, False, enc_padding_mask, combined_mask, dec_padding_mask)
                    
                else:
                    pred, _ = plain(inp, inp_seg, tar_inp, tar_seg, False, enc_padding_mask, combined_mask, dec_padding_mask)
                #print("tar_real = ", tar_real)
                #print("pred = ", pred)
                
                # loss_ = loss_function(tar_real, pred)
                loss_ = loss_function_(tar_real, pred, tar_mask) # considering mask
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                return loss

            losses = mirrored_strategy.run(step_fn, args = (dist_inputs,pred_emot,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            loss_metric(mean_loss)

        my_loss.reset_states()

        #for inputs in tqdm(test_dataset, total = N // batch_size):
        for (i, inputs) in tqdm(enumerate(test_dataset), total = ceil(N / batch_size)):
            if version != 'plain':
                pred_emot = pred_emot_df.iloc[i]['meed2']
                #tar_emot = pred_emot_df.iloc[i]['ground']
                valid_step(inputs, my_loss, pred_emot)
            else:
                valid_step(inputs, my_loss, None)

        return my_loss.result()

if __name__ == '__main__':

    #with open('./auto-metrics/meed2-simple-complex-prob-sampled-ppl.csv', 'a') as infile:

    #    writer = csv.writer(infile, delimiter=',')
    #    writer.writerow(['model', 'dataset', 'version', 'loss val', 'PPL'])

    f = open('auto-metrics/PPL-listener-4turns-prob_sampled_new.txt', 'a')
    #f.write('model', 'dataset', 'version', 'loss val', 'PPL\n')

    for model_name in ['xenbot-os-dramatic-rest-red+_mask_']: # ['plain-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red+', 'xenbot-os-dramatic-rest-red+_mask_']
        for dataset in ['red+_mask']:
            for version in ['meed2+']: # ['plain', 'meed2', 'meed2+']

                print(model_name, dataset, version)

                loss_val = main(model_name, dataset, version)
                ppl = np.exp(loss_val)
                
                #loss_val = round(loss_val, 4)
                #ppl = round(ppl, 4)

                print("loss val = ", loss_val)
                print("ppl = ", ppl)

                print(model_name, dataset, version)

                if version == 'simple' or version == 'complex':
                    #writer.writerow([model_name, dataset, version+" (prob-sampled)", loss_val, ppl])
                    #f.write('model', 'dataset', 'version', 'loss val', 'PPL\n')
                    f.write('Calculating ppl of {} on {} version {} (prob-sampled)\n\n'.format(model_name, dataset, version))
                else:
                    #writer.writerow([model_name, dataset, version, loss_val, ppl])
                    f.write('Calculating ppl of {} on {} version {}\n\n'.format(model_name, dataset, version))

                f.write('ppl = {}\n'.format(ppl))
                f.write('loss val = {}\n\n\n\n'.format(loss_val))


'''if __name__ == '__main__':
    f = open('../meed/prediction/scores/ppl_meed.csv', 'w')
    f.write('model,os,osed,ed\n')
    for model_name in ['meed_os', 'meed_os_osed', 'meed_os_ed']:
        f.write(model_name)
        for dataset in ['os', 'osed', 'ed']:
            print('Calculating ppl of {} on {}...'.format(model_name, dataset))
            loss_val = main(model_name, dataset)
            ppl = np.exp(loss_val)
            f.write(',{}'.format(ppl))
        f.write('\n')
    f.close()'''




