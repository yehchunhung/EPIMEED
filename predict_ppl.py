import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from math import ceil
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model import MEED, loss_function
from datasets import *
from beam_search import beam_search

# changed

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
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

beam_width = 32
alpha = 1.0  # Decoding length normalization coefficient
n_gram = 4  # n-gram repeat blocking in beam search

num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 1  # For prediction, we always use batch size 1.
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

ED_emotions = ['afraid', 'angry','annoyed',
            'anticipating','anxious','apprehensive','ashamed','caring','confident','content','devastated','disappointed',
            'disgusted','embarrassed','excited','faithful','furious','grateful','guilty','hopeful','impressed','jealous',
            'joyful','lonely','nostalgic','prepared','proud','sad','sentimental','surprised','terrified','trusting',
            'agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']

#model_optimal_epoch = {'meed_os': 50, 'meed_os_osed': 6, 'meed_os_ed': 10}
model_optimal_epoch = {'xenbot-os-dramatic-rest': 50,
                        'xenbot-os-dramatic-rest-ed': 9,
                        'xenbot-os-dramatic-rest-osed-dramatic': 3
                        }


def evaluate(meed, inp, inp_seg, inp_emot, pred_tar_emot, tar_seg):
    enc_padding_mask = create_padding_mask(inp)
    enc_output = meed.encode(inp, inp_seg, inp_emot, False, enc_padding_mask)

    def iter_func(dec_inp, bw):
        enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
        dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
        dec_target_padding_mask = create_padding_mask(dec_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
        pred_tar_emot_tiled = tf.constant([pred_tar_emot] * dec_inp.shape[0])

        pred, attention_weights = meed.decode(enc_output_tiled, pred_tar_emot_tiled, dec_inp,
            dec_inp_seg, False, combined_mask, dec_padding_mask)
        return pred.numpy()

    result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)

    return result_seqs, log_probs

def main(model_name, dataset, version):
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        optimal_epoch = model_optimal_epoch[model_name]
        data_path = './datasets/'+dataset+'/test'
        #checkpoint_path = 'checkpoints_ebp/{}'.format(model_name)
        checkpoint_path = 'checkpoints/{}'.format(model_name)
        #pred_emot_path = 'prediction/{}/emo_pred_{}.csv'.format(dataset, model_name[5:])
        
        #pred_emot_path = 'emotions/emo_pred_os.csv'
        index_path = 'prediction/{}_500.npy'.format(dataset)

        if version == 'meed2':
            pred_emot_path = 'emotions/model-{}/meed2/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            #save_path = 'prediction/{}/meed2/{}-{}.csv'.format(dataset, model_name, version)
        else:
            pred_emot_path = 'emotions/model-{}/prob-sampled-heldout/{}_test_int.csv'.format(model_name.replace('xenbot-',''), dataset)
            #save_path = 'prediction/{}/argmax-heldout/{}-{}.csv'.format(dataset, model_name, version)


        print("data path", data_path)
        print("checkpoint path: ", checkpoint_path)
        print("optimal epoch: ", str(optimal_epoch))
        print("pred emot path: ", pred_emot_path)
        #print("save path: ", save_path)

        index = np.load(index_path)
            
        #test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length, index)
        test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, index)
        #test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length)

        # Define the model.
        meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

        # Build the model.
        build_meed_model(meed, max_length, vocab_size)
        print('Model has been built.')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        my_loss = tf.keras.metrics.Mean(name = 'my_loss')

        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        # Restore from the optimal_epoch.

        ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
        print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

        pred_emot_df = pd.read_csv(pred_emot_path).iloc[index]
        #pred_emot_df = pd.read_csv(pred_emot_path)
        print('pred_emot_df.shape = {}'.format(pred_emot_df.shape))

        #contexts = []
        #pred_ys = []
        #pred_emots = []
        #tar_ys = []
        #tar_emots = []


        #@tf.function
        def valid_step(meed, i, dist_inputs, version, loss_metric):
            def step_fn(meed, i, inputs, version):
                

                inp, inp_seg, inp_emot, _, tar_real, tar_seg, _ = inputs
                pred_emot = pred_emot_df.iloc[i][''+version]
                tar_emot = pred_emot_df.iloc[i]['ground']

                enc_padding_mask = create_padding_mask(inp)
                enc_output = meed.encode(inp, inp_seg, inp_emot, False, enc_padding_mask)

                '''
                def iter_func(dec_inp, bw):
                    enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
                    dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

                    look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
                    dec_target_padding_mask = create_padding_mask(dec_inp)
                    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

                    dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
                    pred_tar_emot_tiled = tf.constant([pred_emot] * dec_inp.shape[0])

                    pred, attention_weights = meed.decode(enc_output_tiled, pred_tar_emot_tiled, dec_inp,
                        dec_inp_seg, False, combined_mask, dec_padding_mask)
                    return pred.numpy()

                result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)


                print(log_probs)

                tar_pred_arr = np.ones((1, max_length), dtype = np.int32)
                for j in range(len(result_seqs[0])):
                    tar_pred_arr[0][j] = result_seqs[0][j]
                tar_pred = tf.convert_to_tensor(tar_pred_arr, dtype=tf.int32)
                '''

                bw = beam_width
                
                enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
                dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

                look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
                dec_target_padding_mask = create_padding_mask(dec_inp)
                combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

                dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
                

                pred_tar_emot_tiled = tf.constant([pred_emot] * dec_inp.shape[0])

                pred, attention_weights = meed.decode(enc_output=enc_output_tiled, pred_tar_emot=pred_tar_emot_tiled, 
                    tar=dec_inp, tar_seg=dec_inp_seg, training=False, look_ahead_mask=combined_mask, dec_padding_mask=dec_padding_mask)
                

                #decode(self, enc_output, pred_tar_emot, tar, tar_seg, training, look_ahead_mask, dec_padding_mask)

                print("tar_real = ", tar_real)
                print("tar pred = ", pred)
                print(tf.shape(tar_real), tf.shape(pred))

                loss_ = loss_function(tar_real, pred)
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)
                return loss

                '''inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, tar_emot, False,
                    enc_padding_mask, combined_mask, dec_padding_mask)
                loss_ = loss_function(tar_real, pred)
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                return loss'''

            losses = mirrored_strategy.experimental_run_v2(step_fn, args = (meed, i, dist_inputs,version,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            loss_metric(mean_loss)

        
        my_loss.reset_states()
        for (i, inputs) in tqdm(enumerate(test_dataset), total = ceil(N / batch_size)):
            valid_step(meed, i, inputs, version, my_loss)

        print(model_name, dataset, version)
        print(my_loss.result())

        return my_loss.result()

f = open('./auto-metrics/simple-complex-prob-sampled-heldout-ppl.txt', 'a')

if __name__ == '__main__':
    #for model_name in ['meed_os', 'meed_os_osed', 'meed_os_ed']:
    for model_name in ['xenbot-os-dramatic-rest', 'xenbot-os-dramatic-rest-ed', 'xenbot-os-dramatic-rest-osed-dramatic']:
        #for dataset in ['os', 'osed', 'ed']:
        #for dataset in ['os-dramatic-rest', 'osed-dramatic', 'ed']:
        for dataset in ['ed', 'osed-dramatic', 'os-dramatic-rest']:
            #for version in ['ground', 'simple', 'complex']:
            #for version in ['meed2']:
            for version in ['simple', 'complex']:
                f.write('Calculating ppl of {} on {} version {} (prob-sampled)\n\n'.format(model_name, dataset, version))
                print("***")
                print(model_name, dataset, version)
                print("***")
                loss_val = main(model_name, dataset, version)
                ppl = np.exp(loss_val)
                ppl = round(ppl, 4)
                f.write('Loss val: {}\n\n\n\n'.format(loss_val))
                f.write('PPL: {}\n\n\n\n'.format(ppl))

f.close()
