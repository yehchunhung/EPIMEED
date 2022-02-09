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
from model import MEED, MEEDPlus
from model_plain import PlainTransformer
from datasets import *
from beam_search import beam_search

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

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
num_comms = 7
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


def evaluate_meed2_plus(meed_plus, inp, inp_seg, inp_emot, inp_comm, pred_tar_emot, tar_comm, tar_seg):
    enc_padding_mask = create_padding_mask(inp)
    enc_output = meed_plus.encode(inp, inp_seg, inp_emot, inp_comm, False, enc_padding_mask)

    def iter_func(dec_inp, bw):
        enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
        dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
        dec_target_padding_mask = create_padding_mask(dec_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
        pred_tar_emot_tiled = tf.constant([pred_tar_emot] * dec_inp.shape[0])
        # pred_tar_comm_tiled = tf.constant([inp_comm] * dec_inp.shape[0])

        pred, attention_weights = meed_plus.decode(enc_output_tiled, pred_tar_emot_tiled, tar_comm, dec_inp,
            dec_inp_seg, False, combined_mask, dec_padding_mask)
        return pred.numpy()

    result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)

    return result_seqs, log_probs

                   
def evaluate_plain(plain, inp, inp_seg, tar_seg):
    enc_padding_mask = create_padding_mask(inp)
    enc_output = plain.encode(inp, inp_seg, False, enc_padding_mask)

    def iter_func(dec_inp, bw):
        enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
        dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
        dec_target_padding_mask = create_padding_mask(dec_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]

        pred, attention_weights = plain.decode(enc_output_tiled, dec_inp, dec_inp_seg, False,
                                               combined_mask, dec_padding_mask)
        return pred.numpy()

    result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)

    return result_seqs, log_probs

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
    index_path = 'prediction-listener/{}.npy'.format(dataset)
    # index_path = 'prediction-listener/red+_mask.npy' # use the fixed indices

    print("data path", data_path)
    print("checkpoint path: ", checkpoint_path)
    print("optimal epoch: ", str(optimal_epoch))
    #print("pred emot path: ", pred_emot_path)
    print("save path: ", save_path)

    index = np.load(index_path)
        
    # test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length, index)
    # test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, index, contain_comm = True)
    # test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, index, contain_mask = True)
    test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, index, contain_comm = True, contain_mask = True)
    #test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length)

    # Define the model.
    if version == 'plain':
        plain = PlainTransformer(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size)
        
        build_plain_model(plain, max_length, vocab_size)
    elif version == 'meed2':
        meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)
        
        build_meed_model(meed, max_length, vocab_size)
    else:
        meed_plus = MEEDPlus(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions, num_comms)

        # Build the model.
        build_meed_plus_model(meed_plus, max_length, vocab_size)
        
    print('Model has been built.')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    my_loss = tf.keras.metrics.Mean(name = 'my_loss')

    # Define the checkpoint manager.
    if version == 'plain':
        ckpt = tf.train.Checkpoint(model = plain, optimizer = optimizer)
    elif version == 'meed2':
        ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
    else:
        ckpt = tf.train.Checkpoint(model = meed_plus, optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal_epoch.
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

    if version != 'plain':
        pred_emot_df = pd.read_csv(pred_emot_path).iloc[index]
        #pred_emot_df = pd.read_csv(pred_emot_path)
        print('pred_emot_df.shape = {}'.format(pred_emot_df.shape))

    contexts = []
    pred_ys = []
    pred_emots = []
    tar_ys = []
    tar_emots = []
    context_tokens = []

    for (i, inputs) in tqdm(enumerate(test_dataset), total = ceil(N / batch_size)):
        # # for meed2
        # inp, inp_seg, inp_emot, _, tar_real, tar_seg, _, tar_mask = inputs
        
        # # for meed2+ (not considering mask)
        # inp, inp_seg, inp_emot, inp_comm, _, tar_real, tar_seg, _, tar_comm = inputs
        
        # for meed2++ (considering mask)
        inp, inp_seg, inp_emot, inp_comm, _, tar_real, tar_seg, _, tar_comm, tar_mask = inputs
        
        if version != 'plain':
            if version == 'meed2':
                pred_emot = pred_emot_df.iloc[i][''+version]
                tar_emot = pred_emot_df.iloc[i]['ground']
                #pred_emot = pred_emot_df.iloc[i]['y_pred']
                #tar_emot = pred_emot_df.iloc[i]['y_true']
                pred_emots.append(ED_emotions[pred_emot])
                tar_emots.append(ED_emotions[tar_emot])
            else:
                pred_emot = pred_emot_df.iloc[i][''+version[:-1]]
                tar_emot = pred_emot_df.iloc[i]['ground']
                #pred_emot = pred_emot_df.iloc[i]['y_pred']
                #tar_emot = pred_emot_df.iloc[i]['y_true']
                pred_emots.append(ED_emotions[pred_emot])
                tar_emots.append(ED_emotions[tar_emot])
            

        context = tokenizer.decode(inp[0].numpy().tolist())
        context = ['--> {}'.format(u.strip()) for u in context if '<pad>' not in u]
        
        # remove adjacent duplicate turns (this happens sometimes in some dialogs)
        context = [elem for i, elem in enumerate(context) if i == 0 or context[i-1] != elem] 
        
        context = '\n'.join(context)
        contexts.append(context)
        context_tokens.append(inp[0].numpy().tolist())

        if version == 'plain':
            tar_preds, log_probs = evaluate_plain(plain, inp, inp_seg, tar_seg)
        elif version == 'meed2':
            tar_preds, log_probs = evaluate(meed, inp, inp_seg, inp_emot, pred_emot, tar_seg)
        else:
            tar_preds, log_probs = evaluate_meed2_plus(meed_plus, inp, inp_seg, inp_emot, inp_comm, pred_emot, tar_comm, tar_seg)
        #evaluate(meed, inp, inp_seg, inp_emot, pred_tar_emot, tar_seg)

        tar_pred_dec = tokenizer.decode(tar_preds[0])  # top candidate of beam search
        pred_y = tar_pred_dec[0].strip() if len(tar_pred_dec) > 0 else ''
        pred_ys.append(pred_y)


        tar_y = tokenizer.decode([SOS_ID] + tar_real[0].numpy().tolist())[0].strip()
        tar_ys.append(tar_y)

        #print(context, ' || ', pred_y, ' || ', pred_emot, ' || ', tar_y, ' || ', tar_emot)

    print('Saving the prediction results...')
    if version == 'plain':
        data = {'context': contexts, 'pred_y': pred_ys, 'tar_y': tar_ys}
    else:
        data = {'context': contexts, 'pred_y': pred_ys, 'pred_emot': pred_emots,
            'tar_y': tar_ys, 'tar_emot': tar_emots}
    pd.DataFrame(data).to_csv(save_path, index = False)
    np.save('context_tokens', context_tokens)


if __name__ == '__main__':
    #for model_name in ['meed_os', 'meed_os_osed', 'meed_os_ed']:
    for model_name in ['xenbot-os-dramatic-rest-red+']: # ['plain-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red+', 'xenbot-os-dramatic-rest-red+_mask_']
        #for dataset in ['os', 'osed', 'ed']:
        #for dataset in ['os-dramatic-rest', 'osed-dramatic', 'ed']:
        for dataset in ['red+_mask']:
            #for version in ['ground', 'simple', 'complex']:
            #for version in ['meed2']:
            #for version in ['simple', 'complex']:
            
            #for version in ['plain', 'ground', 'meed2', 'simple', 'complex']:  # argmax models
            for version in ['meed2+']: # ['plain', 'meed2', 'meed2+']
                
                print("***")
                print(model_name, dataset, version)
                print("***")
                main(model_name, dataset, version)


