import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from tqdm import tqdm
from math import ceil
from sklearn.metrics import precision_recall_fscore_support
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model_emo_pred import EmotionPredictor, EmotionPredictorPlus, loss_function
from datasets import *


configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


# Some hyper-parameters
'''num_layers = 4
d_model = 300
num_heads = 6
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 102
type_vocab_size = 2  # Segments'''

num_layers = 12
d_model = 768
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

'''num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 256
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6'''

num_emotions = 41
num_comms = 7
max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 256
num_epochs = 10
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'red+': 3, 'red': 2, 'os-dramatic-rest': 2, 'os-dramatic-rest-ed': 1, 'os-dramatic-rest-osed-dramatic': 1} 
log_path = 'emotions/meed2_emo_pred_red+.log'


def main(model_name, dataset, f_out):
    f_out.write('{} predicting on {}...\n'.format(model_name, dataset))

    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'emo_checkpoints/{}'.format(model_name) # it's where your newly trained models will be saved after each epoch.
    save_path = './emotions/model-'+model_name+'/meed2+/'+dataset+'_test_int.csv' # a variable specifying the file path to the prediction result 
    #index_path = './prediction/{}_500.npy'.format(dataset)
    data_path = './datasets/'+dataset+'/test'

    '''if dataset == 'os' or dataset == 'osed':
        data_path = '../../os/{}_emobert/test_human'.format(dataset)
        test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length)
    elif dataset == 'ed':
        data_path = '../../empathetic_dialogues/data_ebp'
        _, _, test_dataset, _, N = create_ed_datasets(tokenizer, data_path, buffer_size, batch_size, max_length)'''
    
    #index = np.load(index_path)
    
    # # for meed2
    # test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length)
    
    # for meed2+
    test_dataset, N = create_test_dataset(tokenizer, data_path, batch_size, max_length, None, contain_comm = True)

    # Define the model.
    # for meed2
    emotion_predictor = EmotionPredictor(num_layers, d_model, num_heads, dff, hidden_act,
        dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)
    
    # # for meed2+
    # emotion_predictor = EmotionPredictorPlus(num_layers, d_model, num_heads, dff, hidden_act,
    #     dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions, num_comms)

    # Build the model.
    # for meed2
    build_emo_pred_model(emotion_predictor, max_length, vocab_size)
    
    # # for meed2+
    # build_emo_pred_plus_model(emotion_predictor, max_length, vocab_size)
    
    f_out.write('Model has been built.\n')
    # for meed2
    emotion_predictor.embedder.load_weights('weights/roberta2emo_pred_embedder_ebp.h5') # if max length is 100, use the original
    emotion_predictor.encoder.load_weights('weights/roberta2emo_pred_encoder.h5')
        
    # # for meed2+
    # emotion_predictor.embedder.load_weights('weights/roberta2emo_pred_embedder_ebp_red+.h5') 
    # emotion_predictor.encoder.load_weights('weights/roberta2emo_pred_encoder_red+.h5')
    print('Weights initialized from RoBERTa.')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emotion_predictor, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal epoch. 
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    f_out.write('Checkpoint {} restored.\n'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

    y_true = []
    y_pred = []
    
    for inputs in tqdm(test_dataset, total = ceil(N / batch_size)):
        # # for meed2
        # inp, inp_seg, inp_emot, _, tar_real, _, tar_emot = inputs
        
        # for meed2+
        inp, inp_seg, inp_emot, inp_comm, _, tar_real, _, tar_emot, tar_comm = inputs
        enc_padding_mask = create_padding_mask(inp)
        
        # for meed2
        pred_emot = emotion_predictor(inp, inp_seg, inp_emot, False, enc_padding_mask)
        
        # # for meed2+
        # pred_emot = emotion_predictor(inp, inp_seg, inp_emot, inp_comm, False, enc_padding_mask)
        
        pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
        y_true += tar_emot.numpy().tolist()
        y_pred += pred_emot.tolist()
        
    

    f_out.write('Number of testing examples: {}\n'.format(len(y_true)))
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f_out.write('Accuracy\t{:.4f}\n'.format(acc))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro')
    f_out.write('Macro\tP: {:.4f}, R: {:.4f}, F: {:.4f}\n'.format(p, r, f))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    f_out.write('Weighted\tP: {:.4f}, R: {:.4f}, F: {:.4f}\n'.format(p, r, f))

    f_out.write('Saving the prediction results...\n\n')
    prediction = {'ground': y_true, 'meed2': y_pred}
    pd.DataFrame(prediction).to_csv(save_path, index = False)


if __name__ == '__main__':
    f_out = open(log_path, 'a')
    #for model_name in ['emo_pred_os', 'emo_pred_os_osed', 'emo_pred_os_ed']:
    #for model_name in ['os-dramatic-rest', 'os-dramatic-rest-ed', 'os-dramatic-rest-osed-dramatic']:
    for model_name in ['red+']: # ['red', 'red+']
        #for dataset in ['os-dramatic-rest', 'ed', 'osed-dramatic']:
        for dataset in ['red+_mask']: # focus on red+_mask now
            print(model_name, dataset)
            print("=====")
            main(model_name, dataset, f_out)
    f_out.close()



