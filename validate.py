import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model import MEED, loss_function
from datasets import *


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
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 100
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'meed_os': 50, 'meed_os_osed': 6, 'meed_os_ed': 10}


def main(model_name, dataset):
    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'checkpoints_ebp/{}'.format(model_name)
    index_path = 'prediction/{}_2000.npy'.format(dataset)

    index = np.load(index_path)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        if dataset == 'os' or dataset == 'osed':
            data_path = '../../os/{}_emobert/test_human'.format(dataset)
            test_dataset, N = create_os_test_dataset(tokenizer, data_path, batch_size, max_length, index)
        elif dataset == 'ed':
            data_path = '../../empathetic_dialogues/data_ebp'
            _, _, test_dataset, _, N = create_ed_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, index)

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

        # Restore from the restore_epoch.
        ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
        print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

        @tf.function
        def valid_step(dist_inputs, loss_metric):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, tar_emot, False,
                    enc_padding_mask, combined_mask, dec_padding_mask)
                loss_ = loss_function(tar_real, pred)
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                return loss

            losses = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            loss_metric(mean_loss)

        my_loss.reset_states()
        for inputs in tqdm(test_dataset, total = 2000 // batch_size):
            valid_step(inputs, my_loss)

        return my_loss.result()


if __name__ == '__main__':
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
    f.close()
