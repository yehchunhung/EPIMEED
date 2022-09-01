import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
#from model import MEED, loss_function
from model_plain import PlainTransformer, loss_function
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

#num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = int(15e6)
batch_size = 512
num_epochs = 50
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

#checkpoint_path = 'checkpoints_ebp/meed_os'
#log_path = 'log_ebp/meed_os.log'
#data_path = '../os/os_emobert'

checkpoint_path = './checkpoints/plain-os-dramatic-rest'
log_path = './log/plain-os-dramatic-rest.log'
data_path = './datasets/os-dramatic-rest'


def main():
    if not exists('log'):
        mkdir('log')
    f = open(log_path, 'a', encoding = 'utf-8')

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_dataset, val_dataset, test_dataset = create_datasets(tokenizer,
            data_path, buffer_size, batch_size, max_length)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
        test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

        # Define the model.
        plain = PlainTransformer(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size)

        # Build the model.
        build_plain_model(plain, max_length, vocab_size)
        print('Model has been built.')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        test_loss = tf.keras.metrics.Mean(name = 'test_loss')


        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = plain, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            f.write('Latest checkpoint restored!!\n')

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                with tf.GradientTape() as tape:
                    pred, _ = plain(inp, inp_seg, tar_inp, tar_seg,
                                           True,
                                           enc_padding_mask,
                                           combined_mask,
                                           dec_padding_mask)
                    loss_ = loss_function(tar_real, pred)
                    loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                gradients = tape.gradient(loss, plain.trainable_variables)    
                optimizer.apply_gradients(zip(gradients, plain.trainable_variables))
                return loss

            #losses = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            losses = mirrored_strategy.experimental_run_v2(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            '''
            def step_fn(inputs):
                inp, inp_seg, tar_inp, tar_real, tar_seg = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                with tf.GradientTape() as tape:
                    predictions, _ = plain(inp, inp_seg, tar_inp, tar_seg,
                                           True,
                                           enc_padding_mask,
                                           combined_mask,
                                           dec_padding_mask)
                    losses_per_examples = loss_function(tar_real, predictions)
                    loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                gradients = tape.gradient(loss, plain.trainable_variables)    
                optimizer.apply_gradients(zip(gradients, plain.trainable_variables))
                return loss

            losses_per_replica = mirrored_strategy.experimental_run_v2(
                step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)
            '''

            train_loss(mean_loss)
            return mean_loss

        @tf.function
        def valid_step(dist_inputs, loss_metric):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                pred, _ = plain(inp, inp_seg, tar_inp, tar_seg,
                                       False,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
                loss_ = loss_function(tar_real, pred)
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                return loss

            #losses = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            losses = mirrored_strategy.experimental_run_v2(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            loss_metric(mean_loss)


            '''
            def step_fn(inputs):
                inp, inp_seg, tar_inp, tar_real, tar_seg = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                predictions, _ = plain(inp, inp_seg, tar_inp, tar_seg,
                                       False,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
                losses_per_examples = loss_function(tar_real, predictions)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                return loss

            losses_per_replica = mirrored_strategy.experimental_run_v2(
                step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)

            loss_metric(mean_loss)
            '''


        # Start training
        # Start training
        for epoch in range(num_epochs):
            start = time.time()

            train_loss.reset_states()

            for (batch, inputs) in enumerate(train_dataset):
                loss = train_step(inputs)
                mean_loss = train_loss.result()
                print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                    epoch + 1, batch, mean_loss, loss))
                f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                    epoch + 1, batch, mean_loss, loss))

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            epoch_loss = train_loss.result()
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss))
            f.write('Epoch {} Loss {:.4f}\n'.format(epoch + 1, epoch_loss))

            current_time = time.time()
            print('Time taken for 1 epoch: {} secs'.format(current_time - start))
            f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

            valid_loss.reset_states()
            for inputs in val_dataset:
                valid_step(inputs, valid_loss)
            epoch_val_loss = valid_loss.result()
            print('Epoch {} Valid loss {:.4f}'.format(epoch + 1, epoch_val_loss))
            f.write('Epoch {} Valid loss {:.4f}\n'.format(epoch + 1, epoch_val_loss))

            test_loss.reset_states()
            for inputs in test_dataset:
                valid_step(inputs, test_loss)
            epoch_test_loss = test_loss.result()
            print('Epoch {} Test loss {:.4f}\n'.format(epoch + 1, epoch_test_loss))
            f.write('Epoch {} Test loss {:.4f}\n\n'.format(epoch + 1, epoch_test_loss))

    f.close()

if __name__ == '__main__':
    main()
