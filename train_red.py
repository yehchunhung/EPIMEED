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
num_comms = 7 # the type of communication level outcomes: 0, 1, 2, 3, 4, 5, 6
max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 512
num_epochs = 50
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

restore_epoch = 50
restore_path = './checkpoints/xenbot-os-dramatic-rest'
checkpoint_path = './checkpoints/xenbot-os-dramatic-rest-red+_mask_'
log_path = './log/xenbot-os-dramatic-rest-red+_mask_.log'
data_path = './datasets/red+_mask'


def main():
    f = open(log_path, 'a', encoding = 'utf-8')

    mirrored_strategy = tf.distribute.MirroredStrategy(devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"], 
                                                       cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())

    with mirrored_strategy.scope():
        # train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, contain_comm = True)
        train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, contain_mask = True)
        # train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size, batch_size, max_length, contain_comm = True, contain_mask = True)
        # create_datasets(tokenizer, path, buffer_size, batch_size, max_length)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
        test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

        # Define the model.
        # for meed2
        meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)
        
        # # for meed2+
        # meed_plus = MEEDPlus(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        #     layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions, num_comms)
        
        # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
        build_meed_model(meed, max_length, vocab_size)
        
        # # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
        # build_meed_plus_model(meed_plus, max_length, vocab_size)
        
        print('Model has been built.')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        test_loss = tf.keras.metrics.Mean(name = 'test_loss')


        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
        # ckpt = tf.train.Checkpoint(model = meed_plus, optimizer = optimizer)
        
        ckpt_manager_save = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
        ckpt_manager_restore = tf.train.CheckpointManager(ckpt, restore_path, max_to_keep = None)

        # Restore from the restore_epoch.
        ckpt.restore(ckpt_manager_restore.checkpoints[restore_epoch - 1]).expect_partial()
        print('Checkpoint {} restored!!'.format(ckpt_manager_restore.checkpoints[restore_epoch - 1]))

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager_save.latest_checkpoint:
            ckpt.restore(ckpt_manager_save.latest_checkpoint)
            print('Latest checkpoint restored!!')
            f.write('Latest checkpoint restored!!\n')

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                # # for meed2 (not considering mask)
                # inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs
                
                # for meed2 (considering mask)
                inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot, tar_mask = inputs
                
                # # for meed2+ (not considering mask)
                # inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_real, tar_seg, tar_emot, tar_comm = inputs
                
                # # for meed2+ (considering mask)
                # inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_real, tar_seg, tar_emot, tar_comm, tar_mask = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

                with tf.GradientTape() as tape:
                    # for meed2
                    pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, tar_emot, True,
                        enc_padding_mask, combined_mask, dec_padding_mask)
                    # # for meed2+
                    # pred, _ = meed_plus(inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_seg, tar_emot, tar_comm, True,
                    #     enc_padding_mask, combined_mask, dec_padding_mask)
                    
                    
                    # loss_ = loss_function(tar_real, pred)
                    loss_ = loss_function_(tar_real, pred, tar_mask) # considering mask
                    loss = tf.reduce_sum(loss_) * (1.0 / batch_size)


                gradients = tape.gradient(loss, meed.trainable_variables)    
                optimizer.apply_gradients(zip(gradients, meed.trainable_variables))
                return loss

            losses = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            train_loss(mean_loss)
            return mean_loss

        @tf.function
        def valid_step(dist_inputs, loss_metric):
            def step_fn(inputs):
                # # for meed2 (not considering mask)
                # inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot = inputs
                
                # for meed2 (considering mask)
                inp, inp_seg, inp_emot, tar_inp, tar_real, tar_seg, tar_emot, tar_mask = inputs
                
                # # for meed2+ (not considering mask)
                # inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_real, tar_seg, tar_emot, tar_comm = inputs
                
                # # for meed2+ (considering mask)
                # inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_real, tar_seg, tar_emot, tar_comm, tar_mask = inputs

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
                
                # for meed2
                pred, _ = meed(inp, inp_seg, inp_emot, tar_inp, tar_seg, tar_emot, False,
                    enc_padding_mask, combined_mask, dec_padding_mask)
                
                # # for meed2+
                # pred, _ = meed_plus(inp, inp_seg, inp_emot, inp_comm, tar_inp, tar_seg, tar_emot, tar_comm, False,
                #     enc_padding_mask, combined_mask, dec_padding_mask)
                
                # loss_ = loss_function(tar_real, pred)
                loss_ = loss_function_(tar_real, pred, tar_mask) # considering mask
                loss = tf.reduce_sum(loss_) * (1.0 / batch_size)

                return loss

            losses = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis = None)

            loss_metric(mean_loss)


        # Start training
        #for epoch in range(10, 10 + num_epochs):
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

            ckpt_save_path = ckpt_manager_save.save()
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
