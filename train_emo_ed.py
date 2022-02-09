import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model_emo_pred import EmotionPredictor, loss_function
from datasets import *


# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
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
buffer_size = 100000
batch_size = 256
num_epochs = 10
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

'''restore_epoch = 3
restore_path = 'checkpoints_ebp/emo_pred_os'
checkpoint_path = 'checkpoints_ebp/emo_pred_ed_roberta_ebp_weighted'
log_path = 'log_ebp/emo_pred_ed_roberta_ebp_weighted.log'
data_path = '../ed/data_ebp'
emot_freq_path = 'balance_emot/emobert+/ed_ebp_dist.csv'''


restore_epoch = 2
restore_path = 'emo_checkpoints/os-dramatic-rest'
checkpoint_path = './emo_checkpoints/os-dramatic-rest-ed'
log_path = 'emo_log/os-dramatic-rest-ed.log'
data_path = './datasets/ed'


'''def get_class_weights():
    class_weights = np.zeros(num_emotions, dtype = np.float32)
    with open(emot_freq_path, 'r') as f:
        for line in f:
            emot_id, _, freq = line.strip().split(',')
            class_weights[int(emot_id)] = float(freq)
    class_weights = np.min(class_weights) / class_weights
    return class_weights'''

def main():
    f = open(log_path, 'a', encoding = 'utf-8')

    #print('Reading the class weights...')
    #class_weights = get_class_weights()

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_dataset, val_dataset, test_dataset = create_datasets(tokenizer,
            data_path, buffer_size, batch_size, max_length)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
        test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

        # Define the model.
        emotion_predictor = EmotionPredictor(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

        # Build the model and initialize weights from RoBERTa
        build_emo_pred_model(emotion_predictor, max_length, vocab_size)
        emotion_predictor.embedder.load_weights('./weights/roberta2emo_pred_embedder_ebp.h5')
        emotion_predictor.encoder.load_weights('./weights/roberta2emo_pred_encoder.h5')
        print('Weights initialized from RoBERTa.')
        f.write('Weights initialized from RoBERTa.\n')
        # print('Model has been built.')
        # f.write('Model has been built.\n')

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
        test_loss = tf.keras.metrics.Mean(name = 'test_loss')


        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = emotion_predictor, optimizer = optimizer)
        ckpt_manager_save = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
        ckpt_manager_restore = tf.train.CheckpointManager(ckpt, restore_path, max_to_keep = None)

        # # Restore from the restore_epoch.
        ckpt.restore(ckpt_manager_restore.checkpoints[restore_epoch - 1]).expect_partial()
        print('Checkpoint {} restored!!'.format(ckpt_manager_restore.checkpoints[restore_epoch - 1]))

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, _, _, _, tar_emot = inputs
                #tar_emot_one_hot = tf.one_hot(tar_emot, num_emotions)

                enc_padding_mask = create_padding_mask(inp)

                with tf.GradientTape() as tape:
                    pred_emot = emotion_predictor(inp, inp_seg, inp_emot, True, enc_padding_mask)
                    #losses_per_examples = loss_function_weighted(tar_emot_one_hot, pred_emot, class_weights)
                    losses_per_examples = loss_function(tar_emot, pred_emot)
                    loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                gradients = tape.gradient(loss, emotion_predictor.trainable_variables)    
                optimizer.apply_gradients(zip(gradients, emotion_predictor.trainable_variables))
                return loss

            losses_per_replica = mirrored_strategy.experimental_run_v2(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                losses_per_replica, axis = None)

            train_loss(mean_loss)
            return mean_loss

        @tf.function
        def valid_step(dist_inputs, loss_metric):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, _, _, _, tar_emot = inputs

                enc_padding_mask = create_padding_mask(inp)

                pred_emot = emotion_predictor(inp, inp_seg, inp_emot, False, enc_padding_mask)
                losses_per_examples = loss_function(tar_emot, pred_emot)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                return loss

            losses_per_replica = mirrored_strategy.experimental_run_v2(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                losses_per_replica, axis = None)

            loss_metric(mean_loss)


        # Start training
        for epoch in range(num_epochs):
            start = time.time()

            train_loss.reset_states()

            for (batch, inputs) in enumerate(train_dataset):
                current_loss = train_step(inputs)
                current_mean_loss = train_loss.result()
                print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))
                f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))

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
            print('Epoch {} Validation loss {:.4f}'.format(epoch + 1, epoch_val_loss))
            f.write('Epoch {} Validation loss {:.4f}\n'.format(epoch + 1, epoch_val_loss))

            test_loss.reset_states()
            for inputs in test_dataset:
                valid_step(inputs, test_loss)
            epoch_test_loss = test_loss.result()
            print('Epoch {} Test loss {:.4f}\n'.format(epoch + 1, epoch_test_loss))
            f.write('Epoch {} Test loss {:.4f}\n\n'.format(epoch + 1, epoch_test_loss))

    f.close()

if __name__ == '__main__':
    main()
