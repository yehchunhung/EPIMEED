import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import *
from model_utils import *
from model_emo_pred import EmotionPredictor, loss_function
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

num_emotions = 64
max_length = 100  # Maximum number of tokens
buffer_size = int(15e6)
batch_size = 256
num_epochs = 5
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

# restore_epoch = 4
checkpoint_path = 'checkpoints/emo_pred_osed_from_3'
data_path = '../os_ed/test'


def main():
    # f = open('log/emo_pred_os_valid.log', 'w')

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        dataset = create_os_test_dataset(tokenizer, data_path, buffer_size, batch_size, max_length)
        dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

        # Define the model.
        emotion_predictor = EmotionPredictor(num_layers, d_model, num_heads, dff, hidden_act,
            dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

        # Build the model.
        build_emo_pred_model(emotion_predictor, max_length, vocab_size)

        # Define optimizer and metrics.
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        my_loss = tf.keras.metrics.Mean(name = 'my_loss')

        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = emotion_predictor, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        @tf.function
        def valid_step(dist_inputs):
            def step_fn(inputs):
                inp, inp_seg, inp_emot, _, _, _, tar_emot = inputs

                enc_padding_mask = create_padding_mask(inp)

                pred_emot = emotion_predictor(inp, inp_seg, inp_emot, False, enc_padding_mask)
                losses_per_examples = loss_function(tar_emot, pred_emot)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                return loss

            losses_per_replica = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                losses_per_replica, axis = None)

            my_loss(mean_loss)

        for epoch in range(num_epochs):
            # Restore from the restore_epoch.
            ckpt.restore(ckpt_manager.checkpoints[epoch]).expect_partial()
            print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[epoch]))

            my_loss.reset_states()
            for inputs in dataset:
                valid_step(inputs)
            my_mean_loss = my_loss.result()
            print('Epoch {} My loss {:.4f}'.format(epoch + 1, my_mean_loss))
            # f.write('Epoch {} My loss {:.4f}\n'.format(epoch + 1, my_mean_loss))

    # f.close()

if __name__ == '__main__':
    main()
