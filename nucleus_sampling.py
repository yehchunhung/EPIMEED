import numpy as np
import tensorflow as tf
from scipy.special import logsumexp, softmax


def nucleus_sampling(iter_func, p_threshold, n_sampling, max_length, SOS_ID, EOS_ID, alpha):
    """A simple implementation of nucleus sampling.
    This function assumes that the batch size is 1.
    iter_func should return numpy values.
    alpha is the length normalization coefficient.
    """
    result_seqs = []
    log_probs = np.zeros(n_sampling)

    # Repeat the same sampling procedure for n_sampling times.
    for i in range(n_sampling):
        result_seq = [SOS_ID]
        for _ in range(max_length):
            dec_inp = tf.expand_dims(result_seq, 0)
            pred = iter_func(dec_inp, 1)  # (1, seq_len, vocab_size)
            pred = pred[0,-1,:]  # (vocab_size,)

            prob = softmax(pred)
            log_prob = pred - logsumexp(pred)
            sorted_id = pred.argsort()[::-1]  # Descending

            cut_idx = 0
            prob_cumsum = np.cumsum(prob[sorted_id])
            while prob_cumsum[cut_idx] < p_threshold:
                cut_idx += 1

            new_dist = prob[sorted_id][:(cut_idx+1)]
            new_dist = new_dist / np.sum(new_dist)
            sample = np.random.choice(new_dist.shape[0], p = new_dist)
            next_id = sorted_id[sample]

            result_seq.append(next_id)
            log_probs[i] += log_prob[next_id]
            if next_id == EOS_ID:
                break

        if result_seq[-1] != EOS_ID:
            dec_inp = tf.expand_dims(result_seq, 0)
            pred = iter_func(dec_inp, 1)  # (1, seq_len, vocab_size)
            pred = pred[0,-1,:]  # (vocab_size,)
            log_prob = pred - logsumexp(pred)

            result_seq.append(EOS_ID)
            log_probs[i] += log_prob[EOS_ID]

        result_seqs.append(result_seq)

    # Sort the resulted sequences by their log probabilities.
    result_seqs, log_probs = _sort_by_log_probs(result_seqs, log_probs, alpha)

    return result_seqs, log_probs

def _sort_by_log_probs(result_seqs, log_probs, alpha):
    seq_lens = np.array([len(seq) - 1 for seq in result_seqs])
    len_norm = seq_lens ** alpha
    sorted_id = (log_probs / len_norm).argsort()[::-1]
    new_log_probs = (log_probs / len_norm)[sorted_id]
    new_result_seqs = [result_seqs[i] for i in sorted_id]
    return new_result_seqs, new_log_probs
