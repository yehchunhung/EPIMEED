import nltk
import rouge
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk

nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

from nlgeval import NLGEval

sbert = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

def cal_bleu(read_path, n):
    all_weights = [(1.0,), (0.5,0.5), (1/3,1/3,1/3), (0.25,0.25,0.25,0.25)]
    df = pd.read_csv(read_path).dropna()
    references = df['tar_y'].tolist()
    candidates = df['pred_y'].tolist()
    bleu_scores = []
    for ref, cand in tqdm(zip(references, candidates), total = len(references)):
        ref_t = nltk.word_tokenize(ref)
        cand_t = nltk.word_tokenize(cand)
        score = sentence_bleu([ref_t], cand_t, weights = all_weights[n - 1])
        bleu_scores.append(score)
    return np.mean(bleu_scores)

'''
def cal_dist(read_path, n):
    df = pd.read_csv(read_path)
    candidates = df['pred_y'].tolist()
    dist_scores = []
    for cand in tqdm(candidates):
        cand_t = nltk.word_tokenize(cand.lower())
        if n > len(cand_t):
            continue
        else:
            cand_grams = [cand_t[i:(i+n)] for i in range(len(cand_t) - n + 1)]
            cand_grams = [' '.join(gram) for gram in cand_grams]
        score = len(set(cand_grams)) / len(cand_grams)
        dist_scores.append(score)
    return np.mean(dist_scores)'''

def cal_dist(read_path, n):
    df = pd.read_csv(read_path).dropna()
    candidates = df['pred_y'].tolist()
    all_grams = []
    for cand in tqdm(candidates):
        cand_t = nltk.word_tokenize(cand.lower())
        if n > len(cand_t):
            continue
        else:
            cand_grams = [cand_t[i:(i+n)] for i in range(len(cand_t) - n + 1)]
            cand_grams = [' '.join(gram) for gram in cand_grams]
            all_grams += cand_grams
    return len(set(all_grams)) / len(all_grams)

def cal_rouge(read_path):
    #def prepare_results(metric, p, r, f):
    #    return '{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

    '''apply_avg = (apply_policy == 'avg')
    apply_best = (apply_policy == 'best')'''

    evaluator = rouge.Rouge(metrics = ['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n = 2,
                            limit_length = True,
                            length_limit = 100,
                            length_limit_type = 'words',
                            apply_avg = True,
                            apply_best = False,
                            alpha = 0.5, # Default F1_score
                            weight_factor = 1.2,
                            stemming = True)
    df = pd.read_csv(read_path).dropna()
    references = df['tar_y'].tolist()
    candidates = df['pred_y'].tolist()
    scores = evaluator.get_scores(candidates, references)

    dictionary = {}
    
    for metric, results in sorted(scores.items(), key = lambda x: x[0]):
        print(prepare_results(metric, results['p'], results['r'], results['f']))
        dictionary[metric] = [results['p'], results['r'], results['f']]

    return dictionary

def cal_avg_length(read_path):
    df = pd.read_csv(read_path).dropna()
    candidates = df['pred_y'].tolist()

    tokenizer = nltk.tokenize.WhitespaceTokenizer()

    words = 0

    for cand in candidates:
        arr = tokenizer.tokenize(cand)
        words += len(arr)

    avg_words = words / len(candidates)

    return avg_words

def cal_embed_sim(read_path):
    df = pd.read_csv(read_path).dropna()
    references = df['tar_y'].tolist()
    candidates = df['pred_y'].tolist()
    ref_embed = sbert.encode(references)
    cand_embed = sbert.encode(candidates)
    cos_sim = cosine_similarity(cand_embed, ref_embed)
    return np.mean(np.diag(cos_sim))


def cal_multiple(read_path):
    df = pd.read_csv(read_path).dropna()
    references = df['tar_y'].tolist()
    candidates = df['pred_y'].tolist()

    nlgeval = NLGEval()  # loads the models$
    print(len(references), len(candidates))
    metrics_dict = nlgeval.compute_metrics([references], candidates)
    return metrics_dict

with open('./auto-metrics/auto-metrics-red+_mask_meed2+_2000_.csv', 'a') as outfile:

    writer = csv.writer(outfile, delimiter=',')
    '''writer.writerow(['model', 'dataset', 'version', 'd1', 'd2', 'bleu1', 'bleu2', 'avg words / utterance', 
                    'rouge-1 (prec)', 'rouge-1 (recall)', 'rouge-1 (F1)',
                    'rouge-2 (prec)', 'rouge-2 (recall)', 'rouge-2 (F1)',
                    #'rouge-3 (prec)', 'rouge-3 (recall)', 'rouge-3 (F1)',
                    #'rouge-4 (prec)', 'rouge-4 (recall)', 'rouge-4 (F1)',
                    'rouge-l (prec)', 'rouge-l (recall)', 'rouge-l (F1)',
                    'rouge-w (prec)', 'rouge-w (recall)', 'rouge-w (F1)',])'''

    # writer.writerow(['model', 'dataset', 'version', 'd1', 'd2', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SkipThoughts Cosine Similarity', 'Embedding Average Cosine Similarity', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore', 'avg words / utterance',])
    writer.writerow(['model', 'dataset', 'version', 'd1', 'd2', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SkipThoughts Cosine Similarity', 'Embedding Average Cosine Similarity', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore', 'avg words / utterance', 'sentence similarity'])

    for model_name in ['plain-os-dramatic-rest-red']: # ['plain-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red', 'xenbot-os-dramatic-rest-red+', 'xenbot-os-dramatic-rest-red+_mask_']
        for dataset in ['red+_mask']:
            simple_met = False
            complex_met = False
            for version in ['plain']: # ['plain', 'meed2', 'meed2+']
            #for version in ['plain', 'alexa']:

                print(model_name, dataset, version)

                if version == 'meed2':
#                     read_path = './prediction/{}/meed2/{}-{}.csv'.format(dataset, model_name, version)
                    read_path = './prediction-listener/{}/meed2/{}-{}.csv'.format(dataset, model_name, version)
                elif version == 'meed2+':
#                     read_path = './prediction/{}/meed2/{}-{}.csv'.format(dataset, model_name, version)
                    read_path = './prediction-listener/{}/meed2+/{}-{}.csv'.format(dataset, model_name, version)
                elif version == 'plain':
#                     read_path = './prediction/{}/{}/{}-{}.csv'.format(dataset, version, model_name.replace('xenbot', 'plain'), version)
                    read_path = './prediction-listener/{}/plain/{}-{}.csv'.format(dataset, model_name, version)
                elif version == 'alexa':
                    read_path = './prediction/{}/{}/{}-{}.csv'.format(dataset, version, model_name, version)
                elif version == 'simple':
                    if simple_met == False:
                        method = 'argmax'
                        simple_met = True
                    else:
                        method = 'prob-sampled'
                    read_path = './prediction/{}/{}-heldout/{}-{}.csv'.format(dataset, method, model_name, version)
                elif version == 'complex':
                    if complex_met == False:
                        method = 'argmax'
                        complex_met = True
                    else:
                        method = 'prob-sampled'
                    read_path = './prediction/{}/{}-heldout/{}-{}.csv'.format(dataset, method, model_name, version)
                else:
                    read_path = './prediction/{}/argmax/{}-{}.csv'.format(dataset, model_name, version)

                

                d1 = cal_dist(read_path, 1)
                d2 = cal_dist(read_path, 2)            
                #bleu1 = cal_bleu(read_path, 1)
                #bleu2 = cal_bleu(read_path, 2)
                ses = cal_embed_sim(read_path)
                metrics_dict = cal_multiple(read_path)
                avg_words = cal_avg_length(read_path)
                #rouge_dict = cal_rouge(read_path)
                
                
                d1 = round(d1, 4)
                d2 = round(d2, 4)

                ses = round(ses, 4)
                avg_words = round(avg_words, 4)
                
                
                '''bleu1 = round(bleu1, 4)
                bleu2 = round(bleu2, 4)
                #ses = round(ses, 4)
                
                rouge_arr = []

                for key, val in rouge_dict.items():
                    rouge_arr.append(round(val[0], 4))
                    rouge_arr.append(round(val[1], 4))
                    rouge_arr.append(round(val[2], 4))

                new_row = [model_name, dataset, version, d1, d2, bleu1, bleu2, avg_words] + rouge_arr'''

                metrics_arr = []
                for key, val in metrics_dict.items():
                    print(key, val)
                    metrics_arr.append(round(val, 4))

                if version == 'simple' or version == 'complex':
                    new_row = [model_name, dataset, version+" ("+method+")", d1, d2] + metrics_arr + [avg_words]
                else:
                    new_row = [model_name, dataset, version, d1, d2] + metrics_arr + [avg_words] + [ses]
                writer.writerow(new_row)
