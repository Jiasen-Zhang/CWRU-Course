
import glob
import time
import argparse
import numpy as np
from transformers import logging
from rouge_score import rouge_scorer
import preprocessing
from model import LFIP_SUM

''' Compute the Rouge scores'''
def evaluate(summary, ground_truth, scorer):
    score = scorer.score(summary, ground_truth)

    score_list = []

    score_list.append(score['rouge1'][0]) # rouge1_pre
    score_list.append(score['rouge1'][1]) # rouge1_recall
    score_list.append(score['rouge1'][2]) # rouge1_f

    score_list.append(score['rouge2'][0]) # rouge2_pre
    score_list.append(score['rouge2'][1]) # rouge2_recall
    score_list.append(score['rouge2'][2]) # rouge2_f

    score_list.append(score['rougeL'][0]) # rougeL_pre
    score_list.append(score['rougeL'][1]) # rougeL_recall
    score_list.append(score['rougeL'][2]) # rougeL_f

    return score_list


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    tic = time.perf_counter()
    logging.set_verbosity_error()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    """ Set up argparse arguments """
    parser = argparse.ArgumentParser(description='Perform extractive summaization with LFIP_SUM.')
    parser.add_argument('boundary', metavar='boundary function', type=float)
    parser.add_argument('diffusion', metavar='graph diffusion factor', type=float)
    parser.add_argument('n_att', metavar='number of self-attention', type=int)
    args = parser.parse_args()

    boundary = args.boundary
    diffusion = args.diffusion
    n_att = args.n_att

    if boundary<0:
        print("Boundary factor (first one) must be nonnegative.")
        exit(-1)
    if diffusion>=1 or diffusion<0:
        print("Diffusion factor (second one) must satisfy 0 <= diffusion < 1.")
        exit(-1)
    if n_att<=0:
        print("n_att (third one) must be a positive integer.")
        exit(-1)

    data_path = glob.glob('data/doc/*.txt')

    scores = []
    i = 1
    for file_path in data_path:
        truth_path = file_path.replace('doc\doc_', 'summary\summary_')
        ground_truth = preprocessing.load_text(truth_path)[0]

        summary = LFIP_SUM(file_path, boundary=boundary, diffusion=diffusion, n_att=n_att)

        # ground_truth = '. '.join(ground_truth)

        print(str(i),'/445', file_path)
        # print('---------The summary is: ---------')
        # print(summary)
        # print('---------The label is: ---------')
        # print(ground_truth)

        score_list = evaluate(summary, ground_truth, scorer)

        scores.append(score_list)
        i += 1

    # print(scores)
    print('---------The average rouge scores are: ---------')
    scores = np.mean(scores, axis=0)
    print('Rouge_1_precision = ', scores[0])
    print('Rouge_1_recall = ', scores[1])
    print('Rouge_1_fmeasure = ', scores[2])

    print('Rouge_2_precision = ', scores[3])
    print('Rouge_2_recall = ', scores[4])
    print('Rouge_2_fmeasure = ', scores[5])

    print('Rouge_L_precision = ', scores[6])
    print('Rouge_L_recall = ', scores[7])
    print('Rouge_L_fmeasure = ', scores[8])

    toc = time.perf_counter()
    print('time = ', toc - tic)


