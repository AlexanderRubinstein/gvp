import sys

from datetime import datetime
from datasets import *
import random, pdb
import numpy as np
import tqdm, util
from models import *
import argparse
import os
import pickle

from constants import FEATURIZER_PATH

def recovery(designs, orig):
    N, L = designs.shape
    arr = (designs == orig)
    arr = arr.sum(1) / L
    return np.mean(arr)
    
def sample(sampling_method, structure, mask, n, T=0.1): # [1, N, 4, 3]
    structure = tf.repeat(structure, n, axis=0)
    mask = tf.repeat(mask, n, axis=0)
    return sampling_method(structure, mask, temperature=0.1)

def make_parser():
    parser = argparse.ArgumentParser(description='test cpd script parser')
    parser.add_argument('--model_path', type=str, default=None,
                    help='path to the model to test')
    parser.add_argument('--test_type', type=str,
                    help='what kind of test set to use', choices=["cath", "short", "sc", "ts50"])
    parser.add_argument('--output_file', type=str,
                    help='output file')
    parser.add_argument('--max_structs', type=int, default=0,
                    help='max number of structs to test on')
    parser.add_argument('--pairwise', action="store_true",
                    help='whether to use pairwise model')
    parser.add_argument('--pairwise_logits', action="store_true",
                    help='whether to predict from pairwise logits')
    parser.add_argument('--n_runs', type=int, default=100,
                    help='number of runs to calculate mean result from')
    parser.add_argument('--prediction_type', type=str, default=None,
                    help='how to predict residue label from pairwise logits', choices=["frequency", "logits_sum", None])
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()

    if not args.pairwise and args.model_path is None:
        raise Exception("No model filepath for original model")

    if args.pairwise_logits and not args.pairwise:
        raise Exception("Trying to run predict from logits_pairwise with original model")

    if args.pairwise_logits and not args.model_path:
        raise Exception("Trying to predict from logits_pairwise with untrained pairwise_classificator")

    if args.prediction_type is not None and not args.pairwise_logits:
        raise Exception(f"Prediction_type '{args.prediction_type}' specified without --pairwise_logits argument")

    if args.prediction_type is None and args.pairwise_logits:
        raise Exception(f"--prediction_type argument not specified while using --pairwise_logits")

    optimizer = tf.keras.optimizers.Adam()

    featurizer = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))

    if args.pairwise:

        util.load_checkpoint(featurizer, optimizer, FEATURIZER_PATH)

        model = PairwiseCPDModel(featurizer, num_letters=20, hidden_dim=(16,100), copy_top_gvp=True)

        if args.model_path:
            util.load_checkpoint(model, optimizer, args.model_path)
    else:
        model = featurizer
        util.load_checkpoint(model, optimizer, args.model_path)

    if args.test_type == "cath":
        _, _, testset = cath_dataset(1)
    elif args.test_type == "short":
        _, _, testset = cath_dataset(1, filter_file='../data/test_split_L100.json', cache_name="cath_short")
    elif args.test_type == "sc":
        _, _, testset = cath_dataset(1, filter_file='../data/test_split_sc.json', cache_name="cath_sc")
    elif args.test_type == "ts50":
        testset = ts50_dataset(1)

    num = 0
    # to reset file
    with open(args.output_file, 'w') as f:
            f.write('')
    for structure, seq, mask in tqdm.tqdm(testset):

        num += 1
        length = seq.shape[1]

        N = args.n_runs

        n = min(int(20000 / length), N)
        design = np.zeros((N, length))
        idx = 0
        while (idx < N):
            my_n = min(n, N-idx)
            if args.pairwise_logits:
                if args.prediction_type == "frequency":
                    pred = sample(model.sample_pairwise_from_frequencies, structure, mask, my_n)
                elif args.prediction_type == "logits_sum":
                    pred = sample(model.sample_pairwise_from_logits_sum, structure, mask, my_n)
            else:
                pred = sample(model.sample_independently if args.pairwise else model.sample, structure, mask, my_n)
                pred = tf.cast(pred, tf.int32).numpy()
            design[idx:idx+my_n] = pred
            idx += min(n, N-idx)

        seq = seq.numpy()
        res = recovery(design, seq)
        print(res)
        with open(args.output_file, 'a+') as f:
            f.write(str(res) + '\n')

        if args.max_structs > 0 and num == args.max_structs:
            break
