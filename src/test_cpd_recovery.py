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

from util import (
    labels_to_onehot,
    onehot_to_labels,
    energy_matrix_from_logits_pairwise
)

from optimization import solve_gumbel_soft_max_opt


from constants import FEATURIZER_PATH

def recovery(designs, orig):
    N, L = designs.shape
    arr = (designs == orig)
    arr = arr.sum(1) / L
    return np.mean(arr)

def sample_by_optimizing_energy_wrapper(model):
    def sample_by_optimizing_energy(X, mask, temperature):
        logits_pairwise, E_idx = model.sample_logits_pairwise(X, mask, temperature)
        logits_pairwise = logits_pairwise.numpy()
        residuewise_logits = build_logits_residuewise(
            logits_pairwise,
            E_idx,
            model.num_letters,
            mask=mask
        )
        preds_from_logits = np.argmax(residuewise_logits, axis=-1)
        preds_from_optimization = np.zeros_like(preds_from_logits)
        energy_matrix = energy_matrix_from_logits_pairwise(logits_pairwise, E_idx, mask, model.num_letters)
        for i in range(logits_pairwise.shape[0]):
            energy_matrix_i = energy_matrix[i]
            answer_from_logits = labels_to_onehot(preds_from_logits[i], model.num_letters)
            energy_from_logits = answer_from_logits.T @ energy_matrix_i @ answer_from_logits
            print("energy_from_logits:", energy_from_logits)
            answer_from_optimization, energy_from_optimization = solve_gumbel_soft_max_opt(
                energy_matrix_i,
                compute_loss_func=None,
                minimize=False,
                n_aminos=model.num_letters,
                n_epochs=500,
                n_runs=5,
                stat=False,
                verbose=False,
                tau=1
            )
            print("energy_from_optimization:", energy_from_optimization)
            print("logits sum vs energy optimization seq rec: ", answer_from_optimization.T @ answer_from_logits / (int(answer_from_logits.shape[0] / model.num_letters)))
            preds_from_optimization[i] = onehot_to_labels(answer_from_optimization, model.num_letters)
        return preds_from_optimization
    return sample_by_optimizing_energy
    
def sample(sampling_method, structure, mask, n, T=0.1): # [1, N, 4, 3]
    structure = tf.repeat(structure, n, axis=0)
    mask = tf.repeat(mask, n, axis=0)
    return sampling_method(structure, mask, temperature=T)

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
    parser.add_argument(
        '--prediction_type',
        type=str,
        default=None,
        help='how to predict residue label from pairwise logits',
        choices=["frequency", "logits_sum", "energy_optimization", None]
    )
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

    featurizer = CPDModel(node_features=(8, 100), edge_features=(1, 32), hidden_dim=(16, 100))

    if args.pairwise:

        util.load_checkpoint(featurizer, optimizer, FEATURIZER_PATH)

        model = PairwiseCPDModel(featurizer, num_letters=20, hidden_dim=(16, 100), copy_top_gvp=True)

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
                elif args.prediction_type == "energy_optimization":
                    pred = sample(sample_by_optimizing_energy_wrapper(model), structure, mask, my_n)
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
