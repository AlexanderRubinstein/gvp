import argparse
from datasets import cath_dataset
from models import CPDModel, PairwiseCPDModel
import util
from constants import FEATURIZER_PATH

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import os


def make_parser():
    parser = argparse.ArgumentParser(description='check pairwise prediction consistency parser')
    parser.add_argument('--model_path', type=str, required=True,
                    help='path to the model to check')
    parser.add_argument('--test_type', type=str,
                    help='what kind of test set to use', choices=["cath", "short", "sc", "ts50"])
    parser.add_argument('--output_path', type=str, required=True,
                    help='output folder with pairwise histograms')
    parser.add_argument('--max_structs', type=int, default=0,
                    help='max number of structs to check')
    return parser

def plot_consistency_histogram(prediction_frequencies, output_path, idx, num_letters=20):

    n_residues = prediction_frequencies.shape[0]
    ncols = 3
    nrows = int(np.ceil(n_residues / (1.0 * ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 5))

    for residue_idx in range(n_residues):

        i = int(residue_idx / ncols)
        j = residue_idx % ncols

        ax = axes[i][j]

        ax.bar(range(num_letters), prediction_frequencies[residue_idx], color='blue', label='residue number: {}'.format(residue_idx))
        ax.set_xlabel('Amino type')
        ax.set_ylabel('Votes count')
        ax.set_xticks(range(num_letters))
        leg = ax.legend(loc='upper left')
        leg.draw_frame(False)

    strFile = os.path.join(output_path, f"Protein_{idx}.pdf")
    if os.path.isfile(strFile):
       os.remove(strFile)

    fig.savefig(strFile)

if __name__ == "__main__":
    args = make_parser().parse_args()

    optimizer = tf.keras.optimizers.Adam()

    featurizer = CPDModel(node_features=(8, 100), edge_features=(1, 32), hidden_dim=(16, 100))

    util.load_checkpoint(featurizer, optimizer, FEATURIZER_PATH)

    model = PairwiseCPDModel(featurizer, num_letters=20, hidden_dim=(16, 100), copy_top_gvp=True)

    util.load_checkpoint(model, optimizer, args.model_path)

    if args.test_type == "cath":
        _, _, testset = cath_dataset(1)
    elif args.test_type == "short":
        _, _, testset = cath_dataset(1, filter_file='../data/test_split_L100.json', cache_name="cath_short")
    elif args.test_type == "sc":
        _, _, testset = cath_dataset(1, filter_file='../data/test_split_sc.json', cache_name="cath_sc")
    elif args.test_type == "ts50":
        testset = ts50_dataset(1)

    idx = 0
    temperature = 0.1
    os.makedirs(args.output_path, exist_ok=True)
    for structure, seq, mask in tqdm.tqdm(testset):
        prediction_frequencies = model.sample_pairwise_before_argmax(structure, mask, temperature, "frequency")
        plot_consistency_histogram(prediction_frequencies.squeeze(0), args.output_path, idx)
        idx += 1
        if args.max_structs > 0 and idx == args.max_structs:
            break
