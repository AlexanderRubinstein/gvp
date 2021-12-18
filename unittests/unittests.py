import sys
import os
sys.path.append(os.path.join("..", "src"))

import pickle
import numpy as np
import tensorflow as tf

import util
from constants import FEATURIZER_PATH
from models import CPDModel, PairwiseCPDModel


def test_models_equivalence():
    with open(os.path.join("..", "data", "unittest_data", "test_structure.pkl"), 'rb') as inp:
        (structure, seq, mask) = pickle.load(inp)

    original_model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))
    optimizer1 = tf.keras.optimizers.Adam()
    util.load_checkpoint(original_model, optimizer1, FEATURIZER_PATH)

    featurizer = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))
    optimizer2 = tf.keras.optimizers.Adam()
    util.load_checkpoint(featurizer, optimizer2, FEATURIZER_PATH)

    pairwise_model = PairwiseCPDModel(featurizer, num_letters=20, hidden_dim=(16,100), copy_top_gvp=True)
    # forward by original
    original_result = original_model(structure, seq, mask, train=False).numpy()
    # forward by pairwise with copy_top_gvp
    pairwise_result = pairwise_model(structure, seq, mask, train=False).numpy()

    assert np.isclose(original_result, pairwise_result).all()
    print("test_models_equivalence: OK")

if __name__ == "__main__":
	test_models_equivalence()
