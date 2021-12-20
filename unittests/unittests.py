import sys
import os
sys.path.append(os.path.join("..", "src"))

import pickle
import numpy as np
import tensorflow as tf

import util
from constants import FEATURIZER_PATH
from models import CPDModel, PairwiseCPDModel

TEST_STRUCTS = os.path.join("..", "data", "unittest_data", "test_structure.pkl")


def test_models_equivalence():
    with open(TEST_STRUCTS, 'rb') as inp:
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

def test_pairwise_embeddings():
    with open(TEST_STRUCTS, 'rb') as inp:
        (structure, seq, mask) = pickle.load(inp)

    original_model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))
    V, E, E_idx = original_model.features(structure, mask)
    h_V_train = original_model.train_embeddings(structure, seq, mask, train=False)

    mask_for_test = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ]
    )
    h_V_train_for_test = np.array(
        [
            [
                [1, 2, 3],  # v = 0
                [4, 5, 6],  # v = 1
                [7, 8, 9],  # v = 2
                [10, 11, 12],  # v = 3
                [0, 0, 0]  # v = 4
            ],
            [
                [-1, -2, -3],  # v = 0
                [-4, -5, -6],  # v = 1
                [-7, -8, -9],  # v = 2
                [-10, -11, -12],  # v = 3
                [-13, -14, -15]  # v = 4
            ]
        ]
    )
    # k = 2
    E_idx_for_test = np.array(
        [
            [
                [1, 2],  # nearest for 0
                [0, 2],  # nearest for 1
                [1, 3],  # nearest for 2
                [2, 1],  # nearest for 3
                [0, 1]  # np.arange(2)
            ],
            [
                [1, 2],  # nearest for 0
                [0, 2],  # nearest for 1
                [1, 3],  # nearest for 2
                [2, 4],  # nearest for 3
                [3, 2]  # nearest for 4
            ]
        ]
    )
    expected_h_V_pairwise = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6],  #  0 -> 1
                    [1, 2, 3, 7, 8, 9]  #  0 -> 2
                ],
                [
                    [4, 5, 6, 1, 2, 3],  #  1 -> 0
                    [4, 5, 6, 7, 8, 9]  #  1 -> 2
                ],
                [
                    [7, 8, 9, 4, 5, 6],  #  2 -> 1
                    [7, 8, 9, 10, 11, 12]  #  2 -> 3
                ],
                [
                    [10, 11, 12, 7, 8, 9],  #  3 -> 2
                    [10, 11, 12, 4, 5, 6]  #  3 -> 1
                ],
                [
                    [0, 0, 0, 0, 0, 0],  #  masked 0
                    [0, 0, 0, 0, 0, 0]  #  masked 1
                ],
            ],
            [
                [
                    [-1, -2, -3, -4, -5, -6],  #  0 -> 1
                    [-1, -2, -3, -7, -8, -9]  #  0 -> 2
                ],
                [
                    [-4, -5, -6, -1, -2, -3],  #  1 -> 0
                    [-4, -5, -6, -7, -8, -9]  #  1 -> 2
                ],
                [
                    [-7, -8, -9, -4, -5, -6],  #  2 -> 1
                    [-7, -8, -9, -10, -11, -12]  #  2 -> 3
                ],
                [
                    [-10, -11, -12, -7, -8, -9],  #  3 -> 2
                    [-10, -11, -12, -13, -14, -15]  # 3 -> 4
                ],
                [
                    [-13, -14, -15, -10, -11, -12],  #  4 -> 3
                    [-13, -14, -15, -7, -8, -9]  #  4 -> 2
                ],
            ]
        ]
    )
    # print("mask[0][None, ...].shape:", mask[0][None, ...].shape)
    print("mask_for_test.shape:", mask_for_test.shape)
    # print("V.shape:", V.shape)
    # print("E.shape:", E.shape)
    print("E_idx_for_test.shape:", E_idx_for_test.shape)
    # print("E_idx[0][None, ...].shape:", E_idx[0][None, ...].shape)
    # print("h_V_train[0][None, ...].shape:", h_V_train[0][None, ...].shape)
    print("h_V_train_for_test.shape:", h_V_train_for_test.shape)
    print("expected_h_V_pairwise.shape:", expected_h_V_pairwise.shape)
    # print("mask:", mask)
    # print("h_V_train[0][-1].shape:", h_V_train[0][-1])
    # print("h_V_train[0][-7].shape:", h_V_train[0][-7])
    # print("E_idx[0][-1]:", E_idx[0][-1])
    # print("E_idx[0][-1]:", E_idx[0][-7])


if __name__ == "__main__":
    # test_models_equivalence()
    test_pairwise_embeddings()
