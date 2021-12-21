import sys
import os
sys.path.append(os.path.join("..", "src"))

import pickle
import numpy as np
import tensorflow as tf

import util
from constants import FEATURIZER_PATH
from models import CPDModel, PairwiseCPDModel
from util import (
    make_h_V_pairwise_redneck,
    make_S_pairwise_redneck,
    make_mask_pairwise_redneck,
    labels_pair_to_pairwise_label,
    pairwise_label_to_labels_pair
)

TEST_STRUCTS = os.path.join("..", "data", "unittest_data", "test_structure.pkl")


# tests

def test_label_conversion():
    for i in range(400):
        assert(i == labels_pair_to_pairwise_label(pairwise_label_to_labels_pair(i, 20), 20))
    for i in range(20):
        for j in range(20):
            assert([i,j] == pairwise_label_to_labels_pair(labels_pair_to_pairwise_label([i, j], 20), 20))
    print("test_label_conversion: OK")

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

    # [batch_size, n_nodes, h_dim)]
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
    # n_neighbours = k = 2
    # [batch_size, n_nodes, n_neighbours)]
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
    # [batch_size, n_nodes, n_neighbours, 2 * h_dim)]
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
                    [0, 0, 0, 1, 2, 3],  #  masked 0
                    [0, 0, 0, 4, 5, 6]  #  masked 1
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

    h_V_pairwise = make_h_V_pairwise_redneck(h_V_train_for_test, E_idx_for_test)
    assert np.isclose(h_V_pairwise, expected_h_V_pairwise).all()
    print("test_pairwise_embeddings: OK")

def test_pairwise_sequences():

    # n_neighbours = k = 2
    # [batch_size, n_nodes, n_neighbours)]
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

    # [batch_size, n_nodes, n_neighbours, 2)]
    S_for_test = np.array(
        [
            [1, 2, 3, 4, 0],
            [5, 6, 7, 8, 9]
        ]
    )

    # num_single_labels = 10

    expected_S_pairwise = np.array(
        [
            [
                [
                    12,  # [1, 2],  #  0 -> 1
                    13  # [1, 3]  #  0 -> 2
                ],
                [
                    21,  # [2, 1],  #  1 -> 0
                    23  # [2, 3]  #  1 -> 2
                ],
                [
                    32,  # [3, 2],  #  2 -> 1
                    34  # [3, 4]  #  2 -> 3
                ],
                [
                    43,  # [4, 3],  #  3 -> 2
                    42  # [4, 2]  #  3 -> 1
                ],
                [
                    1,  # [0, 1],  #  masked 0
                    2  # [0, 2]  #  masked 1
                ],
            ],
            [
                [
                    56,  # [5, 6],  #  0 -> 1
                    57  # [5, 7]  #  0 -> 2
                ],
                [
                    65,  # [6, 5],  #  1 -> 0
                    67  # [6, 7]  #  1 -> 2
                ],
                [
                    76,  # [7, 6],  #  2 -> 1
                    78  # [7, 8]  #  2 -> 3
                ],
                [
                    87,  # [8, 7],  #  3 -> 2
                    89  # [8, 9]  # 3 -> 4
                ],
                [
                    98,  # [9, 8],  #  4 -> 3
                    97  # [9, 7]  #  4 -> 2
                ],
            ]
        ]
    )

    S_pairwise = make_S_pairwise_redneck(S_for_test, E_idx_for_test, num_single_labels=10)

    assert np.isclose(S_pairwise, expected_S_pairwise).all()
    print("test_pairwise_sequences: OK")

def test_pairwise_mask():
    # n_neighbours = k = 2
    # [batch_size, n_nodes, n_neighbours)]
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

    # [batch_size, n_nodes)]
    mask_for_test = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ]
    )

    expected_mask_pairwise = np.array(
        [
            [
                [
                    1,  #  0 -> 1
                    1  #  0 -> 2
                ],
                [
                    1,  #  1 -> 0
                    1  #  1 -> 2
                ],
                [
                    1,  #  2 -> 1
                    1  #  2 -> 3
                ],
                [
                    1,  #  3 -> 2
                    1  #  3 -> 1
                ],
                [
                    0,  #  masked 0
                    0  #  masked 1
                ],
            ],
            [
                [
                    1,  #  0 -> 1
                    1  #  0 -> 2
                ],
                [
                    1,  #  1 -> 0
                    1  #  1 -> 2
                ],
                [
                    1,  #  2 -> 1
                    1  #  2 -> 3
                ],
                [
                    1,  #  3 -> 2
                    1  # 3 -> 4
                ],
                [
                    1,  #  4 -> 3
                    1  #  4 -> 2
                ],
            ]
        ]
    )

    mask_pairwise = make_mask_pairwise_redneck(mask_for_test, E_idx_for_test)
    assert np.isclose(mask_pairwise, expected_mask_pairwise).all()
    print("test_pairwise_mask: OK")

if __name__ == "__main__":
    test_label_conversion()
    test_models_equivalence()
    test_pairwise_embeddings()
    test_pairwise_sequences()
    test_pairwise_mask()
