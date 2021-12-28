import sys
import os
sys.path.append(os.path.join("..", "src"))

import pickle
import numpy as np
import tensorflow as tf

import copy

import util
from constants import FEATURIZER_PATH
from models import CPDModel, PairwiseCPDModel
from util import (
    make_h_V_pairwise_redneck,
    make_S_pairwise_redneck,
    make_mask_pairwise_redneck,
    labels_pair_to_pairwise_label,
    pairwise_label_to_labels_pair,
    build_prediction_frequencies_redneck,
    compute_indices_affecting_edge_types,
    build_logits_residuewise
)
tf.random.set_seed(
    14
)

TEST_STRUCTS = os.path.join("..", "data", "unittest_data", "test_structure.pkl")


# tests

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

def test_label_convertation():
    for i in range(400):
        assert(i == labels_pair_to_pairwise_label(pairwise_label_to_labels_pair(i, 20), 20))
    for i in range(20):
        for j in range(20):
            assert([i,j] == pairwise_label_to_labels_pair(labels_pair_to_pairwise_label([i, j], 20), 20))
    print("test_label_convertation: OK")

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

def make_pairwise_model(num_letters, copy_top_gvp=False):
    model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))

    optimizer = tf.keras.optimizers.Adam()
    util.load_checkpoint(model, optimizer, FEATURIZER_PATH)

    return PairwiseCPDModel(model, num_letters=num_letters, hidden_dim=(16,100), copy_top_gvp=copy_top_gvp)

def test_frequencies_from_logits_pairwise():

    # [batch_size, n_nodes]
    mask_for_test = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ]
    )

    # n_neighbours = k = 2
    # [batch_size, n_nodes, n_neighbours]
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

    num_single_labels = 3

    # ground_truth: batch0: v0=1, v1=2, v2=2, v3=0, v4=masked
    # ground_truth: batch1: v0=0, v1=1, v2=2, v3=1, v4=0

    logits_pairwise_for_test = np.array(
        [
            [
                [
                    [-10, -10, -10, -10, -10, 1000, -10, -10, -10],  # 1 * 3 + 2 = 5  # [1, 2],  #  0 -> 1
                    [-10, -10, -10, -10, -10, 1000, -10, -10, -10]  # 1 * 3 + 2 = 5  # [1, 2]  #  0 -> 2
                ],
                [
                    [-10, -10, -10, -10, -10, -10, -10, 1000, -10],  # 2 * 3 + 1 = 7  # [2, 1],  #  1 -> 0
                    [-10, -10, -10, -10, -10, -10, -10, -10, 1000]  # 2 * 3 + 2 = 8  # [2, 2]  #  1 -> 2
                ],
                [
                    [-10, -10, -10, -10, -10, -10, -10, -10, 1000],  # 2 * 3 + 2 = 8  # [2, 2],  #  2 -> 1
                    [-10, -10, -10, -10, -10, -10, 1000, -10, -10]  # 2 * 3 + 0 = 6  # [2, 0]  #  2 -> 3
                ],
                [
                    [-10, -10, 1000, -10, -10, -10, -10, -10, -10],  # 0 * 3 + 2 = 2  # [0, 2],  #  3 -> 2
                    [-10, -10, 1000, -10, -10, -10, -10, -10, -10]  # 0 * 3 + 2 = 2  # [0, 2]  #  3 -> 1
                ],
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],  # some logits that will be masked,  #  masked 0
                    [9, 8, 7, 6, 5, 4, 3, 2, 1]  # some logits that will be masked  #  masked 1
                ],
            ],
            [
                [
                    [-10, 1000, -10, -10, -10, -10, -10, -10, -10],  # 0 * 3 + 1 = 1  # [0, 1],  #  0 -> 1
                    [-10, -10, 1000, -10, -10, -10, -10, -10, -10]  # 0 * 3 + 2 = 2  # [0, 2]  #  0 -> 2
                ],
                [
                    [-10, -10, -10, 1000, -10, -10, -10, -10, -10],  # 1 * 3 + 0 = 3  # [1, 0],  #  1 -> 0
                    [-10, -10, -10, -10, -10, 1000, -10, -10, -10]  # 1 * 3 + 2 = 5  # [1, 2]  #  1 -> 2
                ],
                [
                    [-10, -10, -10, -10, -10, -10, -10, 1000, -10],  # 2 * 3 + 1 = 7  # [2, 1],  #  2 -> 1
                    [-10, -10, -10, -10, -10, -10, -10, 1000, -10]  # 2 * 3 + 1 = 7  # [2, 1]  #  2 -> 3
                ],
                [
                    [-10, -10, -10, -10, -10, 1000, -10, -10, -10],  # 1 * 3 + 2 = 5  # [1, 2],  #  3 -> 2
                    [-10, -10, -10, 1000, -10, -10, -10, -10, -10]  # 1 * 3 + 0 = 3  # [1, 0]  # 3 -> 4
                ],
                [
                    [-10, 1000, -10, -10, -10, -10, -10, -10, -10],  # 0 * 3 + 1 = 1  # [0, 1],  #  4 -> 3
                    [-10, -10, 1000, -10, -10, -10, -10, -10, -10]  # 0 * 3 + 2 = 2  # [0, 2]  #  4 -> 2
                ],
            ]
        ]
    )

    expected_prediction_frequencies = np.array(
        [
            [
                [0, 3, 0],  # for 0 based on: 0 -> 1 [1, 2], 0 -> 2 [1, 2], 1 -> 0 [2, 1]
                [0, 0, 5],  # for 1 based on: 0 -> 1 [1, 2], 1 -> 0 [2, 1], 1 -> 2 [2, 2], 2 -> 1 [2, 2], 3 -> 1 [0, 2]
                [0, 0, 5],  # for 2 based on: 0 -> 2 [1, 2], 1 -> 2 [2, 2], 2 -> 1 [2, 2], 2 -> 3 [2, 0], 3 -> 2 [0, 2]
                [3, 0, 0],  # for 3 based on: 2 -> 3 [2, 0], 3 -> 2 [0, 2], 3 -> 1 [0, 2]
                [0, 0, 0]  # 0 instead of frequencies for masked
            ],
            [
                [3, 0, 0],  # for 0 based on: 0 -> 1 [0, 1], 0 -> 2 [0, 2], 1 -> 0 [1, 0]
                [0, 4, 0],  # for 1 based on: 0 -> 1 [0, 1], 1 -> 0 [1, 0], 1 -> 2 [1, 2], 2 -> 1 [2, 1]
                [0, 0, 6],  # for 2 based on: 0 -> 2 [0, 2], 1 -> 2 [1, 2], 2 -> 1 [2, 1], 2 -> 3 [2, 1], 3 -> 2 [1, 2], 4 -> 2 [0, 2]
                [0, 4, 0],  # for 3 based on: 2 -> 3 [2, 1], 3 -> 2 [1, 2], 3 -> 4 [1, 0], 4 -> 3 [0, 1]
                [3, 0, 0]  # for 4 based on: 3 -> 4 [1, 0], 4 -> 3 [0, 1], 4 -> 2 [0, 2]
            ]
        ]
    )

    prediction_frequencies = build_prediction_frequencies_redneck(logits_pairwise_for_test, E_idx_for_test, num_single_labels, onehot_by_argmax=False, mask=mask_for_test)

    assert np.isclose(prediction_frequencies, expected_prediction_frequencies).all()
    print("test_frequencies_from_logits_pairwise: OK")

def mult_nested_lists(input, mult):
    if isinstance(input, list):
        return [mult_nested_lists(input[i], mult) for i in range(len(input))]
    else:
        return mult * input

def test_logits_extraction():
    num_single_labels = 3
    # type 0: 0 * 3 + 0 = 0, 0 * 3 + 1 = 1, 0 * 3 + 2 = 2 when first; 0 * 3 + 0 = 0, 1 * 3 + 0 = 3, 2 * 3 + 0 = 6 when second
    # type 1: 1 * 3 + 0 = 3, 1 * 3 + 1 = 4, 1 * 3 + 2 = 5 when first; 0 * 3 + 1 = 1, 1 * 3 + 1 = 4, 2 * 3 + 1 = 7 when second
    # type 2: 2 * 3 + 0 = 6, 2 * 3 + 1 = 7, 2 * 3 + 2 = 8 when first; 0 * 3 + 2 = 2, 1 * 3 + 2 = 5, 2 * 3 + 2 = 8 when second
    logits_last_dim = np.array([-11, 1001, -12, -13, -14, -15, -16, -17, -18])

    extracted_for_edge_start = np.zeros((num_single_labels))
    extracted_for_edge_end = np.zeros((num_single_labels))

    indices_affecting_first_type, indices_affecting_second_type = compute_indices_affecting_edge_types(num_single_labels)

    for i in range(len(indices_affecting_first_type)):
        extracted_for_edge_start[i] += np.sum(logits_last_dim[..., indices_affecting_first_type[i]])
        extracted_for_edge_end[i] += np.sum(logits_last_dim[..., indices_affecting_second_type[i]])

    expected_extracted_for_edge_start = np.array(
        [
            logits_last_dim[0 * 3 + 0] + logits_last_dim[0 * 3 + 1] + logits_last_dim[0 * 3 + 2],
            logits_last_dim[1 * 3 + 0] + logits_last_dim[1 * 3 + 1] + logits_last_dim[1 * 3 + 2],
            logits_last_dim[2 * 3 + 0] + logits_last_dim[2 * 3 + 1] + logits_last_dim[2 * 3 + 2]
        ]
    )
    expected_extracted_for_edge_end = np.array(
        [
            logits_last_dim[0 * 3 + 0] + logits_last_dim[1 * 3 + 0] + logits_last_dim[2 * 3 + 0],
            logits_last_dim[0 * 3 + 1] + logits_last_dim[1 * 3 + 1] + logits_last_dim[2 * 3 + 1],
            logits_last_dim[0 * 3 + 2] + logits_last_dim[1 * 3 + 2] + logits_last_dim[2 * 3 + 2]
        ]
    )
    assert np.isclose(expected_extracted_for_edge_start, extracted_for_edge_start).all()
    assert np.isclose(expected_extracted_for_edge_end, extracted_for_edge_end).all()
    print("test_logits_extraction: OK")

def test_logits_residuewise_from_logits_pairwise():
    batch1_multiplier = 666
    mask_batch0 = [1, 1, 1, 1, 0]
    # [batch_size, n_nodes]
    mask_for_test = np.array(
        [
            mask_batch0,
            copy.deepcopy(mask_batch0)
        ]
    )

    # n_neighbours = k = 2
    E_idx_batch0 =  \
        [
            [1, 2],  # nearest for 0
            [0, 2],  # nearest for 1
            [1, 3],  # nearest for 2
            [2, 1],  # nearest for 3
            [0, 1]  # np.arange(2)
        ]
    # [batch_size, n_nodes, n_neighbours)]
    E_idx_for_test = np.array(
        [
            E_idx_batch0,
            copy.deepcopy(E_idx_batch0)
        ]
    )
    logits_pairwise_batch0 =  \
        [
            [
                [-11, -12, -13, -14, -15, 1001, -16, -17, -18],  # 1 * 3 + 2 = 5  # [1, 2],  #  0 -> 1
                [-10, -10, -10, -10, -10, 1002, -10, -10, -10]  # 1 * 3 + 2 = 5  # [1, 2]  #  0 -> 2
            ],
            [
                [-18, -17, -16, -15, -14, -13, -12, 1003, -11],  # 2 * 3 + 1 = 7  # [2, 1],  #  1 -> 0
                [-22, -33, -44, -55, -66, -77, -88, -99, 1004]  # 2 * 3 + 2 = 8  # [2, 2]  #  1 -> 2
            ],
            [
                [-21, -22, -23, -24, -25, -26, -27, -28, 1005],  # 2 * 3 + 2 = 8  # [2, 2],  #  2 -> 1
                [21, 22, 23, 24, 25, 26, 1006, 27, 28]  # 2 * 3 + 0 = 6  # [2, 0]  #  2 -> 3
            ],
            [
                [21, 22, 1007, 23, 24, 25, 26, 27, 28],  # 0 * 3 + 2 = 2  # [0, 2],  #  3 -> 2
                [-28, -27, 1008, -26, -25, -24, -23, -22, -21]  # 0 * 3 + 2 = 2  # [0, 2]  #  3 -> 1
            ],
            [
                [1337, 2, 3, 4, 5, 6, 7, 8, 9],  # some logits that will be masked,  #  masked 0
                [9, 8, 7, 6, 5, 4, 3, 2, 1]  # some logits that will be masked  #  masked 1
            ]
        ]

    logits_pairwise_for_test = np.array(
        [
            logits_pairwise_batch0,
            copy.deepcopy(mult_nested_lists(logits_pairwise_batch0, batch1_multiplier))
        ]
    )

    num_single_labels = 3

    # ground_truth: batch0: v0=1, v1=2, v2=2, v3=0, v4=masked
    # ground_truth: batch1: v0=0, v1=1, v2=2, v3=1, v4=0

    # type 0: 0 * 3 + 0 = 0, 0 * 3 + 1 = 1, 0 * 3 + 2 = 2 when first; 0 * 3 + 0 = 0, 1 * 3 + 0 = 3, 2 * 3 + 0 = 6 when second
    # type 1: 1 * 3 + 0 = 3, 1 * 3 + 1 = 4, 1 * 3 + 2 = 5 when first; 0 * 3 + 1 = 1, 1 * 3 + 1 = 4, 2 * 3 + 1 = 7 when second
    # type 2: 2 * 3 + 0 = 6, 2 * 3 + 1 = 7, 2 * 3 + 2 = 8 when first; 0 * 3 + 2 = 2, 1 * 3 + 2 = 5, 2 * 3 + 2 = 8 when second
    expected_logits_residuewise_batch0 =  \
        [
            # for v0 based on: v0 -> v1, v0 -> v2, v1 -> v0
            [
                # v0 == 0:
                # v0 -> v1:
                # logits(v0 == 0, v1 == 0 aka v0v1 == 0)
                logits_pairwise_for_test[0, 0, 0, 0 * 3 + 0] \
                # + logits(v0 == 0, v1 == 1 aka v0v1 == 1)
                + logits_pairwise_for_test[0, 0, 0, 0 * 3 + 1] \
                # + logits(v0 == 0, v1 == 2 aka v0v1 == 2)
                + logits_pairwise_for_test[0, 0, 0, 0 * 3 + 2] \
                # v0 -> v2:
                # + logits(v0 == 0, v2 == 0 aka v0v2 == 0)
                + logits_pairwise_for_test[0, 0, 1, 0 * 3 + 0] \
                # + logits(v0 == 0, v2 == 1 aka v0v2 == 1)
                + logits_pairwise_for_test[0, 0, 1, 0 * 3 + 1] \
                # + logits(v0 == 0, v2 == 2 aka v0v2 == 2)
                + logits_pairwise_for_test[0, 0, 1, 0 * 3 + 2] \
                # v1 -> v0:
                # + logits(v1 == 0, v0 == 0 aka v1v0 == 0)
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 0] \
                # + logits(v1 == 1, v0 == 0 aka v1v0 == 3)
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 0] \
                # + logits(v1 == 2, v0 == 0 aka v1v0 == 6)
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 0],
                # v0 == 1:
                # v0 -> v1:
                # logits(v0 == 1, v1 == 0 aka v0v1 == 3)
                logits_pairwise_for_test[0, 0, 0, 1 * 3 + 0] \
                # + logits(v0 == 1, v1 == 1 aka v0v1 == 4)
                + logits_pairwise_for_test[0, 0, 0, 1 * 3 + 1] \
                # + logits(v0 == 1, v1 == 2 aka v0v1 == 5)
                + logits_pairwise_for_test[0, 0, 0, 1 * 3 + 2] \
                # v0 -> v2:
                # + logits(v0 == 1, v2 == 0 aka v0v2 == 0)
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 0] \
                # + logits(v0 == 1, v2 == 1 aka v0v2 == 4)
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 1] \
                # + logits(v0 == 1, v2 == 2 aka v0v2 == 5)
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 2] \
                # v1 -> v0:
                # + logits(v1 == 0, v0 == 1 aka v1v0 == 1)
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 1] \
                # + logits(v1 == 1, v0 == 1 aka v1v0 == 4)
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 1] \
                # + logits(v1 == 2, v0 == 1 aka v1v0 == 7)
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 1],
                # v0 == 2:
                # v0 -> v1:
                # logits(v0 == 2, v1 == 0 aka v0v1 == 6)
                logits_pairwise_for_test[0, 0, 0, 2 * 3 + 0] \
                # + logits(v0 == 2, v1 == 1 aka v0v1 == 7)
                + logits_pairwise_for_test[0, 0, 0, 2 * 3 + 1] \
                # + logits(v0 == 2, v1 == 2 aka v0v1 == 8)
                + logits_pairwise_for_test[0, 0, 0, 2 * 3 + 2] \
                # v0 -> v2:
                # + logits(v0 == 2, v2 == 0 aka v0v2 == 6)
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 0] \
                # + logits(v0 == 2, v2 == 1 aka v0v2 == 7)
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 1] \
                # + logits(v0 == 2, v2 == 2 aka v0v2 == 8)
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 2] \
                # v1 -> v0:
                # + logits(v1 == 0, v0 == 2 aka v1v0 == 2)
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 2] \
                # + logits(v1 == 1, v0 == 2 aka v1v0 == 5)
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 2] \
                # + logits(v1 == 2, v0 == 2 aka v1v0 == 8)
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 2]
            ],
            # for v1 based on: v0 -> v1, v1 -> v0, v1 -> v2, v2 -> v1, v3 -> v1
            [
                # v1 == 0:
                # v0 -> v1:
                logits_pairwise_for_test[0, 0, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 0, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 0, 0, 2 * 3 + 0] \
                # v1 -> v0:
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 0, 0 * 3 + 2] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 2] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 0] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 0],
                # v1 == 1:
                # v0 -> v1:
                logits_pairwise_for_test[0, 0, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 0, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 0, 0, 2 * 3 + 1] \
                # v1 -> v0:
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 0, 1 * 3 + 2] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 2] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 1] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 1],
                # v1 == 2:
                # v0 -> v1:
                logits_pairwise_for_test[0, 0, 0, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 0, 0, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 0, 0, 2 * 3 + 2] \
                # v1 -> v0:
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 0, 2 * 3 + 2] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 2] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 2] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 2]
            ],
            # for v2 based on: v0 -> v2, v1 -> v2, v2 -> v1, v2 -> v3, v3 -> v2
            [
                # v2 == 0:
                # v0 -> v2:
                logits_pairwise_for_test[0, 0, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 0] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 0] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 0, 0 * 3 + 2] \
                # v2 -> v3:
                + logits_pairwise_for_test[0, 2, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 1, 0 * 3 + 2] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 0],
                # v2 == 1:
                # v0 -> v2:
                logits_pairwise_for_test[0, 0, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 1] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 1] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 0, 1 * 3 + 2] \
                # v2 -> v3:
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 2] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 1],
                # v2 == 2:
                # v0 -> v2:
                logits_pairwise_for_test[0, 0, 1, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 0, 1, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 0, 1, 2 * 3 + 2] \
                # v1 -> v2:
                + logits_pairwise_for_test[0, 1, 1, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 1, 1, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 1, 1, 2 * 3 + 2] \
                # v2 -> v1:
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 0, 2 * 3 + 2] \
                # v2 -> v3:
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 2] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 2]
            ],
            # for v3 based on: v2 -> v3, v3 -> v2, v3 -> v1
            [
                # v3 == 0:
                # v2 -> v3:
                logits_pairwise_for_test[0, 2, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 0] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 0, 0 * 3 + 2] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 1, 0 * 3 + 2],
                # v3 == 1:
                # v2 -> v3:
                logits_pairwise_for_test[0, 2, 1, 0 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 1] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 0, 1 * 3 + 2] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 1, 1 * 3 + 2],
                # v3 == 2:
                # v2 -> v3:
                logits_pairwise_for_test[0, 2, 1, 0 * 3 + 2] \
                + logits_pairwise_for_test[0, 2, 1, 1 * 3 + 2] \
                + logits_pairwise_for_test[0, 2, 1, 2 * 3 + 2] \
                # v3 -> v2:
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 0, 2 * 3 + 2] \
                # v3 -> v1:
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 0] \
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 1] \
                + logits_pairwise_for_test[0, 3, 1, 2 * 3 + 2]
            ],
            # 0 instead of residuewise logits for masked
            [
                0,
                0,
                0
            ]
        ]

    expected_logits_residuewise = np.array(
        [
            expected_logits_residuewise_batch0,
            copy.deepcopy(mult_nested_lists(expected_logits_residuewise_batch0, batch1_multiplier))
        ]
    )

    logits_residuewise = build_logits_residuewise(logits_pairwise_for_test, E_idx_for_test, num_single_labels, mask_for_test)

    assert np.isclose(logits_residuewise, expected_logits_residuewise).all()
    print("test_logits_residuewise_from_logits_pairwise: OK")


if __name__ == "__main__":
    test_models_equivalence()
    test_label_convertation()
    test_pairwise_embeddings()
    test_pairwise_sequences()
    test_pairwise_mask()
    test_frequencies_from_logits_pairwise()
    test_logits_extraction()
    test_logits_residuewise_from_logits_pairwise()
