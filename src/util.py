import tensorflow as tf
import numpy as np
import tqdm
import scipy.stats, pdb
from collections import defaultdict

models_dir = '../models/{}_{}'

# Here these lookups are only used for labeling the confusion matrix,
# so feel free to use your own / disregard them
three_to_id = {'CYS': 4, 'ASP': 3, 'SER': 15, 'GLN': 5, 'LYS': 11, 'ILE': 9, 'PRO': 14, 'THR': 16, 'PHE': 13, 'ALA': 0, 'GLY': 7, 'HIS': 8, 'GLU': 6, 'LEU': 10, 'ARG': 1, 'TRP': 17, 'VAL': 19, 'ASN': 2, 'TYR': 18, 'MET': 12}
three_to_one = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
id_to_one = {val : three_to_one[key] for key, val in three_to_id.items()}

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

# Train / val / test loop for protein design
def loop(dataset, model, train=False, optimizer=None, pairwise_logits=False):
    acc_metric.reset_states()
    loss_metric.reset_states()
    if pairwise_logits:
        num_pairwise_labels = model.num_letters * model.num_letters
        mat = np.zeros((num_pairwise_labels, num_pairwise_labels))
    else:
        mat = np.zeros((20, 20))
    for batch in tqdm.tqdm(dataset):
        X, S, M = batch
        if pairwise_logits:
            h_V_stacked, E_idx = model.featurizer.train_embeddings(X, S, M, train=False)
            E_idx = E_idx.numpy()

            h_V_pairwise_numpy = make_h_V_pairwise_redneck(h_V_stacked.numpy(), E_idx)
            S_pairwise_numpy = make_S_pairwise_redneck(S.numpy(), E_idx, model.num_letters)
            M_pairwise_numpy = make_mask_pairwise_redneck(M.numpy(), E_idx)

            # make tf
            h_V_pairwise = tf.convert_to_tensor(h_V_pairwise_numpy)
            S_pairwise = tf.convert_to_tensor(S_pairwise_numpy)
            M_pairwise = tf.convert_to_tensor(M_pairwise_numpy)
            if train:
                print("\npairwise_logits mode in train")
                with tf.GradientTape() as tape:
                    logits_pairwise = model.pairwise_classificator(h_V_pairwise)  # [batch_size, n_nodes, n_neighbours, 400]
                    loss_value = loss_fn(S_pairwise, logits_pairwise, sample_weight=M_pairwise)
            else:
                print("\npairwise_logits mode in eval/test")
                logits_pairwise = model.pairwise_classificator(h_V_pairwise)  # [batch_size, n_nodes, n_neighbours, 400]
                loss_value = loss_fn(S_pairwise, logits_pairwise, sample_weight=M_pairwise)

            final_logits = logits_pairwise
            final_S = S_pairwise
            final_M = M_pairwise
        else:
            if train:
                with tf.GradientTape() as tape:
                    logits = model(*batch, train=True)
                    loss_value = loss_fn(S, logits, sample_weight=M)
            else:
                logits = model(*batch, train=False)
                loss_value = loss_fn(S, logits, sample_weight=M)
            final_logits = logits
            final_S = S
            final_M = M
        if train:
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc_metric.update_state(final_S, final_logits, sample_weight=M)
        loss_metric.update_state(final_S, final_logits, sample_weight=M)
        pred = tf.math.argmax(final_logits, axis=-1)
        mat += tf.math.confusion_matrix(tf.reshape(final_S, [-1]),
                            tf.reshape(pred, [-1]), weights=tf.reshape(final_M, [-1]))
    loss, acc = loss_metric.result(), acc_metric.result()
    print("len(model.trainable_weights):", len(model.trainable_weights)) # DEBUG
    print("len(model.non_trainable_weights):", len(model.non_trainable_weights)) # DEBUG
    return loss, acc, mat

# Save model and optimizer state
def save_checkpoint(model, optimizer, model_id, epoch):
    path = models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.write(path)
    print('CHECKPOINT SAVED TO ' + path)

# Load model and optimizer state
def load_checkpoint(model, optimizer, path):
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(path).expect_partial()
    print('CHECKPOINT RESTORED FROM ' + path)

# Pretty print confusion matrix
def save_confusion(mat):
    counts = mat.numpy()
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(20):
        res += '\t{}'.format(id_to_one[i])
    res += '\n'
    for i in range(20):
        res += '{}\t'.format(id_to_one[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)

def repeat_vertical(m, n_times):
    # repeat <n_times> matrices along axis-0
    n_rows = m.shape[0]
    repeated_indices = np.tile(np.arange(n_rows), n_times)
    return m[repeated_indices,:]

def repeat_block_n(a, block_size, n_times):
    # for a 1-dim indices array repeat each <block_size> elements <n_times> times
    if block_size == 1:
        result = np.empty((a.shape[0] * n_times,), dtype=a.dtype)
        for i in range(n_times):
            result[i::n_times] = a
    else:
        a = a.reshape(-1, block_size)
        result = np.empty((a.shape[0] * n_times, a.shape[1]), dtype=a.dtype)
        for i in range(n_times):
            result[i::n_times, :] = a
        result = result.reshape(-1)
    return result

# make embedding for each pair by concatenating two corresponding node embeddings
# [batch_size, n_nodes, n_neighbours, 2 * h_dim)]
def make_h_V_pairwise_redneck(h_V, E_idx):
    h_dim = h_V.shape[-1]
    result_shape = (E_idx.shape) + tuple([2 * h_dim])

    n_neighbours = E_idx.shape[-1]
    result = np.zeros((result_shape[0], result_shape[1] * result_shape[2], result_shape[3]))
    first_indices = repeat_block_n(np.arange(h_V.shape[1]), 1, n_neighbours)

    for t in range(h_V.shape[0]):
        second_indices = E_idx[t, ...].reshape(-1)
        result[t, ..., np.arange(h_dim)] = h_V[t, first_indices, ...].T
        result[t, ..., np.arange(h_dim, 2 * h_dim)] = h_V[t, second_indices, ...].T
    return result.reshape(result_shape)

# write amino types for each two neighbour nodes
# [batch_size, n_nodes, n_neighbours, 2)]
def make_S_pairwise_redneck(S, E_idx, num_single_labels):
    result_shape = (E_idx.shape) + tuple([2])
    result = np.zeros((result_shape[0], result_shape[1] * result_shape[2], result_shape[3]))
    n_neighbours = E_idx.shape[-1]
    first_indices = repeat_block_n(np.arange(S.shape[1]), 1, n_neighbours)
    for t in range(S.shape[0]):
        second_indices = E_idx[t, ...].reshape(-1)
        result[t, ..., 0] = S[t, first_indices, ...].T
        result[t, ..., 1] = S[t, second_indices, ...].T
    # convert label pairs to pairwise_labels
    result = result @ np.array([num_single_labels, 1])[..., None]
    return result.reshape(result_shape[:-1])

# make mask with zeros for pairs that involve masked nodes
# [batch_size, n_nodes, n_neighbours, 1)]
def make_mask_pairwise_redneck(mask, E_idx):
    result_shape = (E_idx.shape)
    result = np.zeros((result_shape[0], result_shape[1] * result_shape[2]))
    n_neighbours = E_idx.shape[-1]
    first_indices = repeat_block_n(np.arange(mask.shape[1]), 1, n_neighbours)
    for t in range(mask.shape[0]):
        second_indices = E_idx[t, ...].reshape(-1)
        result[t, ...] = mask[t, first_indices].T
    return result.reshape(result_shape)

def labels_pair_to_pairwise_label(labels_pair, num_single_labels):
    return labels_pair[0] * num_single_labels + labels_pair[1]

def pairwise_label_to_labels_pair(pairwise_label, num_single_labels):
    return [pairwise_label // num_single_labels, pairwise_label % num_single_labels]

# [batch_size, n_nodes, n_letters = num_single_labels)]
# compute predicted labels frequencies from logits_pairwise for each node
def build_prediction_frequencies_redneck(logits_pairwise, E_idx, num_single_labels, onehot_by_argmax, mask):
    prediction_frequencies = np.zeros(logits_pairwise.shape[:-2] + tuple([num_single_labels]))
    input_shape = logits_pairwise.shape
    if onehot_by_argmax:
        labels_pairwise = np.argmax(logits_pairwise, axis=-1)
    else:
        labels_pairwise = tf.random.categorical(tf.convert_to_tensor(logits_pairwise.reshape(-1, input_shape[-1]), dtype=tf.float32), 1)
        labels_pairwise = (labels_pairwise.numpy()).reshape(input_shape[:-1])

    for batch_idx in range(E_idx.shape[0]):
        for edge_start in range(E_idx.shape[1]):
            if mask[batch_idx, edge_start] == 1:
                for edge_end_idx in range(E_idx.shape[2]):
                    edge_end = E_idx[batch_idx, edge_start, edge_end_idx]

                    edge_pairwise_label = labels_pairwise[batch_idx, edge_start, edge_end_idx]
                    edge_start_type, edge_end_type = pairwise_label_to_labels_pair(edge_pairwise_label, num_single_labels)

                    prediction_frequencies[batch_idx, edge_start, edge_start_type] += 1
                    prediction_frequencies[batch_idx, edge_end, edge_end_type] += 1

    return prediction_frequencies
