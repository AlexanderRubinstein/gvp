import tensorflow as tf
import argparse

#tf.debugging.enable_check_numerics()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from datetime import datetime
from datasets import *
import random
import tqdm, sys
import util, pdb
from models import *

def make_model(copy_top_gvp):
    model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))

    optimizer = tf.keras.optimizers.Adam()
    util.load_checkpoint(model, optimizer, os.path.join("..", "models", "cath_pretrained"))

    # return model
    return PairwiseCPDModel(model, num_letters=20, hidden_dim=(16,100), copy_top_gvp=copy_top_gvp)

def apply_model(dataset, model, optimizer, model_id, epoch, train, description):
    loss, acc, confusion = loop_func(dataset, model, train=train, optimizer=optimizer)
    if train:
        util.save_checkpoint(model, optimizer, model_id, epoch)
    print('EPOCH {} {} {:.4f} {:.4f}'.format(epoch, description, loss, acc))
    util.save_confusion(confusion)

def main(
    dataset_file,
    only_eval,
    checkpoint,
    copy_top_gvp
):
    trainset, valset, testset = cath_dataset(1800, jsonl_file=dataset_file) # batch size = 1800 residues
    optimizer = tf.keras.optimizers.Adam()
    model = make_model(copy_top_gvp)

    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        util.load_checkpoint(model, optimizer, checkpoint)


    loop_func = util.loop
    # best_epoch, best_val = 0, np.inf
    epoch = 0

    if only_eval:
        if checkpoint is None:
            raise Exception("Trying to eval without checkpoint")
        apply_model(valset, model, optimizer, model_id, epoch, train=False, description="VAL")
    else:
        model_id = int(datetime.timestamp(datetime.now()))
        NUM_EPOCHS = 100
        # for epoch in range(NUM_EPOCHS):
        while epoch < NUM_EPOCHS:
            # loss, acc, confusion = loop_func(trainset, model, train=True, optimizer=optimizer)
            # util.save_checkpoint(model, optimizer, model_id, epoch)
            # print('EPOCH {} TRAIN {:.4f} {:.4f}'.format(epoch, loss, acc))
            # util.save_confusion(confusion)
            apply_model(trainset, model, optimizer, model_id, epoch, train=True, description="TRAIN")
            apply_model(valset, model, optimizer, model_id, epoch, train=False, description="VAL")

            # loss, acc, confusion = loop_func(valset, model, train=False)

            # if loss < best_val:
            #         best_epoch, best_val = epoch, loss

            # print('EPOCH {} VAL {:.4f} {:.4f}'.format(epoch, loss, acc))
            # util.save_confusion(confusion)
            epoch += 1

        # Test with best validation loss
        path = util.models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
        util.load_checkpoint(model, optimizer, path)

    # loss, acc, confusion = loop_func(testset, model, train=False)
    # print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))
    # util.save_confusion(confusion)
    apply_model(testset, model, optimizer, model_id, epoch, train=False, description="TEST")

def make_parser():
    parser = argparse.ArgumentParser(description='Train/eval script parser')
    parser.add_argument('--dataset_file', type=str,
                    help='dataset file to work with')
    parser.add_argument('--checkpoint', type=str,
                    help='Model checkpoint', default=None)
    parser.add_argument('--only_eval', action='store_true',
                    help='Whether to only eval')
    parser.add_argument('--copy_top_gvp', action='store_true',
                    help='Whether to init top gvp layer with the one from the CPDmodel featurizer')

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(
        dataset_file=args.dataset_file,
        only_eval=args.only_eval,
        checkpoint=args.checkpoint,
        copy_top_gvp=args.copy_top_gvp
    )
