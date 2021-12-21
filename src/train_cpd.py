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
from constants import FEATURIZER_PATH

def make_model(pairwise, copy_top_gvp):
    model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))

    if pairwise:

        optimizer = tf.keras.optimizers.Adam()
        util.load_checkpoint(model, optimizer, FEATURIZER_PATH)

        # return model
        return PairwiseCPDModel(model, num_letters=20, hidden_dim=(16,100), copy_top_gvp=copy_top_gvp)

    return model

def apply_model(dataset, model, optimizer, model_id, epoch, train, description, pairwise_logits):
    loss, acc, confusion = util.loop(dataset, model, train=train, optimizer=optimizer, pairwise_logits=pairwise_logits)
    if train:
        util.save_checkpoint(model, optimizer, model_id, epoch)
    print('EPOCH {} {} {:.4f} {:.4f}'.format(epoch, description, loss, acc))
    util.save_confusion(confusion)

def main(
    dataset_file,
    only_eval,
    checkpoint,
    copy_top_gvp,
    pairwise,
    pairwise_logits
):
    if not pairwise and copy_top_gvp:
        raise Exception("Trying to copy top gvp to the original model")

    if pairwise_logits and not pairwise:
        raise Exception("Trying to run train/eval/test loop with pairwise_logits with original model")

    trainset, valset, testset = cath_dataset(1800, jsonl_file=dataset_file) # batch size = 1800 residues
    optimizer = tf.keras.optimizers.Adam()


    model = make_model(pairwise, copy_top_gvp)

    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        util.load_checkpoint(model, optimizer, checkpoint)

    if only_eval:
        if checkpoint is None and not copy_top_gvp:
            raise Exception("Trying to eval with random top_gvp")
        apply_model(
            valset,
            model,
            optimizer,
            model_id=None,
            epoch=None,
            train=False,
            description="VAL",
            pairwise_logits=pairwise_logits
        )
    else:
        model_id = int(datetime.timestamp(datetime.now()))
        NUM_EPOCHS = 100
        for epoch in range(NUM_EPOCHS):

            apply_model(
                trainset,
                model,
                optimizer,
                model_id,
                epoch,
                train=True,
                description="TRAIN",
                pairwise_logits=pairwise_logits
            )
            apply_model(
                valset,
                model,
                optimizer,
                model_id=None,
                epoch=epoch,
                train=False,
                description="VAL",
                pairwise_logits=pairwise_logits
            )

        # Test with best validation loss
        path = util.models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
        util.load_checkpoint(model, optimizer, path)

    apply_model(
        testset,
        model,
        optimizer,
        model_id=None,
        epoch=None,
        train=False,
        description="TEST",
        pairwise_logits=pairwise_logits
    )

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
    parser.add_argument('--pairwise', action="store_true",
                    help='whether to use pairwise model')
    parser.add_argument('--pairwise_logits', action="store_true",
                    help='whether to train for pairwise_logits')
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(
        dataset_file=args.dataset_file,
        only_eval=args.only_eval,
        checkpoint=args.checkpoint,
        copy_top_gvp=args.copy_top_gvp,
        pairwise=args.pairwise,
        pairwise_logits=args.pairwise_logits
    )
