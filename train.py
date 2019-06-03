""" Script to train the selected model """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse
import os
import tempfile
from six.moves import xrange
from absl import app as absl_app
from absl import flags

from data_loader import get_dataset
from src.models import model_params
from src.layers.encoders import Encoder
from src.models import transformer

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

# data arguments
parser = argparse.ArgumentParser(description="Main Arguments")
parser.add_argument(
    '--src_path', type=str, required=True, help='Path to source.triple file')
parser.add_argument(
    '--tgt_path', type=str, required=True, help='Path to target.lex file')
parser.add_argument(
    '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument(
    '--emb_dim', type=int, required=True, help='Embedding dimension')
parser.add_argument(
    '--enc_units', type=int, required=True, help='Number of encoder units')
parser.add_argument(
    '--num_examples', default=None, type=int, required=False,
    help='Number of examples to be processed')

args = parser.parse_args()

dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_inp_size, vocab_tgt_size = get_dataset(args)
example_input_batch, example_target_batch = next(iter(dataset))
encoder = Encoder(vocab_inp_size, args.emb_dim, args.enc_units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))




