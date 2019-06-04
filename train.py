""" Script to train the selected model """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse
import os
import time
import io
import tempfile
from six.moves import xrange
from absl import app as absl_app
from absl import flags

from data_loader import get_dataset, get_gat_dataset, convert
from src.models import model_params
from src.layers.attention_layer import BahdanauAttention
from src.layers.encoders import GraphEncoder
from src.layers.decoders import Decoder
from src.models import transformer

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

# data arguments
parser = argparse.ArgumentParser(description="Main Arguments")
parser.add_argument(
    '--enc_type', default='rnn', type=str, required=True,
    help='Type of encoder Transformer | gat | rnn')
parser.add_argument(
    '--dec_type', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')
parser.add_argument(
    '--src_path', type=str, required=True, help='Path to source.triple file')
parser.add_argument(
    '--tgt_path', type=str, required=True, help='Path to target.lex file')
parser.add_argument(
    '--graph_adj', type=str, required=False, help='Path to adj matrices of examples')
parser.add_argument(
    '--graph_nodes', type=str, required=False, help='Path to nodes list of each example')
parser.add_argument(
    '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument(
    '--emb_dim', type=int, required=True, help='Embedding dimension')
parser.add_argument(
    '--hidden_size', type=int, required=True, help='Size of hidden layer output')
parser.add_argument(
    '--num_layers', type=int, required=True, help='Number of layers in encoder')
parser.add_argument(
    '--num_heads', type=int, required=True, help='Number of heads in self-attention')
parser.add_argument(
    '--use_bias', type=bool, required=False, help='Add bias or not')
parser.add_argument(
    '--use_edges', type=bool, required=False, help='Add edges to embeddings')
parser.add_argument(
    '--dropout', type=float, required=False, help='Dropout rate')
parser.add_argument(
    '--enc_units', type=int, required=False, help='Number of encoder units')
parser.add_argument(
    '--num_examples', default=None, type=int, required=False,
    help='Number of examples to be processed')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.enc_type == 'gat':
        #dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_tgt_size = get_gat_dataset(args)
        #example_input_batch, example_target_batch= next(iter(dataset))
        encoder = GraphEncoder(args, train=True)
        inputs = tf.random.uniform(shape=[32, 16, 512])
        adj = tf.random.uniform(shape=[32, 16, 16])
        outputs = encoder(inputs, adj)
    else:
        dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_inp_size, vocab_tgt_size = get_dataset(args)
        example_input_batch, example_target_batch= next(iter(dataset))






