""" File to hold arguments """
import argparse
import io
import os
# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

# model paramteres
parser.add_argument(
    '--enc_type', default='rnn', type=str, required=True,
    help='Type of encoder Transformer | gat | rnn')
parser.add_argument(
    '--dec_type', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')
parser.add_argument(
    '--train', type=bool, required=False, help='In training mode or eval mode')

# Colab options
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use Google Colab or not')

# preprocess arguments
parser.add_argument(
    '--src_path', type=str, required=True, help='Path to source.triple file')
parser.add_argument(
    '--tgt_path', type=str, required=True, help='Path to target.lex file')
parser.add_argument(
    '--graph_adj', type=str, required=False, help='Path to adj matrices of examples')
parser.add_argument(
    '--graph_nodes', type=str, required=False, help='Path to nodes list of each example')
parser.add_argument(
    '--graph_edges', type=str, required=False, help='Path to edge list of each example')

# training parameters
parser.add_argument(
    '--steps', type=int, required=False, help='Number of training steps')
parser.add_argument(
    '--eval_steps', type=int, required=False, help='Evaluate every x steps')
parser.add_argument(
    '--checkpoint', type=int, required=False, help='Save checkpoint every these steps')
parser.add_argument(
    '--checkpoint_dir', type=str, required=False, help='Path to checkpoints')

parser.add_argument(
    '--epochs', type=int, default=None,
        required=False, help='Number of epochs (deprecated)')
parser.add_argument(
    '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument(
    '--emb_dim', type=int, required=True, help='Embedding dimension')
parser.add_argument(
    '--hidden_size', type=int, required=True, help='Size of hidden layer output')
parser.add_argument(
    '--enc_layers', type=int, required=True, help='Number of layers in encoder')
parser.add_argument(
    '--dec_layers', type=int, required=True, help='Number of layers in decoder')
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
parser.add_argument(
    '--tensorboard', type=bool, required=False, help='Use tensorboard or not')
parser.add_argument(
    '--colab', type=bool, required=False, help='Use Google-Colab')

# hyper-parameters
parser.add_argument(
    '--optimizer', type=str, required=False, help='Optimizer that will be used')
parser.add_argument(
    '--alpha', type=float, required=False, default= 0.2, help='Alpha value for LeakyRELU')
parser.add_argument(
    '--loss', type=str, required=False, help='Loss function to calculate loss')
parser.add_argument(
    '--learning_rate', type=float, required=False, help='Learning rate')
parser.add_argument(
    '--decay', type=bool, required=False, help='Use learning rate decay')
parser.add_argument(
    '--decay_rate', type=float, required=False, help='Decay rate ')
parser.add_argument(
    '--decay_steps', type=int, required=False, help='Decay every this steps ')
parser.add_argument(
    '--scheduler_step', type=int, required=False, help='Step to start learning rate scheduler')

#inference parameters

def get_args():
    args = parser.parse_args()
    return args