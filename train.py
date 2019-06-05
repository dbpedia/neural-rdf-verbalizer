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
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
gpu = GPUs[0]

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

# model paramteres 
parser.add_argument(
    '--enc_type', default='rnn', type=str, required=True,
    help='Type of encoder Transformer | gat | rnn')
parser.add_argument(
    '--dec_type', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')

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
    '--checkpoint', type=int, required=False, help='Save checkpoint every these steps')
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
parser.add_argument(
    '--tensorboard', type=bool, required=False, help='Use tensorboard or not')
parser.add_argument(
    '--colab', type=bool, required=False, help='Use Google-Colab')

# hyper-parameters 
parser.add_argument(
    '--optimizer', type=str, required=False, help='Optimizer that will be used')
parser.add_argument(
    '--loss', type=str, required=False, help='Loss function to calculate loss')
parser.add_argument(
    '--learning_rate', type=float, required=False, help='Learning rate')
parser.add_argument(
    '--scheduler_step', type=int, required=False, help='Step to start learning rate scheduler')

def print_stats():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


def loss_function(real, pred, loss_object):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask 

    return tf.reduce_mean(loss_)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.enc_type == 'gat':
        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, 
        vocab_tgt_size, vocab_nodes_size, target_lang) = get_gat_dataset(args)

        embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim) 
        
        encoder = GraphEncoder(args, train=True)
        decoder = Decoder(vocab_tgt_size, args.emb_dim, args.enc_units, BATCH_SIZE)

        optimizer = tf.train.AdamOptimizer()
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        
        #checkpoint_dir = './training_checkpoints'
        #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        #checkpoint = tf.train.Checkpoint( optimizer=optimizer,
        #                                    encoder= encoder,
        #                                    decoder= decoder)
        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, edges, targ):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(nodes, adj)
                dec_input=tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1)

                # Apply teacher forcing 
                for t in range(1, targ.shape[1]):
                    # pass encoder output to decoder
                    # TO-DO: figure out a way to get graph attention network hidden state
                    #dec_hidden = tf.random.uniform(shape=(BATCH_SIZE, args.enc_units))
                    #print(dec_hidden.shape, enc_hidden.shape)
                    predictions, dec_hidden, _ = decoder(dec_input, enc_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions, loss_object) 
                    #using teacher forcing 
                    dec_input = tf.expand_dims(targ[:, t], 1) 
                    
            batch_loss = (loss / int(targ.shape[1]))
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables) 

            optimizer.apply_gradients(zip(gradients, variables))
            

            return batch_loss
        
        total_loss =0
        for (batch, (adj, nodes, edges, targ)) in enumerate(dataset.take(steps)):
            start = time.time()

            # type cast all tensors for uniformity 
            adj = tf.cast(adj, tf.float32)
            nodes = tf.cast(nodes, tf.float32) 
            edges = tf.cast(edges, tf.float32) 
            targ = tf.cast(targ, tf.float32)

            #embed nodes 
            nodes = embedding(nodes)    
            batch_loss = train_step(adj, nodes, edges, targ)
            
            print('Step {} Loss{:.4f}'.format(batch,
                                                batch_loss.numpy()))
            print_stats()
            
         #   if batch % args.checkpoint == 0:
          #      checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time {} \n'.format(time.time() - start))

    else:
        dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_inp_size, vocab_tgt_size = get_dataset(args)
        example_input_batch, example_target_batch= next(iter(dataset))






